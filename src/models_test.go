package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func getModels(t *testing.T, cfg *Config, method string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, modelsPath+"?client_version=0.0.0", nil)
	rec := httptest.NewRecorder()
	modelsHandler(cfg)(rec, req)
	return rec
}

func TestModelsListedInProviderOrderWithRequiredFields(t *testing.T) {
	cfg := &Config{Providers: map[string]ProviderConfig{
		"zen":    {Models: []string{"b", "a"}},
		"ollama": {Models: []string{"glm-5.3:cloud"}},
		"pod":    {},
	}}
	rec := getModels(t, cfg, http.MethodGet)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q", ct)
	}
	var body struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	want := []string{"ollama/glm-5.3:cloud", "zen/b", "zen/a"}
	if len(body.Models) != len(want) {
		t.Fatalf("got %d models, want %d", len(body.Models), len(want))
	}
	for i, slug := range want {
		m := body.Models[i]
		if m["slug"] != slug {
			t.Errorf("model %d slug = %v, want %s", i, m["slug"], slug)
		}
		if m["priority"] != float64(i) {
			t.Errorf("model %d priority = %v", i, m["priority"])
		}
		// Codex fails the whole list if one of these keys is missing.
		for _, key := range []string{"display_name", "description", "supported_reasoning_levels", "shell_type",
			"visibility", "supported_in_api", "availability_nux", "upgrade", "support_verbosity",
			"default_verbosity", "apply_patch_tool_type", "truncation_policy", "experimental_supported_tools"} {
			if _, ok := m[key]; !ok {
				t.Errorf("%s: missing %s", slug, key)
			}
		}
		if m["visibility"] != "list" || m["supported_in_api"] != true {
			t.Errorf("%s: visibility %v, supported_in_api %v", slug, m["visibility"], m["supported_in_api"])
		}
		if levels, ok := m["supported_reasoning_levels"].([]any); !ok || len(levels) != 0 {
			t.Errorf("%s: supported_reasoning_levels = %v, want an empty list", slug, m["supported_reasoning_levels"])
		}
		if tools, ok := m["experimental_supported_tools"].([]any); !ok || len(tools) != 0 {
			t.Errorf("%s: experimental_supported_tools = %v, want an empty list", slug, m["experimental_supported_tools"])
		}
	}
}

func TestModelsEmptyListIsOK(t *testing.T) {
	for name, cfg := range map[string]*Config{
		"no providers": {},
		"no models":    {Providers: map[string]ProviderConfig{"ollama": {}}},
	} {
		rec := getModels(t, cfg, http.MethodGet)
		if rec.Code != http.StatusOK {
			t.Errorf("%s: status = %d", name, rec.Code)
		}
		if got := strings.TrimSpace(rec.Body.String()); got != `{"models":[]}` {
			t.Errorf("%s: body = %s", name, got)
		}
	}
}

func TestModelsOnlyAnswersGet(t *testing.T) {
	rec := getModels(t, &Config{}, http.MethodPost)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d", rec.Code)
	}
	if allow := rec.Header().Get("Allow"); allow != http.MethodGet {
		t.Errorf("Allow = %q", allow)
	}
}

func TestModelsDoNotExposeProviderSettings(t *testing.T) {
	cfg := &Config{Providers: map[string]ProviderConfig{
		"zen": {BaseURL: "https://backend.example", APIKeyEnv: "SECRET_KEY_NAME", Models: []string{"m"}},
	}}
	body := getModels(t, cfg, http.MethodGet).Body.String()
	for _, leak := range []string{"backend.example", "SECRET_KEY_NAME"} {
		if strings.Contains(body, leak) {
			t.Errorf("response contains %q: %s", leak, body)
		}
	}
}

func TestLoadConfigReadsModels(t *testing.T) {
	path := filepath.Join(t.TempDir(), "providers.json")
	data := `{"providers":{"zen":{"type":"zen","models":["a","b"]},"pod":{}}}`
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.Providers["zen"].Models; len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Errorf("zen models = %v", got)
	}
	if got := cfg.Providers["pod"].Models; len(got) != 0 {
		t.Errorf("pod models = %v", got)
	}
}

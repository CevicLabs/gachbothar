package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Fake backends that answer every request with reply, plus readers for what the
// dialects write.

func usageBackend(t *testing.T, reply string) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(reply))
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func usageCall(t *testing.T, p ProviderAdapter) *ChatResponse {
	t.Helper()
	resp, err := p.Call(&ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	return resp
}

func usageOllama(t *testing.T, reply string) ProviderAdapter {
	t.Helper()
	return NewOllamaProvider("ollama", usageBackend(t, reply), "", false, 0)
}

func usageZen(t *testing.T, protocol, reply string) ProviderAdapter {
	t.Helper()
	t.Setenv("USAGE_TEST_KEY", "k")
	p, err := NewZenProvider("zen", usageBackend(t, reply), "USAGE_TEST_KEY", protocol)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func usageWant(t *testing.T, got *Usage, want Usage) {
	t.Helper()
	if got == nil || *got != want {
		t.Errorf("usage = %+v, want %+v", got, want)
	}
}

func usageJSON(t *testing.T, raw string) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		t.Fatalf("%v in %q", err, raw)
	}
	return out
}

// usageLines returns each JSON object in an SSE or newline-delimited body.
func usageLines(t *testing.T, body string) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(strings.TrimPrefix(line, "data: "))
		if line == "" || line == "[DONE]" {
			continue
		}
		out = append(out, usageJSON(t, line))
	}
	return out
}

func usageWrite(t *testing.T, d DialectAdapter, resp *ChatResponse) string {
	t.Helper()
	w := httptest.NewRecorder()
	if err := d.WriteResponse(w, resp); err != nil {
		t.Fatalf("WriteResponse: %v", err)
	}
	return w.Body.String()
}

var usageSample = &Usage{InputTokens: 17, CachedInputTokens: 10, OutputTokens: 6}

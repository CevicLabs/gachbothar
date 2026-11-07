package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// Ollama does NOT pull on demand: /api/chat for a model that is not in the
// local store answers 404 {"error":"model 'x' not found"} and nothing is
// fetched. Measured against a live Ollama daemon, 4 Oct 2026.
// gachbothar is the layer that should notice that specific failure, fetch the
// model once, prove it actually landed, and retry the original call.
//
// "Prove it landed" is not belt-and-braces: a pull that reports success
// without the model appearing in the store has already happened on this
// fleet (ornith-1.5:9b, 25 Sep 2026), so a pull whose exit status is the
// only evidence is not evidence.

// fakeOllama is a stand-in for the daemon. Models in `present` answer
// /api/chat; everything else 404s the way the real daemon does. /api/pull
// adds to `present` unless pullLeavesAbsent is set, which reproduces the
// lying-pull failure above.
type fakeOllama struct {
	mu                 sync.Mutex
	present            map[string]bool
	pullLeavesAbsent   bool
	chatStatusOverride int

	chatCalls atomic.Int32
	pullCalls atomic.Int32
	showCalls atomic.Int32
	pulled    []string
}

func newFakeOllama(present ...string) *fakeOllama {
	f := &fakeOllama{present: map[string]bool{}}
	for _, m := range present {
		f.present[m] = true
	}
	return f
}

func (f *fakeOllama) has(model string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.present[model]
}

func (f *fakeOllama) server() *httptest.Server {
	mux := http.NewServeMux()

	mux.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		f.chatCalls.Add(1)
		var body struct{ Model string }
		json.NewDecoder(r.Body).Decode(&body)
		if f.chatStatusOverride != 0 {
			w.WriteHeader(f.chatStatusOverride)
			fmt.Fprint(w, `{"error":"internal server error"}`)
			return
		}
		if !f.has(body.Model) {
			w.WriteHeader(http.StatusNotFound)
			// Exactly the real daemon's body, single-quoted inside valid
			// JSON: {"error":"model 'x' not found"}
			fmt.Fprintf(w, `{"error":"model '%s' not found"}`, body.Model)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"message":{"role":"assistant","content":"OK"}}`)
	})

	mux.HandleFunc("/api/pull", func(w http.ResponseWriter, r *http.Request) {
		f.pullCalls.Add(1)
		var body struct {
			Model string `json:"model"`
			Name  string `json:"name"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		model := body.Model
		if model == "" {
			model = body.Name
		}
		f.mu.Lock()
		f.pulled = append(f.pulled, model)
		if !f.pullLeavesAbsent {
			f.present[model] = true
		}
		f.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"status":"success"}`)
	})

	// Either /api/show or /api/tags is a fair way to confirm the model
	// landed, so the fake answers both and counts them together.
	mux.HandleFunc("/api/show", func(w http.ResponseWriter, r *http.Request) {
		f.showCalls.Add(1)
		var body struct {
			Model string `json:"model"`
			Name  string `json:"name"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		model := body.Model
		if model == "" {
			model = body.Name
		}
		if !f.has(model) {
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"model not found"}`)
			return
		}
		fmt.Fprint(w, `{"details":{}}`)
	})

	mux.HandleFunc("/api/tags", func(w http.ResponseWriter, r *http.Request) {
		f.showCalls.Add(1)
		f.mu.Lock()
		names := make([]map[string]string, 0, len(f.present))
		for m := range f.present {
			names = append(names, map[string]string{"name": m})
		}
		f.mu.Unlock()
		json.NewEncoder(w).Encode(map[string]any{"models": names})
	})

	return httptest.NewServer(mux)
}

// providerFromJSON builds the provider the way main.go does - through
// providers.json - so the test never has to name a Go symbol that does not
// exist yet, and the config surface (`auto_pull`) is itself under test.
func providerFromJSON(t *testing.T, cfgJSON string) ProviderAdapter {
	t.Helper()
	path := filepath.Join(t.TempDir(), "providers.json")
	if err := os.WriteFile(path, []byte(cfgJSON), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	providers, err := BuildProviders(cfg)
	if err != nil {
		t.Fatalf("BuildProviders: %v", err)
	}
	p, ok := providers["ollama-local"]
	if !ok {
		t.Fatalf("provider ollama-local not built")
	}
	return p
}

func chatFor(model string) *ChatRequest {
	return &ChatRequest{
		Provider: "ollama-local",
		Model:    model,
		Messages: []ChatMessage{{Role: "user", Content: "say OK"}},
	}
}

func TestOllamaAutoPull_PullsMissingModelThenRetries(t *testing.T) {
	fake := newFakeOllama()
	srv := fake.server()
	defer srv.Close()

	p := providerFromJSON(t, fmt.Sprintf(`{"providers":{"ollama-local":{
		"base_url":%q, "api_key_env":"", "auto_pull":true}}}`, srv.URL))

	resp, err := p.Call(chatFor("smollm2:135m"))
	if err != nil {
		t.Fatalf("Call returned error instead of pulling the model: %v", err)
	}
	if resp.Content != "OK" {
		t.Errorf("content = %q, want %q", resp.Content, "OK")
	}
	if got := fake.pullCalls.Load(); got != 1 {
		t.Errorf("/api/pull called %d times, want exactly 1", got)
	}
	if len(fake.pulled) != 1 || fake.pulled[0] != "smollm2:135m" {
		t.Errorf("pulled %v, want [smollm2:135m]", fake.pulled)
	}
	if got := fake.chatCalls.Load(); got != 2 {
		t.Errorf("/api/chat called %d times, want 2 (first 404, retry after pull)", got)
	}
	if got := fake.showCalls.Load(); got < 1 {
		t.Errorf("pull was never verified against the store (/api/show or /api/tags)")
	}
}

func TestOllamaAutoPull_PresentModelIsNotPulled(t *testing.T) {
	fake := newFakeOllama("qwen2.5-3b-instruct-local:latest")
	srv := fake.server()
	defer srv.Close()

	p := providerFromJSON(t, fmt.Sprintf(`{"providers":{"ollama-local":{
		"base_url":%q, "api_key_env":"", "auto_pull":true}}}`, srv.URL))

	if _, err := p.Call(chatFor("qwen2.5-3b-instruct-local:latest")); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if got := fake.pullCalls.Load(); got != 0 {
		t.Errorf("/api/pull called %d times for a model already present, want 0", got)
	}
	if got := fake.chatCalls.Load(); got != 1 {
		t.Errorf("/api/chat called %d times, want 1", got)
	}
}

func TestOllamaAutoPull_DisabledByDefault(t *testing.T) {
	fake := newFakeOllama()
	srv := fake.server()
	defer srv.Close()

	// No auto_pull key at all: a remote backend (ollama.com, a pod) must
	// never have a pull fired at it just because a model name was wrong.
	p := providerFromJSON(t, fmt.Sprintf(`{"providers":{"ollama-local":{
		"base_url":%q, "api_key_env":""}}}`, srv.URL))

	if _, err := p.Call(chatFor("smollm2:135m")); err == nil {
		t.Fatal("Call succeeded with auto_pull off, want the 404 surfaced")
	}
	if got := fake.pullCalls.Load(); got != 0 {
		t.Errorf("/api/pull called %d times with auto_pull off, want 0", got)
	}
}

func TestOllamaAutoPull_PullThatDoesNotLandIsAnError(t *testing.T) {
	fake := newFakeOllama()
	fake.pullLeavesAbsent = true
	srv := fake.server()
	defer srv.Close()

	p := providerFromJSON(t, fmt.Sprintf(`{"providers":{"ollama-local":{
		"base_url":%q, "api_key_env":"", "auto_pull":true}}}`, srv.URL))

	_, err := p.Call(chatFor("ornith-1.5:9b"))
	if err == nil {
		t.Fatal("Call succeeded although the pulled model never landed in the store")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "ornith-1.5:9b") {
		t.Errorf("error does not name the model that failed to land: %v", err)
	}
	if got := fake.pullCalls.Load(); got != 1 {
		t.Errorf("/api/pull called %d times, want 1 (no retry loop)", got)
	}
	if got := fake.chatCalls.Load(); got != 1 {
		t.Errorf("/api/chat called %d times, want 1 (no pointless retry)", got)
	}
}

func TestOllamaAutoPull_NonMissingModelErrorIsNotAPullTrigger(t *testing.T) {
	fake := newFakeOllama()
	fake.chatStatusOverride = http.StatusInternalServerError
	srv := fake.server()
	defer srv.Close()

	p := providerFromJSON(t, fmt.Sprintf(`{"providers":{"ollama-local":{
		"base_url":%q, "api_key_env":"", "auto_pull":true}}}`, srv.URL))

	if _, err := p.Call(chatFor("smollm2:135m")); err == nil {
		t.Fatal("Call succeeded on a 500 from the backend")
	}
	if got := fake.pullCalls.Load(); got != 0 {
		t.Errorf("/api/pull called %d times on a 500, want 0", got)
	}
}

func TestOllamaAutoPull_ConcurrentCallsPullOnce(t *testing.T) {
	fake := newFakeOllama()
	srv := fake.server()
	defer srv.Close()

	p := providerFromJSON(t, fmt.Sprintf(`{"providers":{"ollama-local":{
		"base_url":%q, "api_key_env":"", "auto_pull":true}}}`, srv.URL))

	const n = 4
	var wg sync.WaitGroup
	errs := make([]error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, errs[i] = p.Call(chatFor("smollm2:135m"))
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("concurrent call %d: %v", i, err)
		}
	}
	if got := fake.pullCalls.Load(); got != 1 {
		t.Errorf("/api/pull called %d times for %d concurrent callers, want 1", got, n)
	}
}

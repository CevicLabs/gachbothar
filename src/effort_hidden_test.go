package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestHiddenRefusalWithOpaqueBody(t *testing.T) {
	for _, route := range effortRoutes() {
		url, calls := effortServer(t, func(n int) (int, string) {
			if n == 1 {
				return http.StatusBadRequest, `{"model":"m"}`
			}
			return http.StatusOK, route.reply
		})
		if _, err := route.build(t, url).Call(effortRequest("max")); err != nil {
			t.Errorf("%s: %v", route.name, err)
		}
		if len(calls.bodies) != 2 {
			t.Errorf("%s: %d calls, want 2", route.name, len(calls.bodies))
		}
	}
}

func TestHiddenRetryKeepsEverythingElse(t *testing.T) {
	for _, route := range effortRoutes() {
		url, calls := effortServer(t, func(n int) (int, string) {
			if n == 1 {
				return http.StatusBadRequest, `{"error":"no"}`
			}
			return http.StatusOK, route.reply
		})
		req := effortRequest("high")
		req.Messages = []ChatMessage{
			{Role: "system", Content: "be brief"},
			{Role: "user", Content: "weather?"},
			{Role: "assistant", ToolCalls: []ToolCall{{ID: "c1", Name: "get", Arguments: []byte(`{"a":1}`)}}},
			{Role: "tool", Content: "14C", ToolCallID: "c1"},
		}
		req.Tools = []ToolSpec{{Type: "function", Function: ToolSpecFunction{Name: "get", Description: "d", Parameters: []byte(`{"type":"object"}`)}}}
		if _, err := route.build(t, url).Call(req); err != nil {
			t.Errorf("%s: %v", route.name, err)
			continue
		}
		if len(calls.bodies) != 2 {
			t.Errorf("%s: %d calls, want 2", route.name, len(calls.bodies))
			continue
		}
		first := map[string]any{}
		for k, v := range calls.bodies[0] {
			if k != route.field {
				first[k] = v
			}
		}
		if !reflect.DeepEqual(first, calls.bodies[1]) {
			t.Errorf("%s: retry differs beyond effort:\n first %v\nsecond %v", route.name, first, calls.bodies[1])
		}
	}
}

func TestHiddenSecondFailureIsReturned(t *testing.T) {
	for _, route := range effortRoutes() {
		for _, second := range []int{http.StatusBadRequest, http.StatusServiceUnavailable} {
			url, calls := effortServer(t, func(n int) (int, string) {
				if n == 1 {
					return http.StatusBadRequest, `{"error":"first refusal"}`
				}
				return second, `{"error":"second refusal"}`
			})
			_, err := route.build(t, url).Call(effortRequest("high"))
			if err == nil || !strings.Contains(err.Error(), "second refusal") {
				t.Errorf("%s then %d: err = %v, want the second failure", route.name, second, err)
			}
			if len(calls.bodies) != 2 {
				t.Errorf("%s then %d: %d calls, want 2", route.name, second, len(calls.bodies))
			}
		}
	}
}

func TestHiddenNoRetryWhenNothingWasSent(t *testing.T) {
	for _, route := range effortRoutes() {
		efforts := []string{""}
		if route.want("none") == nil {
			efforts = append(efforts, "none")
		}
		for _, effort := range efforts {
			url, calls := effortServer(t, effortFails(http.StatusBadRequest))
			if _, err := route.build(t, url).Call(effortRequest(effort)); err == nil {
				t.Errorf("%s effort %q: no error on a 400", route.name, effort)
			}
			if len(calls.bodies) != 1 {
				t.Errorf("%s effort %q: %d calls, want 1", route.name, effort, len(calls.bodies))
			}
		}
	}
}

func TestHiddenOllamaNoneRefusedIsRetried(t *testing.T) {
	route := effortRoutes()[2]
	url, calls := effortServer(t, func(n int) (int, string) {
		if n == 1 {
			return http.StatusBadRequest, `{"error":"no"}`
		}
		return http.StatusOK, route.reply
	})
	if _, err := route.build(t, url).Call(effortRequest("none")); err != nil {
		t.Fatal(err)
	}
	if len(calls.bodies) != 2 {
		t.Fatalf("%d calls, want 2", len(calls.bodies))
	}
	effortField(t, "second call", calls.bodies[1], "think", nil)
}

func TestHiddenCallerRequestUntouched(t *testing.T) {
	for _, route := range effortRoutes() {
		url, _ := effortServer(t, func(n int) (int, string) {
			if n == 1 {
				return http.StatusBadRequest, `{"error":"no"}`
			}
			return http.StatusOK, route.reply
		})
		req := effortRequest("high")
		if _, err := route.build(t, url).Call(req); err != nil {
			t.Errorf("%s: %v", route.name, err)
		}
		if req.Effort != "high" {
			t.Errorf("%s: caller's Effort became %q", route.name, req.Effort)
		}
	}
}

// A model pulled on demand may still refuse thinking: effort must survive the
// pull, and the refusal after it must still be retried without effort.
func TestHiddenEffortAfterAutoPull(t *testing.T) {
	var thinks []any
	mux := http.NewServeMux()
	mux.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		thinks = append(thinks, body["think"])
		switch len(thinks) {
		case 1:
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"model 'm' not found"}`)
		case 2:
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, `{"error":"\"m\" does not support thinking"}`)
		default:
			fmt.Fprint(w, `{"message":{"role":"assistant","content":"OK"},"done":true}`)
		}
	})
	present := false
	mux.HandleFunc("/api/pull", func(w http.ResponseWriter, r *http.Request) {
		present = true
		fmt.Fprint(w, `{"status":"success"}`)
	})
	mux.HandleFunc("/api/show", func(w http.ResponseWriter, r *http.Request) {
		if !present {
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"model not found"}`)
			return
		}
		fmt.Fprint(w, `{"details":{}}`)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	resp, err := NewOllamaProvider("ollama", srv.URL, "", true, 0).Call(effortRequest("high"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "OK" {
		t.Errorf("content = %q", resp.Content)
	}
	want := []any{"high", "high", nil}
	if !reflect.DeepEqual(thinks, want) {
		t.Errorf("think on each chat call = %v, want %v", thinks, want)
	}
}

func TestHiddenResponsesToleratesOddReasoning(t *testing.T) {
	for _, odd := range []string{`"high"`, `5`, `null`, `[]`, `{"effort":5}`, `{"effort":null}`, `{"effort":"HIGH "}`} {
		if got := effortParse(t, odd).Effort; got != "" {
			t.Errorf("reasoning %s: Effort = %q, want empty", odd, got)
		}
	}
}

// The whole way in: a Responses request through the dialect to a backend.
func TestHiddenResponsesRequestReachesBackend(t *testing.T) {
	route := effortRoutes()[2]
	url, calls := effortServer(t, func(int) (int, string) { return http.StatusOK, route.reply })
	req := effortParse(t, `{"effort":"medium","summary":"auto"}`)
	if req.Provider != "ollama" || req.Model != "m" {
		t.Errorf("provider %q model %q", req.Provider, req.Model)
	}
	if _, err := route.build(t, url).Call(req); err != nil {
		t.Fatal(err)
	}
	effortField(t, "backend", calls.bodies[0], "think", "medium")
}

func TestHiddenDropIsLoggedWithoutTheBody(t *testing.T) {
	for _, route := range effortRoutes() {
		var logged bytes.Buffer
		log.SetOutput(&logged)
		url, _ := effortServer(t, func(n int) (int, string) {
			if n == 1 {
				return http.StatusBadRequest, `{"error":"BODY-MARKER"}`
			}
			return http.StatusOK, route.reply
		})
		req := effortRequest("xhigh")
		req.Model = "model-marker"
		_, err := route.build(t, url).Call(req)
		log.SetOutput(os.Stderr)
		if err != nil {
			t.Errorf("%s: %v", route.name, err)
		}
		text := logged.String()
		if !strings.Contains(text, "model-marker") {
			t.Errorf("%s: log does not name the model: %q", route.name, text)
		}
		if strings.Contains(text, "BODY-MARKER") {
			t.Errorf("%s: log repeats the backend's reply: %q", route.name, text)
		}
	}
}

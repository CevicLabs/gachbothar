package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// A fake backend that records every request body. answer decides the status and
// reply for call number n, counting from 1.

type effortCalls struct {
	bodies []map[string]any
}

func effortServer(t *testing.T, answer func(n int) (int, string)) (string, *effortCalls) {
	t.Helper()
	calls := &effortCalls{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		calls.bodies = append(calls.bodies, body)
		status, reply := answer(len(calls.bodies))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(reply))
	}))
	t.Cleanup(srv.Close)
	return srv.URL, calls
}

// effortRoute is one backend wire format: how to build its provider, a good
// reply, the top-level field that carries effort, and the value wanted there.
type effortRoute struct {
	name  string
	build func(t *testing.T, url string) ProviderAdapter
	reply string
	field string
	want  func(effort string) any
}

const effortMessagesReply = `{"content":[{"type":"text","text":"OK"}],"usage":{"input_tokens":1,"output_tokens":1}}`

func effortOutputConfig(effort string) any {
	if effort == "" || effort == "none" {
		return nil
	}
	return map[string]any{"effort": effort}
}

func effortRoutes() []effortRoute {
	return []effortRoute{
		{
			name: "anthropic",
			build: func(t *testing.T, url string) ProviderAdapter {
				t.Setenv("EFFORT_TEST_KEY", "k")
				return NewAnthropicProvider("anthropic", url, "EFFORT_TEST_KEY")
			},
			reply: effortMessagesReply,
			field: "output_config",
			want:  effortOutputConfig,
		},
		{
			name:  "zen messages",
			build: func(t *testing.T, url string) ProviderAdapter { return effortZen(t, url, "messages") },
			reply: effortMessagesReply,
			field: "output_config",
			want:  effortOutputConfig,
		},
		{
			name:  "zen chat",
			build: func(t *testing.T, url string) ProviderAdapter { return effortZen(t, url, "chat") },
			reply: `{"choices":[{"message":{"role":"assistant","content":"OK"}}]}`,
			field: "reasoning_effort",
			want: func(effort string) any {
				if effort == "" {
					return nil
				}
				return effort
			},
		},
		{
			name: "ollama",
			build: func(t *testing.T, url string) ProviderAdapter {
				return NewOllamaProvider("ollama", url, "", false, 0)
			},
			reply: `{"message":{"role":"assistant","content":"OK"},"done":true}`,
			field: "think",
			want: func(effort string) any {
				switch effort {
				case "":
					return nil
				case "none":
					return false
				case "xhigh":
					return "high"
				}
				return effort
			},
		},
	}
}

func effortZen(t *testing.T, url, protocol string) ProviderAdapter {
	t.Helper()
	t.Setenv("EFFORT_TEST_KEY", "k")
	p, err := NewZenProvider("zen", url, "EFFORT_TEST_KEY", protocol)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func effortRequest(effort string) *ChatRequest {
	return &ChatRequest{Model: "m", Effort: effort, Messages: []ChatMessage{{Role: "user", Content: "hi"}}}
}

func effortParse(t *testing.T, reasoning string) *ChatRequest {
	t.Helper()
	body := `{"model":"ollama/m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]`
	if reasoning != "" {
		body += `,"reasoning":` + reasoning
	}
	body += "}"
	r := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	req, err := responsesDialect{}.ParseRequest(r)
	if err != nil {
		t.Fatalf("ParseRequest with reasoning %s: %v", reasoning, err)
	}
	return req
}

func effortField(t *testing.T, label string, body map[string]any, field string, want any) {
	t.Helper()
	got, present := body[field]
	if want == nil {
		if present {
			t.Errorf("%s: %s = %v, want it left out", label, field, got)
		}
		return
	}
	if !present || !reflect.DeepEqual(got, want) {
		t.Errorf("%s: %s = %v (present %v), want %v", label, field, got, present, want)
	}
}

func effortFails(n int) func(int) (int, string) {
	return func(int) (int, string) { return n, fmt.Sprintf(`{"error":"refused with %d"}`, n) }
}

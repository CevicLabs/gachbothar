package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestSSEProvidersStreamAndAssembleTools(t *testing.T) {
	tests := []struct {
		name, protocol string
		events         []string
	}{
		{"zen-chat", "chat", []string{
			`{"choices":[{"index":0,"delta":{"content":"hi"}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"id1","function":{"name":"f","arguments":"{\"x\":"}}]}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}`,
			`{"usage":{"prompt_tokens":5,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":2}},"choices":[]}`, `[DONE]`}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			release := make(chan struct{})
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				for i, e := range tt.events {
					fmt.Fprintf(w, "data: %s\n\n", e)
					w.(http.Flusher).Flush()
					if (tt.protocol == "messages" && i == 1) || (tt.protocol == "chat" && i == 0) {
						<-release
					}
				}
			}))
			defer upstream.Close()
			out := make(chan string, 1)
			done := make(chan *ChatResponse, 1)
			errs := make(chan error, 1)
			go func() {
				resp, err := streamSSE(context.Background(), upstream.Client(), upstream.URL, map[string]any{}, &ChatRequest{Model: "m"}, tt.protocol, func(http.Header) {}, func(s string) error { out <- s; return nil })
				if err != nil {
					errs <- err
				} else {
					done <- resp
				}
			}()
			select {
			case got := <-out:
				if got != "hi" {
					t.Fatalf("got %q", got)
				}
			case <-time.After(3 * time.Second):
				close(release)
				t.Fatal("no incremental text")
			}
			close(release)
			select {
			case resp := <-done:
				if resp.Content != "hi" || len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "id1" || resp.ToolCalls[0].Name != "f" || string(resp.ToolCalls[0].Arguments) != `{"x":1}` || resp.Usage == nil || *resp.Usage != (Usage{InputTokens: 5, CachedInputTokens: 2, OutputTokens: 4}) {
					t.Fatalf("bad response: %+v, calls=%+v usage=%+v", resp, resp.ToolCalls, resp.Usage)
				}
			case err := <-errs:
				t.Fatal(err)
			case <-time.After(3 * time.Second):
				t.Fatal("timeout")
			}
		})
	}
}

func TestStreamClientSeesChunkBeforeCompletion(t *testing.T) {
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"message":{"content":"early"},"done":false}`)
		w.(http.Flusher).Flush()
		<-release
		fmt.Fprintln(w, `{"message":{"content":"late"},"done":true}`)
	}))
	defer upstream.Close()
	p := NewOllamaProvider("p", upstream.URL, "", false, 0)
	server := httptest.NewServer(chatHandler(chatCompletionsDialect{}, map[string]ProviderAdapter{"p": p}))
	defer server.Close()
	resp, err := server.Client().Post(server.URL, "application/json", strings.NewReader(`{"model":"p/m","messages":[],"stream":true}`))
	if err != nil {
		close(release)
		t.Fatal(err)
	}
	defer resp.Body.Close()
	buf := make([]byte, 1024)
	n, err := resp.Body.Read(buf)
	if err != nil || !strings.Contains(string(buf[:n]), "early") {
		close(release)
		t.Fatalf("chunk %q: %v", buf[:n], err)
	}
	close(release)
}

func TestStreamTruncationIsNotReportedAsSuccess(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "data: {\"choices\":[]}\n\n") }))
	defer upstream.Close()
	_, err := streamSSE(context.Background(), upstream.Client(), upstream.URL, map[string]any{}, &ChatRequest{}, "chat", func(http.Header) {}, func(string) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "before completion") {
		t.Fatalf("got %v", err)
	}
}

func TestStreamCancellationClosesBackendRequest(t *testing.T) {
	cancelled := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(cancelled)
	}))
	defer upstream.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err := streamSSE(ctx, upstream.Client(), upstream.URL, map[string]any{}, &ChatRequest{}, "chat", func(http.Header) {}, func(string) error { cancel(); return nil })
	if err == nil {
		t.Fatal("expected cancellation")
	}
	select {
	case <-cancelled:
	case <-time.After(3 * time.Second):
		t.Fatal("backend not cancelled")
	}
}

func TestStreamDialectFinishCarriesToolAndUsage(t *testing.T) {
	for _, d := range []DialectAdapter{chatCompletionsDialect{}, responsesDialect{}, ollamaNativeDialect{}} {
		t.Run(d.Name(), func(t *testing.T) {
			w := httptest.NewRecorder()
			s := newStreamWriter(w, d.Name(), "m")
			if err := s.WriteText("hi"); err != nil {
				t.Fatal(err)
			}
			resp := &ChatResponse{Content: "hi", Model: "m", Stream: true, ToolCalls: []ToolCall{{ID: "id", Name: "f", Arguments: []byte(`{"x":1}`)}}, Usage: &Usage{InputTokens: 5, OutputTokens: 4}}
			if err := s.Finish(d, resp); err != nil {
				t.Fatal(err)
			}
			body := w.Body.String()
			for _, want := range []string{"hi", "id", "f"} {
				if !strings.Contains(body, want) {
					t.Fatalf("missing %q: %s", want, body)
				}
			}
			switch d.Name() {
			case "responses":
				if !strings.Contains(body, "response.output_text.delta") || !strings.Contains(body, "response.completed") || !strings.Contains(body, `"input_tokens":5`) {
					t.Fatal(body)
				}
			case "chat_completions":
				if !strings.Contains(body, `"tool_calls"`) || !strings.Contains(body, `"prompt_tokens":5`) || !strings.Contains(body, "[DONE]") {
					t.Fatal(body)
				}
			case "ollama-native":
				if !strings.Contains(body, `"tool_calls"`) || !strings.Contains(body, `"prompt_eval_count":5`) || !strings.Contains(body, `"done":true`) {
					t.Fatal(body)
				}
			}
		})
	}
}

func TestStreamingEffortRetryAndAuth(t *testing.T) {
	for _, protocol := range []string{"chat", "messages"} {
		t.Run(protocol, func(t *testing.T) {
			attempts := 0
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				attempts++
				var b map[string]any
				if err := json.NewDecoder(r.Body).Decode(&b); err != nil {
					t.Error(err)
					return
				}
				if b["stream"] != true {
					t.Error("stream missing")
				}
				if protocol == "chat" {
					if r.Header.Get("Authorization") != "Bearer test-key" {
						t.Error("no auth")
					}
					if (b["reasoning_effort"] != nil) != (attempts == 1) {
						t.Errorf("effort on attempt %d: %v", attempts, b)
					}
				} else {
					if r.Header.Get("x-api-key") != "test-key" {
						t.Error("no auth")
					}
					if (b["output_config"] != nil) != (attempts == 1) {
						t.Errorf("effort on attempt %d: %v", attempts, b)
					}
				}
				if attempts == 1 {
					http.Error(w, "bad effort", 400)
					return
				}
				if protocol == "chat" {
					fmt.Fprint(w, "data: [DONE]\n\n")
				} else {
					fmt.Fprint(w, "data: {\"type\":\"message_stop\"}\n\n")
				}
			}))
			defer upstream.Close()
			t.Setenv("TEST_STREAM_KEY", "test-key")
			req := &ChatRequest{Model: "m", Effort: "high"}
			z, err := NewZenProvider("z", upstream.URL, "TEST_STREAM_KEY", protocol)
			if err != nil {
				t.Fatal(err)
			}
			var p StreamingProvider = z
			if _, err := p.Stream(context.Background(), req, func(string) error { return nil }); err != nil {
				t.Fatal(err)
			}
			if attempts != 2 || req.Effort != "high" {
				t.Fatalf("attempts=%d request=%+v", attempts, req)
			}
		})
	}
}

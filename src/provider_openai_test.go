package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

func newTestOpenAI(t *testing.T, baseURL, apiKeyEnv string) ProviderAdapter {
	t.Helper()
	build, ok := providerBuilders["openai"]
	if !ok {
		t.Fatal(`providerBuilders has no "openai" type`)
	}
	p, err := build("openai", ProviderConfig{BaseURL: baseURL, APIKeyEnv: apiKeyEnv, Type: "openai"})
	if err != nil {
		t.Fatalf("build openai provider: %v", err)
	}
	return p
}

type openaiCapture struct {
	path    string
	headers http.Header
	body    map[string]any
}

func openaiServer(t *testing.T, answer func(n int) (int, string)) (*httptest.Server, *openaiCapture) {
	t.Helper()
	got := &openaiCapture{}
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		got.path = r.URL.Path
		got.headers = r.Header.Clone()
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &got.body); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		status, reply := answer(calls)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		io.WriteString(w, reply)
	}))
	t.Cleanup(srv.Close)
	return srv, got
}

func openaiCall(t *testing.T, srv *httptest.Server, req *ChatRequest) (*ChatResponse, error) {
	t.Helper()
	t.Setenv("TEST_OPENAI_KEY", "openai-test")
	return newTestOpenAI(t, srv.URL, "TEST_OPENAI_KEY").Call(req)
}

func openaiWireMessages(t *testing.T, got *openaiCapture) []map[string]any {
	t.Helper()
	raw, ok := got.body["messages"].([]any)
	if !ok {
		t.Fatalf("messages is not an array: %v", got.body["messages"])
	}
	out := make([]map[string]any, 0, len(raw))
	for _, m := range raw {
		out = append(out, m.(map[string]any))
	}
	return out
}

func TestOpenAIRegistersAndPostsToVersionedBaseURL(t *testing.T) {
	srv, got := openaiServer(t, func(int) (int, string) {
		return http.StatusOK, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`
	})
	resp, err := openaiCall(t, srv, &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if got.path != "/chat/completions" {
		t.Errorf("path = %q, want /chat/completions directly under base_url", got.path)
	}
	if resp.Content != "ok" || resp.Model != "" {
		t.Errorf("response = %+v", resp)
	}
}

func TestOpenAISendsNoAuthorizationWithoutKey(t *testing.T) {
	srv, got := openaiServer(t, func(int) (int, string) {
		return http.StatusOK, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`
	})
	p := newTestOpenAI(t, srv.URL, "")
	if _, err := p.Call(&ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}}); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if got.headers.Get("Authorization") != "" {
		t.Errorf("Authorization = %q, want it omitted entirely", got.headers.Get("Authorization"))
	}
}

func TestOpenAISendsBearerWhenKeyIsSet(t *testing.T) {
	srv, got := openaiServer(t, func(int) (int, string) {
		return http.StatusOK, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`
	})
	req := &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}}
	if _, err := openaiCall(t, srv, req); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if got.headers.Get("Authorization") != "Bearer openai-test" {
		t.Errorf("Authorization = %q", got.headers.Get("Authorization"))
	}
}

func TestOpenAISharesChatWireWithZen(t *testing.T) {
	srv, got := openaiServer(t, func(int) (int, string) {
		return http.StatusOK, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`
	})
	req := &ChatRequest{
		Model: "m",
		Messages: []ChatMessage{
			{Role: "user", Content: "weather?"},
			{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_7", Name: "get_weather", Arguments: []byte(`{"city":"Dublin"}`)}}},
			{Role: "tool", ToolCallID: "call_7", Content: "12C"},
		},
		Tools: []ToolSpec{{Type: "function", Function: ToolSpecFunction{
			Name:        "get_weather",
			Description: "Get the weather",
			Parameters:  []byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
		}}},
	}
	if _, err := openaiCall(t, srv, req); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if got.body["stream"] != false || got.body["model"] != "m" {
		t.Errorf("model/stream = %v/%v", got.body["model"], got.body["stream"])
	}
	wantTools := []any{map[string]any{"type": "function", "function": map[string]any{
		"name":        "get_weather",
		"description": "Get the weather",
		"parameters": map[string]any{
			"type":       "object",
			"properties": map[string]any{"city": map[string]any{"type": "string"}},
			"required":   []any{"city"},
		},
	}}}
	if !reflect.DeepEqual(got.body["tools"], wantTools) {
		t.Errorf("tools = %#v", got.body["tools"])
	}
	msgs := openaiWireMessages(t, got)
	calls, ok := msgs[1]["tool_calls"].([]any)
	if !ok || len(calls) != 1 {
		t.Fatalf("tool_calls = %#v", msgs[1]["tool_calls"])
	}
	call := calls[0].(map[string]any)
	fn := call["function"].(map[string]any)
	args, ok := fn["arguments"].(string)
	if !ok {
		t.Fatalf("arguments is not a string: %#v", fn["arguments"])
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(args), &decoded); err != nil || decoded["city"] != "Dublin" {
		t.Errorf("arguments = %q (%v)", args, err)
	}
	if msgs[2]["tool_call_id"] != "call_7" || msgs[2]["content"] != "12C" {
		t.Errorf("tool result = %#v", msgs[2])
	}
	if _, present := got.body["stream_options"]; present {
		t.Errorf("buffered body has stream_options: %v", got.body["stream_options"])
	}
}

func TestOpenAIBufferedUsageMapsToNormalizedUsage(t *testing.T) {
	srv, _ := openaiServer(t, func(int) (int, string) {
		return http.StatusOK, `{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":7,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":2}}}`
	})
	resp, err := openaiCall(t, srv, &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	want := &Usage{InputTokens: 7, CachedInputTokens: 2, OutputTokens: 4}
	if resp.Usage == nil || *resp.Usage != *want {
		t.Errorf("usage = %#v, want %#v", resp.Usage, want)
	}
}

func TestOpenAIReturnsProviderNameAndStatus(t *testing.T) {
	srv, _ := openaiServer(t, func(int) (int, string) {
		return http.StatusServiceUnavailable, `{"error":"down"}`
	})
	_, err := openaiCall(t, srv, &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err == nil || !strings.Contains(err.Error(), `"openai"`) || !strings.Contains(err.Error(), "503") || !strings.Contains(err.Error(), "down") {
		t.Fatalf("err = %v, want provider name and status", err)
	}
}

func TestOpenAIBuffersRejectMalformedUsage(t *testing.T) {
	srv, _ := openaiServer(t, func(int) (int, string) {
		return http.StatusOK, `{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":"seven"}}`
	})
	_, err := openaiCall(t, srv, &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err == nil || !strings.Contains(err.Error(), `usage: json: cannot unmarshal string into Go struct field`) {
		t.Fatalf("err = %v, want malformed usage to fail", err)
	}
}

func TestOpenAIStreamSendsIncludeUsageAndMapsUsage(t *testing.T) {
	var streamOptions any
	auth := ""
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		json.Unmarshal(raw, &body)
		streamOptions = body["stream_options"]
		if streamOptions == nil {
			t.Errorf("stream_options missing: %s", raw)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":6,\"completion_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":1}}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	t.Setenv("TEST_OPENAI_KEY", "openai-test")
	p := newTestOpenAI(t, upstream.URL, "TEST_OPENAI_KEY").(StreamingProvider)
	if _, ok := p.(ProviderAdapter); !ok {
		t.Fatal("openai provider is not a ProviderAdapter")
	}
	got := make(chan string, 1)
	done := make(chan *ChatResponse, 1)
	errs := make(chan error, 1)
	go func() {
		resp, err := p.Stream(context.Background(), &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}}, func(s string) error { got <- s; return nil })
		if err != nil {
			errs <- err
			return
		}
		done <- resp
	}()
	select {
	case s := <-got:
		if s != "hi" {
			t.Fatalf("streamed %q", s)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no incremental output")
	}
	select {
	case resp := <-done:
		if resp.Content != "hi" || resp.Usage == nil || *resp.Usage != (Usage{InputTokens: 6, CachedInputTokens: 1, OutputTokens: 3}) {
			t.Fatalf("bad response: %+v usage=%+v", resp, resp.Usage)
		}
	case err := <-errs:
		t.Fatal(err)
	}
	if auth != "Bearer openai-test" {
		t.Errorf("stream Authorization = %q", auth)
	}
	opts, ok := streamOptions.(map[string]any)
	if !ok || opts["include_usage"] != true {
		t.Errorf("stream_options = %#v", streamOptions)
	}
}

func TestOpenAIStreamErrorsNameProviderAndStatus(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"down"}`, http.StatusInternalServerError)
	}))
	defer upstream.Close()
	t.Setenv("TEST_OPENAI_KEY", "openai-test")
	p := newTestOpenAI(t, upstream.URL, "TEST_OPENAI_KEY").(StreamingProvider)
	_, err := p.Stream(context.Background(), &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}}, func(string) error { return nil })
	if err == nil || !strings.Contains(err.Error(), `"openai"`) || !strings.Contains(err.Error(), "500") || !strings.Contains(err.Error(), "down") {
		t.Fatalf("err = %v, want provider name and status", err)
	}
}

func TestOpenAIStreamOmitsAuthorizationWithoutKey(t *testing.T) {
	var auth string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	p := newTestOpenAI(t, upstream.URL, "").(StreamingProvider)
	if _, err := p.Stream(context.Background(), &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}}, func(string) error { return nil }); err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if auth != "" {
		t.Errorf("stream Authorization = %q, want it omitted", auth)
	}
}

func TestOpenAIStreamToolCallsAssembleInOrder(t *testing.T) {
	var chunk1, chunk2 []byte
	{
		first := map[string]any{
			"choices": []map[string]any{
				{"delta": map[string]any{"tool_calls": []map[string]any{
					{"index": 0, "id": "a", "function": map[string]any{"name": "read", "arguments": `{"path":`}},
				}}},
			},
		}
		second := map[string]any{
			"choices": []map[string]any{
				{"delta": map[string]any{"tool_calls": []map[string]any{
					{"index": 0, "function": map[string]any{"arguments": `"/tmp"}`}},
				}}},
			},
		}
		chunk1, _ = json.Marshal(first)
		chunk2, _ = json.Marshal(second)
	}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\n", chunk1)
		fmt.Fprintf(w, "data: %s\n\n", chunk2)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	p := newTestOpenAI(t, upstream.URL, "").(StreamingProvider)
	resp, err := p.Stream(context.Background(), &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "read"}}}, func(string) error { return nil })
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "a" || resp.ToolCalls[0].Name != "read" || string(resp.ToolCalls[0].Arguments) != `{"path":"/tmp"}` {
		t.Errorf("tool calls = %#v", resp.ToolCalls)
	}
}

func TestOpenAIStreamEffortRetryOmitsReasoningEffort(t *testing.T) {
	attempts := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Error(err)
			return
		}
		if (body["reasoning_effort"] != nil) != (attempts == 1) {
			t.Errorf("attempt %d reasoning_effort = %v", attempts, body["reasoning_effort"])
		}
		if attempts == 1 {
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error":"effort unsupported"}`, http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	p := newTestOpenAI(t, upstream.URL, "TEST_OPENAI_KEY").(StreamingProvider)
	req := &ChatRequest{Model: "m", Effort: "high", Messages: []ChatMessage{{Role: "user", Content: "hi"}}}
	if _, err := p.Stream(context.Background(), req, func(string) error { return nil }); err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if attempts != 2 || req.Effort != "high" {
		t.Fatalf("attempts=%d request=%+v", attempts, req)
	}
}

func TestOpenAIOmitsAuthorizationWhenEnvVarUnsetOrEmpty(t *testing.T) {
	for _, tc := range []struct {
		name string
		env  string
	}{
		{"named env var is unset", "GACHBOTHAR_OPENAI_UNSET_KEY"},
		{"named env var is empty", "GACHBOTHAR_OPENAI_EMPTY_KEY"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.name == "named env var is empty" {
				t.Setenv(tc.env, "")
			} else {
				os.Unsetenv(tc.env)
			}
			srv, got := openaiServer(t, func(int) (int, string) {
				return http.StatusOK, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`
			})
			p := newTestOpenAI(t, srv.URL, tc.env)
			if _, err := p.Call(&ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}}); err != nil {
				t.Fatalf("Call: %v", err)
			}
			if got.headers.Get("Authorization") != "" {
				t.Errorf("Authorization = %q, want it omitted when %s", got.headers.Get("Authorization"), tc.name)
			}
		})
	}
}

// truncatedErrorServer replies 503 with a Content-Length longer than the body it
// sends, so the client's body read ends in an unexpected EOF.
func truncatedErrorServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "100")
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte(`{"error":"do`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestFix_OpenAI_TruncatedErrorBodyKeepsProviderAndStatus(t *testing.T) {
	srv := truncatedErrorServer(t)
	_, err := newTestOpenAI(t, srv.URL, "").Call(&ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err == nil || !strings.Contains(err.Error(), `"openai"`) || !strings.Contains(err.Error(), "503") {
		t.Fatalf("err = %v, want provider name and status", err)
	}
}

func TestFix_OpenAIStream_TruncatedErrorBodyKeepsProviderAndStatus(t *testing.T) {
	srv := truncatedErrorServer(t)
	p := newTestOpenAI(t, srv.URL, "").(StreamingProvider)
	_, err := p.Stream(context.Background(), &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}}, func(string) error { return nil })
	if err == nil || !strings.Contains(err.Error(), `"openai"`) || !strings.Contains(err.Error(), "503") {
		t.Fatalf("err = %v, want provider name and status", err)
	}
}

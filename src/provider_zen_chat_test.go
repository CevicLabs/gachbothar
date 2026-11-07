package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// Gate for the Zen "chat" protocol: tools, tool calls and tool results must
// reach the backend in the OpenAI chat wire shape.

const zenPlainReply = `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`

type zenCapture struct {
	path    string
	headers http.Header
	body    map[string]any
}

func zenServer(t *testing.T, reply string) (*httptest.Server, *zenCapture) {
	t.Helper()
	got := &zenCapture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.path = r.URL.Path
		got.headers = r.Header.Clone()
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &got.body); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, reply)
	}))
	t.Cleanup(srv.Close)
	return srv, got
}

func newTestZen(t *testing.T, baseURL string) ProviderAdapter {
	t.Helper()
	t.Setenv("TEST_ZEN_KEY", "zen-test")
	build, ok := providerBuilders["zen"]
	if !ok {
		t.Fatal(`providerBuilders has no "zen" type`)
	}
	p, err := build("zen", ProviderConfig{BaseURL: baseURL, APIKeyEnv: "TEST_ZEN_KEY", Type: "zen", Protocol: "chat"})
	if err != nil {
		t.Fatalf("build zen provider: %v", err)
	}
	return p
}

func zenSend(t *testing.T, req *ChatRequest) *zenCapture {
	t.Helper()
	srv, got := zenServer(t, zenPlainReply)
	if _, err := newTestZen(t, srv.URL).Call(req); err != nil {
		t.Fatalf("Call: %v", err)
	}
	return got
}

func zenWireMessages(t *testing.T, got *zenCapture) []map[string]any {
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

func TestZenChatSendsTools(t *testing.T) {
	got := zenSend(t, &ChatRequest{
		Model:    "some-model",
		Messages: []ChatMessage{{Role: "user", Content: "weather?"}},
		Tools: []ToolSpec{{Type: "function", Function: ToolSpecFunction{
			Name:        "get_weather",
			Description: "Get the weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
		}}},
	})
	if got.path != "/zen/v1/chat/completions" {
		t.Errorf("path = %q", got.path)
	}
	if got.headers.Get("Authorization") != "Bearer zen-test" {
		t.Errorf("Authorization = %q", got.headers.Get("Authorization"))
	}
	if got.body["model"] != "some-model" || got.body["stream"] != false {
		t.Errorf("model/stream = %v/%v", got.body["model"], got.body["stream"])
	}
	want := []any{map[string]any{"type": "function", "function": map[string]any{
		"name":        "get_weather",
		"description": "Get the weather",
		"parameters": map[string]any{
			"type":       "object",
			"properties": map[string]any{"city": map[string]any{"type": "string"}},
			"required":   []any{"city"},
		},
	}}}
	if !reflect.DeepEqual(got.body["tools"], want) {
		t.Errorf("tools = %#v", got.body["tools"])
	}
}

func TestZenChatPlainRequestStaysPlain(t *testing.T) {
	got := zenSend(t, &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if _, present := got.body["tools"]; present {
		t.Errorf("tools sent when there are none: %v", got.body["tools"])
	}
	want := []map[string]any{{"role": "user", "content": "hi"}}
	if msgs := zenWireMessages(t, got); !reflect.DeepEqual(msgs, want) {
		t.Errorf("messages = %#v", msgs)
	}
}

func TestZenChatSendsAssistantToolCalls(t *testing.T) {
	got := zenSend(t, &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "user", Content: "weather?"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_7", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Dublin"}`)}}},
	}})
	msgs := zenWireMessages(t, got)
	if len(msgs) != 2 {
		t.Fatalf("got %d messages", len(msgs))
	}
	calls, ok := msgs[1]["tool_calls"].([]any)
	if !ok || len(calls) != 1 {
		t.Fatalf("tool_calls = %#v", msgs[1]["tool_calls"])
	}
	call := calls[0].(map[string]any)
	if call["id"] != "call_7" || call["type"] != "function" {
		t.Errorf("id/type = %v/%v", call["id"], call["type"])
	}
	fn := call["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("name = %v", fn["name"])
	}
	// The OpenAI wire carries arguments as a JSON-encoded string, not an object.
	args, ok := fn["arguments"].(string)
	if !ok {
		t.Fatalf("arguments is not a string: %#v", fn["arguments"])
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(args), &decoded); err != nil || decoded["city"] != "Dublin" {
		t.Errorf("arguments = %q (%v)", args, err)
	}
}

func TestZenChatSendsToolResults(t *testing.T) {
	got := zenSend(t, &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "tool", ToolCallID: "call_7", Content: "12C"},
	}})
	want := []map[string]any{{"role": "tool", "tool_call_id": "call_7", "content": "12C"}}
	if msgs := zenWireMessages(t, got); !reflect.DeepEqual(msgs, want) {
		t.Errorf("messages = %#v", msgs)
	}
}

func TestZenChatKeepsConversationOrder(t *testing.T) {
	got := zenSend(t, &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "weather?"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c1", Name: "w", Arguments: json.RawMessage(`{}`)}}},
		{Role: "tool", ToolCallID: "c1", Content: "12C"},
		{Role: "assistant", Content: "12 degrees"},
		{Role: "user", Content: "thanks"},
	}})
	msgs := zenWireMessages(t, got)
	roles := make([]string, 0, len(msgs))
	for _, m := range msgs {
		roles = append(roles, m["role"].(string))
	}
	if strings.Join(roles, ",") != "system,user,assistant,tool,assistant,user" {
		t.Fatalf("roles = %v", roles)
	}
	for i, want := range map[int]string{0: "be brief", 1: "weather?", 3: "12C", 4: "12 degrees", 5: "thanks"} {
		if msgs[i]["content"] != want {
			t.Errorf("message %d content = %v, want %q", i, msgs[i]["content"], want)
		}
	}
}

func TestZenChatReturnsToolCalls(t *testing.T) {
	srv, _ := zenServer(t, `{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[
		{"id":"c9","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Cork\"}"}}]}}]}`)
	resp, err := newTestZen(t, srv.URL).Call(&ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if resp.Content != "" || len(resp.ToolCalls) != 1 {
		t.Fatalf("resp = %+v", resp)
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "c9" || tc.Name != "get_weather" || string(tc.Arguments) != `{"city":"Cork"}` {
		t.Errorf("tool call = %+v", tc)
	}
}

func TestZenBufferedChatRejectsMalformedUsage(t *testing.T) {
	_, err := newTestZen(t, usageBackend(t, `{"choices":[{"message":{"role":"assistant","content":"ok"}}],
		"usage":{"prompt_tokens":"seven"}}`)).Call(&ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err == nil || !strings.Contains(err.Error(), "parse zen chat response: parse usage:") {
		t.Errorf("err = %v, want malformed usage to fail", err)
	}
}

func TestZenChatFromChatCompletionsClient(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{
		"model": "zen/m",
		"messages": [
			{"role":"user","content":"weather?"},
			{"role":"assistant","content":null,"tool_calls":[{"id":"c3","type":"function","function":{"name":"w","arguments":"{\"city\":\"Galway\"}"}}]},
			{"role":"tool","tool_call_id":"c3","content":"9C"}
		],
		"tools": [{"type":"function","function":{"name":"w","parameters":{"type":"object"}}}]
	}`))
	req, err := chatCompletionsDialect{}.ParseRequest(r)
	if err != nil {
		t.Fatalf("ParseRequest: %v", err)
	}
	got := zenSend(t, req)
	msgs := zenWireMessages(t, got)
	if len(msgs) != 3 || msgs[2]["tool_call_id"] != "c3" || msgs[1]["tool_calls"] == nil {
		t.Errorf("messages = %#v", msgs)
	}
	if tools, _ := got.body["tools"].([]any); len(tools) != 1 {
		t.Errorf("tools = %#v", got.body["tools"])
	}
}

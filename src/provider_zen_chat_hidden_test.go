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

// Second set for the Zen "chat" gate, with other values and the cases a fix
// shaped only around the visible set would miss.

type zenHiddenCapture struct {
	path    string
	headers http.Header
	body    map[string]any
}

func zenHiddenBackend(t *testing.T) (*httptest.Server, *zenHiddenCapture) {
	t.Helper()
	got := &zenHiddenCapture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.path = r.URL.Path
		got.headers = r.Header.Clone()
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &got.body); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/zen/v1/messages" {
			io.WriteString(w, `{"content":[{"type":"text","text":"ok"}]}`)
			return
		}
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	t.Cleanup(srv.Close)
	return srv, got
}

func zenHiddenCall(t *testing.T, protocol string, req *ChatRequest) *zenHiddenCapture {
	t.Helper()
	srv, got := zenHiddenBackend(t)
	t.Setenv("TEST_ZEN_HIDDEN_KEY", "zen-hidden")
	p, err := providerBuilders["zen"]("zen", ProviderConfig{BaseURL: srv.URL, APIKeyEnv: "TEST_ZEN_HIDDEN_KEY", Type: "zen", Protocol: protocol})
	if err != nil {
		t.Fatalf("build zen provider: %v", err)
	}
	if _, err := p.Call(req); err != nil {
		t.Fatalf("Call: %v", err)
	}
	return got
}

func zenHiddenMessages(t *testing.T, got *zenHiddenCapture) []map[string]any {
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

func zenHiddenCalls(t *testing.T, msg map[string]any) []map[string]any {
	t.Helper()
	raw, ok := msg["tool_calls"].([]any)
	if !ok {
		t.Fatalf("tool_calls = %#v", msg["tool_calls"])
	}
	out := make([]map[string]any, 0, len(raw))
	for _, c := range raw {
		out = append(out, c.(map[string]any))
	}
	return out
}

func TestZenChatKeepsEveryToolCallInOrder(t *testing.T) {
	got := zenHiddenCall(t, "chat", &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "assistant", ToolCalls: []ToolCall{
			{ID: "first", Name: "read_file", Arguments: json.RawMessage(`{"path":"a"}`)},
			{ID: "second", Name: "list_dir", Arguments: json.RawMessage(`{"path":"b"}`)},
		}},
		{Role: "tool", ToolCallID: "first", Content: "A"},
		{Role: "tool", ToolCallID: "second", Content: "B"},
	}})
	msgs := zenHiddenMessages(t, got)
	if len(msgs) != 3 {
		t.Fatalf("got %d messages", len(msgs))
	}
	calls := zenHiddenCalls(t, msgs[0])
	if len(calls) != 2 || calls[0]["id"] != "first" || calls[1]["id"] != "second" {
		t.Fatalf("calls = %#v", calls)
	}
	if calls[1]["function"].(map[string]any)["name"] != "list_dir" {
		t.Errorf("second call = %#v", calls[1])
	}
	if msgs[1]["tool_call_id"] != "first" || msgs[2]["tool_call_id"] != "second" {
		t.Errorf("results = %#v", msgs[1:])
	}
}

func TestZenChatArgumentsSurviveEncoding(t *testing.T) {
	original := `{"q":"say \"hi\" é\n","n":[1,2],"deep":{"x":null}}`
	got := zenHiddenCall(t, "chat", &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c", Name: "n", Arguments: json.RawMessage(original)}}},
	}})
	args, ok := zenHiddenCalls(t, zenHiddenMessages(t, got)[0])[0]["function"].(map[string]any)["arguments"].(string)
	if !ok {
		t.Fatal("arguments is not a string")
	}
	var a, b any
	if err := json.Unmarshal([]byte(args), &a); err != nil {
		t.Fatalf("arguments %q: %v", args, err)
	}
	json.Unmarshal([]byte(original), &b)
	if !reflect.DeepEqual(a, b) {
		t.Errorf("arguments = %q", args)
	}
}

func TestZenChatMissingArgumentsBecomeEmptyObject(t *testing.T) {
	got := zenHiddenCall(t, "chat", &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c", Name: "n"}}},
	}})
	args := zenHiddenCalls(t, zenHiddenMessages(t, got)[0])[0]["function"].(map[string]any)["arguments"]
	if args != "{}" {
		t.Errorf("arguments = %#v, want the string {}", args)
	}
}

func TestZenChatKeepsTextBesideToolCalls(t *testing.T) {
	got := zenHiddenCall(t, "chat", &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "assistant", Content: "checking", ToolCalls: []ToolCall{{ID: "c", Name: "n", Arguments: json.RawMessage(`{}`)}}},
	}})
	msg := zenHiddenMessages(t, got)[0]
	if msg["content"] != "checking" || len(zenHiddenCalls(t, msg)) != 1 {
		t.Errorf("message = %#v", msg)
	}
}

func TestZenChatSendsEveryTool(t *testing.T) {
	got := zenHiddenCall(t, "chat", &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "go"}}, Tools: []ToolSpec{
		{Type: "function", Function: ToolSpecFunction{Name: "alpha", Parameters: json.RawMessage(`{"type":"object","properties":{"a":{"enum":["x","y"]}}}`)}},
		{Type: "function", Function: ToolSpecFunction{Name: "beta", Description: "second"}},
	}})
	want := []any{
		map[string]any{"type": "function", "function": map[string]any{"name": "alpha", "parameters": map[string]any{
			"type": "object", "properties": map[string]any{"a": map[string]any{"enum": []any{"x", "y"}}},
		}}},
		map[string]any{"type": "function", "function": map[string]any{"name": "beta", "description": "second"}},
	}
	if !reflect.DeepEqual(got.body["tools"], want) {
		t.Errorf("tools = %#v", got.body["tools"])
	}
}

// Codex reaches providers through the Responses dialect, so this is the real
// route for its tool loop.
func TestZenChatFromResponsesClient(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model": "zen/m",
		"instructions": "be brief",
		"input": [
			{"type":"message","role":"user","content":[{"type":"input_text","text":"weather?"}]},
			{"type":"function_call","call_id":"call_r","name":"get_weather","arguments":"{\"city\":\"Sligo\"}"},
			{"type":"function_call_output","call_id":"call_r","output":"8C"}
		],
		"tools": [{"type":"function","name":"get_weather","parameters":{"type":"object"}}]
	}`))
	req, err := responsesDialect{}.ParseRequest(r)
	if err != nil {
		t.Fatalf("ParseRequest: %v", err)
	}
	got := zenHiddenCall(t, "chat", req)
	if tools, _ := got.body["tools"].([]any); len(tools) != 1 {
		t.Errorf("tools = %#v", got.body["tools"])
	}
	var callID, resultID string
	for _, m := range zenHiddenMessages(t, got) {
		if m["tool_calls"] != nil {
			callID, _ = zenHiddenCalls(t, m)[0]["id"].(string)
		}
		if m["role"] == "tool" {
			resultID, _ = m["tool_call_id"].(string)
			if m["content"] != "8C" {
				t.Errorf("tool content = %v", m["content"])
			}
		}
	}
	if callID != "call_r" || resultID != "call_r" {
		t.Errorf("call id %q, result id %q", callID, resultID)
	}
}

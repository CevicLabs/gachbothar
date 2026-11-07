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

// Second set for the Zen "messages" gate, with other values and the cases a
// fix shaped only around the visible set would miss.

func zenMsgBackend(t *testing.T, reply string) (*httptest.Server, *string) {
	t.Helper()
	raw := new(string)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		*raw = string(b)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, reply)
	}))
	t.Cleanup(srv.Close)
	return srv, raw
}

func zenMsgCall(t *testing.T, reply string, req *ChatRequest) (*ChatResponse, string, map[string]any) {
	t.Helper()
	srv, raw := zenMsgBackend(t, reply)
	t.Setenv("TEST_ZEN_HIDDEN_KEY", "zen-hidden")
	p, err := providerBuilders["zen"]("zen", ProviderConfig{BaseURL: srv.URL, APIKeyEnv: "TEST_ZEN_HIDDEN_KEY", Type: "zen", Protocol: "messages"})
	if err != nil {
		t.Fatalf("build zen provider: %v", err)
	}
	resp, err := p.Call(req)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal([]byte(*raw), &body); err != nil {
		t.Fatalf("request body: %v", err)
	}
	return resp, *raw, body
}

const zenMsgOK = `{"content":[{"type":"text","text":"ok"}]}`

func zenMsgList(t *testing.T, v any) []map[string]any {
	t.Helper()
	raw, ok := v.([]any)
	if !ok {
		t.Fatalf("not an array: %#v", v)
	}
	out := make([]map[string]any, 0, len(raw))
	for _, x := range raw {
		out = append(out, x.(map[string]any))
	}
	return out
}

func TestZenMessagesKeepsEveryToolUseInOrder(t *testing.T) {
	_, _, body := zenMsgCall(t, zenMsgOK, &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "assistant", ToolCalls: []ToolCall{
			{ID: "first", Name: "read_file", Arguments: json.RawMessage(`{"path":"a"}`)},
			{ID: "second", Name: "list_dir", Arguments: json.RawMessage(`{"path":"b"}`)},
		}},
	}})
	want := []map[string]any{
		{"type": "tool_use", "id": "first", "name": "read_file", "input": map[string]any{"path": "a"}},
		{"type": "tool_use", "id": "second", "name": "list_dir", "input": map[string]any{"path": "b"}},
	}
	blocks := zenMsgList(t, zenMsgList(t, body["messages"])[0]["content"])
	if !reflect.DeepEqual(blocks, want) {
		t.Errorf("blocks = %#v", blocks)
	}
}

func TestZenMessagesMissingArgumentsBecomeEmptyObject(t *testing.T) {
	_, _, body := zenMsgCall(t, zenMsgOK, &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c", Name: "n"}}},
	}})
	input := zenMsgList(t, zenMsgList(t, body["messages"])[0]["content"])[0]["input"]
	if !reflect.DeepEqual(input, map[string]any{}) {
		t.Errorf("input = %#v", input)
	}
}

func TestZenMessagesToolWithoutParametersHasSchema(t *testing.T) {
	_, _, body := zenMsgCall(t, zenMsgOK, &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "go"}}, Tools: []ToolSpec{
		{Type: "function", Function: ToolSpecFunction{Name: "ping"}},
		{Type: "function", Function: ToolSpecFunction{Name: "pong", Parameters: json.RawMessage(`{"type":"object","required":["x"]}`)}},
	}})
	tools := zenMsgList(t, body["tools"])
	if len(tools) != 2 || tools[0]["name"] != "ping" || tools[1]["name"] != "pong" {
		t.Fatalf("tools = %#v", tools)
	}
	if !reflect.DeepEqual(tools[0]["input_schema"], map[string]any{"type": "object"}) {
		t.Errorf("ping schema = %#v", tools[0]["input_schema"])
	}
	if !reflect.DeepEqual(tools[1]["input_schema"], map[string]any{"type": "object", "required": []any{"x"}}) {
		t.Errorf("pong schema = %#v", tools[1]["input_schema"])
	}
}

func TestZenMessagesDeveloperGoesToSystem(t *testing.T) {
	_, _, body := zenMsgCall(t, zenMsgOK, &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "system", Content: "one"},
		{Role: "developer", Content: "two"},
		{Role: "user", Content: "hi"},
	}})
	if body["system"] != "one\n\ntwo" {
		t.Errorf("system = %#v", body["system"])
	}
	if msgs := zenMsgList(t, body["messages"]); len(msgs) != 1 || msgs[0]["role"] != "user" {
		t.Errorf("messages = %#v", msgs)
	}
}

func TestZenMessagesSendsNoCacheControl(t *testing.T) {
	_, raw, _ := zenMsgCall(t, zenMsgOK, &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "system", Content: "s"},
		{Role: "user", Content: "hi"},
	}, Tools: []ToolSpec{{Type: "function", Function: ToolSpecFunction{Name: "n"}}}})
	if strings.Contains(raw, "cache_control") {
		t.Errorf("request has cache_control: %s", raw)
	}
}

func TestZenMessagesReturnsEveryToolCallAndSkipsOtherBlocks(t *testing.T) {
	resp, _, _ := zenMsgCall(t, `{"content":[
		{"type":"thinking","thinking":"hmm"},
		{"type":"text","text":"a"},
		{"type":"tool_use","id":"x","name":"one","input":{"k":1}},
		{"type":"text","text":"b"},
		{"type":"tool_use","id":"y","name":"two","input":{}}
	]}`, &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if resp.Content != "ab" {
		t.Errorf("content = %q", resp.Content)
	}
	if len(resp.ToolCalls) != 2 || resp.ToolCalls[0].ID != "x" || resp.ToolCalls[1].Name != "two" {
		t.Fatalf("calls = %+v", resp.ToolCalls)
	}
	if string(resp.ToolCalls[0].Arguments) != `{"k":1}` {
		t.Errorf("arguments = %s", resp.ToolCalls[0].Arguments)
	}
}

// Codex reaches providers through the Responses dialect, so this is the real
// route for its tool loop.
func TestZenMessagesFromResponsesClient(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model": "zen-anthropic/m",
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
	_, _, body := zenMsgCall(t, zenMsgOK, req)
	msgs := zenMsgList(t, body["messages"])
	if len(msgs) != 3 {
		t.Fatalf("messages = %#v", msgs)
	}
	use := zenMsgList(t, msgs[1]["content"])
	result := zenMsgList(t, msgs[2]["content"])
	if msgs[1]["role"] != "assistant" || use[0]["id"] != "call_r" || !reflect.DeepEqual(use[0]["input"], map[string]any{"city": "Sligo"}) {
		t.Errorf("tool use = %#v", msgs[1])
	}
	if msgs[2]["role"] != "user" || result[0]["tool_use_id"] != "call_r" || result[0]["content"] != "8C" {
		t.Errorf("tool result = %#v", msgs[2])
	}
	if tools := zenMsgList(t, body["tools"]); len(tools) != 1 || tools[0]["name"] != "get_weather" {
		t.Errorf("tools = %#v", tools)
	}
}

// The "chat" protocol is the earlier fix; this one must not disturb it.
func TestZenChatStillUsesItsOwnWire(t *testing.T) {
	srv, raw := zenMsgBackend(t, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	t.Setenv("TEST_ZEN_HIDDEN_KEY", "zen-hidden")
	p, err := providerBuilders["zen"]("zen", ProviderConfig{BaseURL: srv.URL, APIKeyEnv: "TEST_ZEN_HIDDEN_KEY", Type: "zen", Protocol: "chat"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if _, err := p.Call(&ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "tool", ToolCallID: "c", Content: "r"}}}); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if !strings.Contains(*raw, `"tool_call_id":"c"`) || strings.Contains(*raw, "tool_result") {
		t.Errorf("chat body = %s", *raw)
	}
}

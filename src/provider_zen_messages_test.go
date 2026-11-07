package main

import (
	"encoding/json"
	"reflect"
	"testing"
)

// Gate for the Zen "messages" protocol: tools, tool calls and tool results
// must reach the backend in the Anthropic Messages wire shape, and tool_use
// blocks must come back as tool calls.

const zenMessagesPlainReply = `{"content":[{"type":"text","text":"ok"}]}`

func newTestZenMessages(t *testing.T, baseURL string) ProviderAdapter {
	t.Helper()
	t.Setenv("TEST_ZEN_KEY", "zen-test")
	p, err := providerBuilders["zen"]("zen", ProviderConfig{BaseURL: baseURL, APIKeyEnv: "TEST_ZEN_KEY", Type: "zen", Protocol: "messages"})
	if err != nil {
		t.Fatalf("build zen provider: %v", err)
	}
	return p
}

func zenMessagesSend(t *testing.T, req *ChatRequest) *zenCapture {
	t.Helper()
	srv, got := zenServer(t, zenMessagesPlainReply)
	if _, err := newTestZenMessages(t, srv.URL).Call(req); err != nil {
		t.Fatalf("Call: %v", err)
	}
	return got
}

// zenBlocks returns a wire message's content blocks.
func zenBlocks(t *testing.T, msg map[string]any) []map[string]any {
	t.Helper()
	raw, ok := msg["content"].([]any)
	if !ok {
		t.Fatalf("content is not a block array: %#v", msg["content"])
	}
	out := make([]map[string]any, 0, len(raw))
	for _, b := range raw {
		out = append(out, b.(map[string]any))
	}
	return out
}

// zenText reads a wire message's text, which may be a string or text blocks.
func zenText(t *testing.T, msg map[string]any) string {
	t.Helper()
	if s, ok := msg["content"].(string); ok {
		return s
	}
	text := ""
	for _, b := range zenBlocks(t, msg) {
		if b["type"] == "text" {
			text += b["text"].(string)
		}
	}
	return text
}

func TestZenMessagesSendsTools(t *testing.T) {
	got := zenMessagesSend(t, &ChatRequest{
		Model:    "some-model",
		Messages: []ChatMessage{{Role: "user", Content: "weather?"}},
		Tools: []ToolSpec{{Type: "function", Function: ToolSpecFunction{
			Name:        "get_weather",
			Description: "Get the weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
		}}},
	})
	if got.path != "/zen/v1/messages" || got.headers.Get("x-api-key") != "zen-test" || got.headers.Get("anthropic-version") == "" {
		t.Errorf("path %q, headers %v", got.path, got.headers)
	}
	want := []any{map[string]any{
		"name":        "get_weather",
		"description": "Get the weather",
		"input_schema": map[string]any{
			"type":       "object",
			"properties": map[string]any{"city": map[string]any{"type": "string"}},
		},
	}}
	if !reflect.DeepEqual(got.body["tools"], want) {
		t.Errorf("tools = %#v", got.body["tools"])
	}
}

func TestZenMessagesPlainRequestStaysPlain(t *testing.T) {
	got := zenMessagesSend(t, &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "hi"},
	}})
	if _, present := got.body["tools"]; present {
		t.Errorf("tools sent when there are none: %v", got.body["tools"])
	}
	if got.body["system"] != "be brief" {
		t.Errorf("system = %#v", got.body["system"])
	}
	if n, _ := got.body["max_tokens"].(float64); n <= 0 {
		t.Errorf("max_tokens = %v", got.body["max_tokens"])
	}
	msgs := zenWireMessages(t, got)
	if len(msgs) != 1 || msgs[0]["role"] != "user" || zenText(t, msgs[0]) != "hi" {
		t.Errorf("messages = %#v", msgs)
	}
}

func TestZenMessagesSendsToolUse(t *testing.T) {
	got := zenMessagesSend(t, &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "assistant", Content: "checking", ToolCalls: []ToolCall{{ID: "call_7", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Dublin"}`)}}},
	}})
	want := []map[string]any{
		{"type": "text", "text": "checking"},
		{"type": "tool_use", "id": "call_7", "name": "get_weather", "input": map[string]any{"city": "Dublin"}},
	}
	msgs := zenWireMessages(t, got)
	if len(msgs) != 1 || msgs[0]["role"] != "assistant" || !reflect.DeepEqual(zenBlocks(t, msgs[0]), want) {
		t.Errorf("messages = %#v", msgs)
	}
}

func TestZenMessagesSendsToolResults(t *testing.T) {
	got := zenMessagesSend(t, &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "tool", ToolCallID: "call_7", Content: "12C"},
	}})
	want := []map[string]any{{"type": "tool_result", "tool_use_id": "call_7", "content": "12C"}}
	msgs := zenWireMessages(t, got)
	if len(msgs) != 1 || msgs[0]["role"] != "user" || !reflect.DeepEqual(zenBlocks(t, msgs[0]), want) {
		t.Errorf("messages = %#v", msgs)
	}
}

// The Messages API wants all results of one assistant turn in one user message.
func TestZenMessagesGroupsToolResults(t *testing.T) {
	got := zenMessagesSend(t, &ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "assistant", ToolCalls: []ToolCall{
			{ID: "a", Name: "n", Arguments: json.RawMessage(`{}`)},
			{ID: "b", Name: "n", Arguments: json.RawMessage(`{}`)},
		}},
		{Role: "tool", ToolCallID: "a", Content: "1"},
		{Role: "tool", ToolCallID: "b", Content: "2"},
		{Role: "user", Content: "thanks"},
	}})
	msgs := zenWireMessages(t, got)
	if len(msgs) != 3 {
		t.Fatalf("got %d messages, want 3: %#v", len(msgs), msgs)
	}
	results := zenBlocks(t, msgs[1])
	if msgs[1]["role"] != "user" || len(results) != 2 || results[0]["tool_use_id"] != "a" || results[1]["tool_use_id"] != "b" {
		t.Errorf("results = %#v", msgs[1])
	}
	if zenText(t, msgs[2]) != "thanks" {
		t.Errorf("last = %#v", msgs[2])
	}
}

func TestZenMessagesReturnsToolCalls(t *testing.T) {
	srv, _ := zenServer(t, `{"content":[
		{"type":"text","text":"looking"},
		{"type":"tool_use","id":"tu_1","name":"get_weather","input":{"city":"Cork"}}
	]}`)
	resp, err := newTestZenMessages(t, srv.URL).Call(&ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if resp.Content != "looking" || len(resp.ToolCalls) != 1 {
		t.Fatalf("resp = %+v", resp)
	}
	tc := resp.ToolCalls[0]
	var args map[string]any
	if err := json.Unmarshal(tc.Arguments, &args); err != nil || args["city"] != "Cork" || tc.ID != "tu_1" || tc.Name != "get_weather" {
		t.Errorf("tool call = %+v", tc)
	}
}

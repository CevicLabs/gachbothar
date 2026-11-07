package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Checker for tasks/anthropic-provider.md: a direct Anthropic Messages API
// provider with tools both ways. Must fail until the provider exists.

type capturedAnthropic struct {
	path    string
	headers http.Header
	body    map[string]any
}

func anthropicServer(t *testing.T, reply string) (*httptest.Server, *capturedAnthropic) {
	t.Helper()
	got := &capturedAnthropic{}
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

func newTestAnthropic(t *testing.T, baseURL string) ProviderAdapter {
	t.Helper()
	t.Setenv("TEST_ANTHROPIC_KEY", "sk-test")
	build, ok := providerBuilders["anthropic"]
	if !ok {
		t.Fatal(`providerBuilders has no "anthropic" type`)
	}
	p, err := build("anthropic", ProviderConfig{BaseURL: baseURL, APIKeyEnv: "TEST_ANTHROPIC_KEY", Type: "anthropic"})
	if err != nil {
		t.Fatalf("build anthropic provider: %v", err)
	}
	return p
}

func TestAnthropicRegisteredAndRequiresKey(t *testing.T) {
	build, ok := providerBuilders["anthropic"]
	if !ok {
		t.Fatal(`providerBuilders has no "anthropic" type`)
	}
	t.Setenv("EMPTY_KEY", "")
	p, err := build("anthropic", ProviderConfig{BaseURL: "http://127.0.0.1:1", APIKeyEnv: "EMPTY_KEY", Type: "anthropic"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if _, err := p.Call(&ChatRequest{Model: "claude-sonnet-5-5", Messages: []ChatMessage{{Role: "user", Content: "hi"}}}); err == nil {
		t.Fatal("Call with no API key must fail")
	}
}

func TestAnthropicRequestShapeAndHeaders(t *testing.T) {
	srv, got := anthropicServer(t, `{"content":[{"type":"text","text":"OK"}],"stop_reason":"end_turn"}`)
	p := newTestAnthropic(t, srv.URL)
	resp, err := p.Call(&ChatRequest{Model: "claude-sonnet-5-5", Messages: []ChatMessage{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "hi"},
	}})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if got.path != "/v1/messages" {
		t.Errorf("path = %q, want /v1/messages", got.path)
	}
	if got.headers.Get("x-api-key") != "sk-test" {
		t.Errorf("x-api-key header missing or wrong")
	}
	if got.headers.Get("anthropic-version") == "" {
		t.Errorf("anthropic-version header missing")
	}
	if got.body["model"] != "claude-sonnet-5-5" || systemText(got.body["system"]) != "be brief" {
		t.Errorf("model/system wrong: %v / %v", got.body["model"], got.body["system"])
	}
	if mt, _ := got.body["max_tokens"].(float64); mt <= 0 {
		t.Errorf("max_tokens must be set, got %v", got.body["max_tokens"])
	}
	msgs, _ := got.body["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("system must not be sent as a message; messages = %v", msgs)
	}
	if resp.Content != "OK" {
		t.Errorf("Content = %q, want OK", resp.Content)
	}
}

func TestAnthropicToolsBothWays(t *testing.T) {
	reply := `{"content":[{"type":"text","text":"Checking."},{"type":"tool_use","id":"toolu_2","name":"shell","input":{"cmd":"ls"}}],"stop_reason":"tool_use"}`
	srv, got := anthropicServer(t, reply)
	p := newTestAnthropic(t, srv.URL)
	resp, err := p.Call(&ChatRequest{
		Model: "claude-sonnet-5-5",
		Tools: []ToolSpec{{Type: "function", Function: ToolSpecFunction{Name: "shell", Description: "run", Parameters: json.RawMessage(`{"type":"object"}`)}}},
		Messages: []ChatMessage{
			{Role: "user", Content: "list files"},
			{Role: "assistant", Content: "", ToolCalls: []ToolCall{{ID: "toolu_1", Name: "shell", Arguments: json.RawMessage(`{"cmd":"pwd"}`)}}},
			{Role: "tool", Content: "/tmp", ToolCallID: "toolu_1"},
		},
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	// Outgoing: tools as {name, description, input_schema}.
	tools, _ := got.body["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools not forwarded: %v", got.body["tools"])
	}
	tool := tools[0].(map[string]any)
	if tool["name"] != "shell" || tool["input_schema"] == nil {
		t.Errorf("tool shape wrong: %v", tool)
	}
	// Outgoing: assistant tool call -> tool_use block; tool result -> user tool_result block.
	raw, _ := json.Marshal(got.body["messages"])
	s := string(raw)
	for _, want := range []string{`"type":"tool_use"`, `"id":"toolu_1"`, `"type":"tool_result"`, `"tool_use_id":"toolu_1"`} {
		if !strings.Contains(s, want) {
			t.Errorf("messages missing %s: %s", want, s)
		}
	}
	// Incoming: text and tool_use blocks.
	if resp.Content != "Checking." {
		t.Errorf("Content = %q", resp.Content)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "toolu_2" || resp.ToolCalls[0].Name != "shell" || !strings.Contains(string(resp.ToolCalls[0].Arguments), `"ls"`) {
		t.Errorf("ToolCalls = %+v", resp.ToolCalls)
	}
}

func TestAnthropicEmptyToolFieldsAndAssistantContent(t *testing.T) {
	srv, got := anthropicServer(t, `{"content":[],"stop_reason":"end_turn"}`)
	p := newTestAnthropic(t, srv.URL)
	_, err := p.Call(&ChatRequest{
		Model: "claude-sonnet-5-5",
		Tools: []ToolSpec{{Type: "function", Function: ToolSpecFunction{Name: "empty_schema"}}},
		Messages: []ChatMessage{
			{Role: "assistant", Content: ""},
			{Role: "assistant", Content: "", ToolCalls: []ToolCall{{ID: "toolu_empty", Name: "empty_args"}}},
		},
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}

	tools, _ := got.body["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %v, want one", got.body["tools"])
	}
	schema := tools[0].(map[string]any)["input_schema"]
	if schema == nil || schema.(map[string]any)["type"] != "object" {
		t.Errorf("input_schema = %v, want {type: object}", schema)
	}

	msgs, _ := got.body["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %v, want two assistant messages", msgs)
	}
	first := msgs[0].(map[string]any)["content"].([]any)
	if len(first) != 0 {
		t.Errorf("empty assistant content produced blocks: %v", first)
	}
	second := msgs[1].(map[string]any)["content"].([]any)
	if len(second) != 1 {
		t.Fatalf("assistant tool message blocks = %v, want only tool_use", second)
	}
	input, ok := second[0].(map[string]any)["input"]
	if !ok || len(input.(map[string]any)) != 0 {
		t.Errorf("tool input = %v, want empty object", input)
	}
}

func TestAnthropicErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		io.WriteString(w, `{"type":"error","error":{"type":"authentication_error"}}`)
	}))
	t.Cleanup(srv.Close)
	p := newTestAnthropic(t, srv.URL)
	if _, err := p.Call(&ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}}); err == nil {
		t.Fatal("a 401 from the API must be an error")
	}
}

func TestAnthropicDeveloperRoleIsSystem(t *testing.T) {
	// Codex sends its instructions as role "developer" (OpenAI Responses); Anthropic only accepts user/assistant.
	srv, got := anthropicServer(t, `{"content":[{"type":"text","text":"OK"}]}`)
	p := newTestAnthropic(t, srv.URL)
	if _, err := p.Call(&ChatRequest{Model: "m", Messages: []ChatMessage{
		{Role: "system", Content: "base"}, {Role: "developer", Content: "rules"}, {Role: "user", Content: "hi"},
	}}); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if systemText(got.body["system"]) != "base\n\nrules" {
		t.Errorf("system = %q, want base + developer text", got.body["system"])
	}
	if msgs, _ := got.body["messages"].([]any); len(msgs) != 1 {
		t.Errorf("developer must not be sent as a message: %v", got.body["messages"])
	}
}

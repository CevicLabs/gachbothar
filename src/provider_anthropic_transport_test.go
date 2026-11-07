package main

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

type anthropicRoundTrip func(*http.Request) (*http.Response, error)

func (f anthropicRoundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestAnthropicTransportRoundTrip(t *testing.T) {
	t.Setenv("ANTHROPIC_TEST_KEY", "test-key")
	p := NewAnthropicProvider("direct", "https://example.invalid/", "ANTHROPIC_TEST_KEY")
	if p.Name() != "direct" {
		t.Fatal(p.Name())
	}
	p.client.Transport = anthropicRoundTrip(func(r *http.Request) (*http.Response, error) {
		if r.Method != http.MethodPost || r.URL.String() != "https://example.invalid/v1/messages" || r.Header.Get("x-api-key") != "test-key" || r.Header.Get("anthropic-version") != "2023-06-01" || r.Header.Get("Content-Type") != "application/json" {
			t.Fatalf("unexpected request: %+v", r)
		}
		var body struct {
			System    any                `json:"system"`
			MaxTokens int                `json:"max_tokens"`
			Messages  []anthropicMessage `json:"messages"`
			Tools     []anthropicTool    `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if systemText(body.System) != "one\n\ntwo" || body.MaxTokens != 8192 {
			t.Fatalf("system/token settings: %+v", body)
		}
		if len(body.Tools) != 1 || body.Tools[0].Name != "shell" || body.Tools[0].Description != "run" || string(body.Tools[0].InputSchema) != `{"type":"object"}` {
			t.Fatalf("tools: %+v", body.Tools)
		}
		if len(body.Messages) != 4 {
			t.Fatalf("messages: %+v", body.Messages)
		}
		assistant := body.Messages[1]
		if assistant.Role != "assistant" || len(assistant.Content) != 3 || assistant.Content[0].Text != "Checking" || assistant.Content[1].ID != "a" || assistant.Content[2].Name != "shell" || string(assistant.Content[1].Input) != `{"cmd":"pwd"}` {
			t.Fatalf("assistant: %+v", assistant)
		}
		results := body.Messages[2]
		if results.Role != "user" || len(results.Content) != 2 || results.Content[0].Type != "tool_result" || results.Content[0].ToolUseID != "a" || results.Content[1].ToolUseID != "b" || results.Content[1].Content == nil || *results.Content[1].Content != "" {
			t.Fatalf("results: %+v", results)
		}
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"content":[{"type":"text","text":"one"},{"type":"tool_use","id":"c","name":"shell","input":{"cmd":"ls"}},{"type":"text","text":"two"}]}`)), Header: make(http.Header)}, nil
	})
	result, err := p.Call(&ChatRequest{Model: "claude", Tools: []ToolSpec{{Type: "function", Function: ToolSpecFunction{Name: "shell", Description: "run", Parameters: json.RawMessage(`{"type":"object"}`)}}}, Messages: []ChatMessage{
		{Role: "system", Content: "one"}, {Role: "system", Content: "two"}, {Role: "user", Content: "hi"},
		{Role: "assistant", Content: "Checking", ToolCalls: []ToolCall{{ID: "a", Name: "shell", Arguments: json.RawMessage(`{"cmd":"pwd"}`)}, {ID: "b", Name: "shell", Arguments: json.RawMessage(`{}`)}}},
		{Role: "tool", ToolCallID: "a", Content: "/tmp"}, {Role: "tool", ToolCallID: "b", Content: ""}, {Role: "user", Content: "continue"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "onetwo" || len(result.ToolCalls) != 1 || result.ToolCalls[0].ID != "c" || result.ToolCalls[0].Name != "shell" || string(result.ToolCalls[0].Arguments) != `{"cmd":"ls"}` {
		t.Fatalf("response: %+v", result)
	}
}

func TestAnthropicTransportFailures(t *testing.T) {
	t.Setenv("ANTHROPIC_TEST_KEY", "key")
	for _, tc := range []struct {
		name       string
		status     int
		body, want string
	}{
		{"status", 429, "rate limit details", "429: rate limit details"},
		{"malformed", 200, "{", "parse anthropic response"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := NewAnthropicProvider("direct", "https://example.invalid", "ANTHROPIC_TEST_KEY")
			p.client.Transport = anthropicRoundTrip(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: tc.status, Body: io.NopCloser(strings.NewReader(tc.body))}, nil
			})
			if _, err := p.Call(&ChatRequest{Model: "claude"}); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v", err)
			}
		})
	}
	t.Setenv("ANTHROPIC_TEST_KEY", "")
	p := NewAnthropicProvider("direct", "https://example.invalid", "ANTHROPIC_TEST_KEY")
	p.client.Transport = anthropicRoundTrip(func(*http.Request) (*http.Response, error) { t.Fatal("missing key sent a request"); return nil, nil })
	if _, err := p.Call(&ChatRequest{}); err == nil {
		t.Fatal("expected missing-key error")
	}
}

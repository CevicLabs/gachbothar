package main

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

// Checker for tasks/anthropic-cache-usage.md: prompt caching and usage reporting.

func TestAnthropicPromptCaching(t *testing.T) {
	srv, got := anthropicServer(t, `{"content":[{"type":"text","text":"OK"}]}`)
	p := newTestAnthropic(t, srv.URL)
	_, err := p.Call(&ChatRequest{
		Model: "m",
		Tools: []ToolSpec{
			{Type: "function", Function: ToolSpecFunction{Name: "a", Parameters: json.RawMessage(`{"type":"object"}`)}},
			{Type: "function", Function: ToolSpecFunction{Name: "b", Parameters: json.RawMessage(`{"type":"object"}`)}},
		},
		Messages: []ChatMessage{
			{Role: "system", Content: "rules"},
			{Role: "user", Content: "one"},
			{Role: "assistant", Content: "two"},
			{Role: "user", Content: "three"},
		},
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	raw, _ := json.Marshal(got.body)
	s := string(raw)
	if n := strings.Count(s, `"cache_control"`); n < 3 || n > 4 {
		t.Fatalf("want 3-4 cache breakpoints (system, last tool, last message), got %d: %s", n, s)
	}
	// system becomes a block list so it can carry cache_control.
	sys, ok := got.body["system"].([]any)
	if !ok || len(sys) == 0 || sys[len(sys)-1].(map[string]any)["cache_control"] == nil {
		t.Errorf("system must be blocks with cache_control on the last one: %v", got.body["system"])
	}
	tools := got.body["tools"].([]any)
	if tools[len(tools)-1].(map[string]any)["cache_control"] == nil || tools[0].(map[string]any)["cache_control"] != nil {
		t.Errorf("only the last tool carries cache_control: %v", tools)
	}
	msgs := got.body["messages"].([]any)
	last := msgs[len(msgs)-1].(map[string]any)["content"].([]any)
	if last[len(last)-1].(map[string]any)["cache_control"] == nil {
		t.Errorf("last message's last block must carry cache_control: %v", last)
	}
}

func TestAnthropicUsageParsed(t *testing.T) {
	srv, _ := anthropicServer(t, `{"content":[{"type":"text","text":"OK"}],
		"usage":{"input_tokens":10,"cache_creation_input_tokens":5,"cache_read_input_tokens":100,"output_tokens":7}}`)
	p := newTestAnthropic(t, srv.URL)
	resp, err := p.Call(&ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if resp.Usage == nil {
		t.Fatal("Usage not set")
	}
	// Input = uncached + cache writes + cache reads, as OpenAI counts it.
	if resp.Usage.InputTokens != 115 || resp.Usage.CachedInputTokens != 100 || resp.Usage.OutputTokens != 7 {
		t.Errorf("Usage = %+v, want input 115, cached 100, output 7", *resp.Usage)
	}
}

func TestResponsesCompletedCarriesUsage(t *testing.T) {
	rec := httptest.NewRecorder()
	err := responsesDialect{}.WriteResponse(rec, &ChatResponse{Content: "OK",
		Usage: &Usage{InputTokens: 115, CachedInputTokens: 100, OutputTokens: 7}})
	if err != nil {
		t.Fatalf("WriteResponse: %v", err)
	}
	body := rec.Body.String()
	i := strings.Index(body, "event: response.completed")
	if i < 0 {
		t.Fatalf("no response.completed: %s", body)
	}
	data := body[i:]
	data = data[strings.Index(data, "data: ")+6:]
	data = strings.SplitN(data, "\n", 2)[0]
	var ev struct {
		Response struct {
			Usage struct {
				InputTokens        int `json:"input_tokens"`
				InputTokensDetails struct {
					CachedTokens int `json:"cached_tokens"`
				} `json:"input_tokens_details"`
				OutputTokens int `json:"output_tokens"`
				TotalTokens  int `json:"total_tokens"`
			} `json:"usage"`
		} `json:"response"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		t.Fatalf("completed data not JSON: %v: %s", err, data)
	}
	u := ev.Response.Usage
	if u.InputTokens != 115 || u.InputTokensDetails.CachedTokens != 100 || u.OutputTokens != 7 || u.TotalTokens != 122 {
		t.Errorf("completed usage = %+v", u)
	}
}

// systemText reads the Messages "system" field as plain text, whether it was sent
// as a string or (with caching) as a list of text blocks joined by a blank line.
func systemText(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case []any:
		parts := make([]string, 0, len(x))
		for _, b := range x {
			if m, ok := b.(map[string]any); ok {
				if t, ok := m["text"].(string); ok {
					parts = append(parts, t)
				}
			}
		}
		return strings.Join(parts, "\n\n")
	}
	return ""
}

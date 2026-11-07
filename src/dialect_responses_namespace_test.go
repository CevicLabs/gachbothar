package main

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

// Checker for tasks/namespace-tools.md. Codex sends MCP tools grouped as
// {"type":"namespace","name":"mcp__stage","tools":[...]} and expects calls back as
// {"type":"function_call","name":"read","namespace":"mcp__stage",...}. Providers such as
// Anthropic only take flat function tools, so the dialect flattens on the way in and
// splits on the way out. Found 5 Nov 2026: Sonnet via gachbothar never saw MCP tools.

func TestParseRequest_NamespaceToolsFlattened(t *testing.T) {
	body := `{
		"model": "anthropic/claude-sonnet-5-5",
		"input": [{"type":"message","role":"user","content":[{"type":"input_text","text":"go"}]}],
		"tools": [
			{"type":"function","name":"list_mcp_resources","description":"list","parameters":{"type":"object"}},
			{"type":"namespace","name":"mcp__stage","description":"stage tools","tools":[
				{"type":"function","name":"read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}},
				{"type":"function","name":"write","description":"Write a file","parameters":{"type":"object"}}
			]}
		]
	}`
	req := parseResponsesRequest(t, body)
	names := map[string]ToolSpec{}
	for _, tool := range req.Tools {
		names[tool.Function.Name] = tool
	}
	for _, want := range []string{"list_mcp_resources", "mcp__stage__read", "mcp__stage__write"} {
		if _, ok := names[want]; !ok {
			t.Errorf("tool %q missing; got %v", want, keys(names))
		}
	}
	if len(req.Tools) != 3 {
		t.Errorf("want 3 tools (namespace flattened, no namespace entry itself), got %d: %v", len(req.Tools), keys(names))
	}
	read := names["mcp__stage__read"]
	if read.Function.Description != "Read a file" || !strings.Contains(string(read.Function.Parameters), `"path"`) {
		t.Errorf("flattened tool lost its description/parameters: %+v", read.Function)
	}
}

func TestParseRequest_NamespacedCallHistoryFlattened(t *testing.T) {
	body := `{
		"model": "anthropic/claude-sonnet-5-5",
		"input": [
			{"type":"message","role":"user","content":[{"type":"input_text","text":"go"}]},
			{"type":"function_call","name":"read","namespace":"mcp__stage","arguments":"{\"path\":\"src/a.py\"}","call_id":"call_1"},
			{"type":"function_call_output","call_id":"call_1","output":"print(1)"}
		],
		"tools": [{"type":"namespace","name":"mcp__stage","tools":[{"type":"function","name":"read","parameters":{"type":"object"}}]}]
	}`
	req := parseResponsesRequest(t, body)
	var found bool
	for _, m := range req.Messages {
		for _, tc := range m.ToolCalls {
			found = true
			if tc.Name != "mcp__stage__read" || tc.ID != "call_1" {
				t.Errorf("history call = %+v, want name mcp__stage__read id call_1", tc)
			}
		}
	}
	if !found {
		t.Fatal("namespaced function_call missing from the history")
	}
}

func TestWriteResponse_SplitsNamespacedCall(t *testing.T) {
	rec := httptest.NewRecorder()
	err := responsesDialect{}.WriteResponse(rec, &ChatResponse{ToolCalls: []ToolCall{
		{ID: "toolu_1", Name: "mcp__stage__read", Arguments: json.RawMessage(`{"path":"src/a.py"}`)},
		{ID: "toolu_2", Name: "mcp__session_memory__search_memories", Arguments: json.RawMessage(`{"query":"x"}`)},
		{ID: "toolu_3", Name: "list_mcp_resources", Arguments: json.RawMessage(`{}`)},
	}})
	if err != nil {
		t.Fatalf("WriteResponse: %v", err)
	}
	items := map[string]map[string]any{}
	for _, line := range strings.Split(rec.Body.String(), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var ev struct {
			Item map[string]any `json:"item"`
		}
		if json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &ev) == nil && ev.Item["type"] == "function_call" {
			items[ev.Item["call_id"].(string)] = ev.Item
		}
	}
	check := func(id, name, namespace string) {
		it := items[id]
		if it == nil {
			t.Errorf("%s: no function_call emitted", id)
			return
		}
		gotNS, _ := it["namespace"].(string)
		if it["name"] != name || gotNS != namespace {
			t.Errorf("%s: name=%v namespace=%q, want name=%s namespace=%q", id, it["name"], gotNS, name, namespace)
		}
	}
	check("toolu_1", "read", "mcp__stage")
	check("toolu_2", "search_memories", "mcp__session_memory")
	check("toolu_3", "list_mcp_resources", "") // plain tools: no namespace field
}

func keys(m map[string]ToolSpec) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

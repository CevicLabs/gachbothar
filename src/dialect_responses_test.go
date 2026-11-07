package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func parseResponsesRequest(t *testing.T, body string) *ChatRequest {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	req, err := responsesDialect{}.ParseRequest(r)
	if err != nil {
		t.Fatalf("ParseRequest: %v", err)
	}
	return req
}

// This is the actual bug reported live: codex sends tools on /v1/responses,
// qwen3.6:35b never sees them, and narrates a fake tool call in prose instead
// of executing one. Verified the model itself is fine - the same tools
// reached ollama's own /v1/responses and /api/chat correctly and produced a
// real function_call. The gap is entirely in this dialect's ParseRequest,
// which has no "tools" field in its wire struct at all.
func TestParseRequest_ToolsForwarded(t *testing.T) {
	body := `{
		"model": "pod/qwen3.6:35b",
		"input": [{"type":"message","role":"user","content":[{"type":"input_text","text":"weather?"}]}],
		"tools": [{
			"type": "function",
			"name": "get_weather",
			"description": "Get the current weather for a city",
			"parameters": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
		}]
	}`
	req := parseResponsesRequest(t, body)

	if len(req.Tools) != 1 {
		t.Fatalf("expected 1 tool forwarded into ChatRequest.Tools, got %d", len(req.Tools))
	}
	tool := req.Tools[0]
	if tool.Type != "function" {
		t.Errorf("tool.Type = %q, want %q", tool.Type, "function")
	}
	if tool.Function.Name != "get_weather" {
		t.Errorf("tool.Function.Name = %q, want %q", tool.Function.Name, "get_weather")
	}
	if tool.Function.Description != "Get the current weather for a city" {
		t.Errorf("tool.Function.Description = %q", tool.Function.Description)
	}
	var params map[string]any
	if err := json.Unmarshal(tool.Function.Parameters, &params); err != nil {
		t.Fatalf("tool.Function.Parameters is not valid JSON: %v", err)
	}
	if params["type"] != "object" {
		t.Errorf("parameters.type = %v, want object", params["type"])
	}
}

func TestParseRequest_NoTools_StillWorksAsText(t *testing.T) {
	body := `{"model":"pod/qwen3.6:35b","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`
	req := parseResponsesRequest(t, body)
	if len(req.Tools) != 0 {
		t.Errorf("expected no tools, got %d", len(req.Tools))
	}
	if len(req.Messages) != 1 || req.Messages[0].Content != "hi" {
		t.Fatalf("plain text path regressed: %+v", req.Messages)
	}
}

// A real tool-calling turn always arrives as history on the NEXT request:
// codex replays the prior function_call item plus the function_call_output
// (the tool's real result) so the model can continue. Dropping these means
// the model has no idea its own tool call happened or what it returned.
func TestParseRequest_FunctionCallHistoryRoundTrips(t *testing.T) {
	body := `{
		"model": "pod/qwen3.6:35b",
		"input": [
			{"type":"message","role":"user","content":[{"type":"input_text","text":"weather in Dublin?"}]},
			{"type":"function_call","name":"get_weather","arguments":"{\"city\":\"Dublin\"}","call_id":"call_abc123"},
			{"type":"function_call_output","call_id":"call_abc123","output":"14C, overcast"}
		]
	}`
	req := parseResponsesRequest(t, body)

	if len(req.Messages) != 3 {
		t.Fatalf("expected 3 messages (user, assistant tool-call, tool result), got %d: %+v", len(req.Messages), req.Messages)
	}

	assistantMsg := req.Messages[1]
	if len(assistantMsg.ToolCalls) != 1 {
		t.Fatalf("expected the function_call item to become a ToolCall, got %+v", assistantMsg)
	}
	tc := assistantMsg.ToolCalls[0]
	if tc.ID != "call_abc123" || tc.Name != "get_weather" {
		t.Errorf("tool call = %+v, want ID=call_abc123 Name=get_weather", tc)
	}
	var args map[string]any
	if err := json.Unmarshal(tc.Arguments, &args); err != nil || args["city"] != "Dublin" {
		t.Errorf("tool call arguments = %s, want {\"city\":\"Dublin\"}", tc.Arguments)
	}

	resultMsg := req.Messages[2]
	if resultMsg.ToolCallID != "call_abc123" {
		t.Errorf("result message ToolCallID = %q, want call_abc123", resultMsg.ToolCallID)
	}
	if resultMsg.Content != "14C, overcast" {
		t.Errorf("result message Content = %q, want the tool's real output", resultMsg.Content)
	}
}

// WriteResponse's function_call emission already worked before this fix -
// this just locks it in as a regression guard now that ParseRequest changes
// alongside it.
func TestWriteResponse_EmitsFunctionCall(t *testing.T) {
	resp := &ChatResponse{
		Content: "",
		ToolCalls: []ToolCall{
			{ID: "call_xyz", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Dublin"}`)},
		},
	}
	w := httptest.NewRecorder()
	if err := (responsesDialect{}).WriteResponse(w, resp); err != nil {
		t.Fatalf("WriteResponse: %v", err)
	}
	out := w.Body.String()
	if !bytes.Contains([]byte(out), []byte(`"type":"function_call"`)) {
		t.Fatalf("expected a function_call output item in SSE stream, got:\n%s", out)
	}
	if !strings.Contains(out, `"name":"get_weather"`) || !strings.Contains(out, `"call_id":"call_xyz"`) {
		t.Fatalf("function_call item missing name/call_id, got:\n%s", out)
	}
	if !strings.Contains(out, "response.completed") {
		t.Fatalf("missing mandatory response.completed event, got:\n%s", out)
	}
}

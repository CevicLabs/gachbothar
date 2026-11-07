package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// responsesDialect implements the OpenAI Responses API dialect
// (/v1/responses), the wire format Codex CLI requires - this build's
// model-provider-info rejects "chat" outright ("wire_api = \"chat\" is no
// longer supported"), so a user-defined provider pointed at gachbothar has
// no other way in. Confirmed live 30 Sep 2026: without this dialect, codex
// exec -p gachbothar got 404 on /v1/responses.
//
// Tool calling: confirmed live 2 Oct 2026 that omitting "tools" here made
// qwen3.6:35b narrate a fake tool call in prose instead of executing one -
// not a model failure, the model was never offered any tools. The wire
// shape differs from chat_completions: per codex-rs's own
// LoadableToolSpec/ResponsesApiTool (tools/src/responses_api.rs), a tool
// definition is tagged flat on "type" (name/description/parameters sit
// alongside "type":"function", not nested under a "function" key), and
// per protocol/src/models.rs's ResponseItem enum (#[serde(tag = "type",
// rename_all = "snake_case")]), a prior tool call and its result arrive as
// "function_call" / "function_call_output" input items on the NEXT
// request - codex replays them as history so the model knows its own call
// happened and what it returned. Dropping those (as before) leaves the
// model blind to its own tool's result on every later turn.
type responsesDialect struct{}

func (responsesDialect) Name() string { return "responses" }

func (responsesDialect) RoutePath() string { return "/v1/responses" }

type responsesWireContentItem struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// responsesWireInputItem covers every input item type this dialect
// understands. Fields are optional per-type: a "message" item uses
// Role/Content, a "function_call" item uses Name/Arguments/CallID, and a
// "function_call_output" item uses CallID/Output. Output is json.RawMessage
// because codex-rs's FunctionCallOutputPayload serializes as either a plain
// string or an array of content items (protocol/src/models.rs) - decoded by
// hand in ParseRequest rather than given its own wire struct.
type responsesWireInputItem struct {
	Type    string                     `json:"type"`
	Role    string                     `json:"role"`
	Content []responsesWireContentItem `json:"content"`

	Name      string          `json:"name,omitempty"`
	Namespace string          `json:"namespace,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
	CallID    string          `json:"call_id,omitempty"`
	Output    json.RawMessage `json:"output,omitempty"`
}

// responsesWireTool is the Responses API's flat tool-definition shape -
// deliberately NOT the same struct as chat_completions' openAIWireToolCall,
// which nests under "function". See the type's doc comment on
// responsesDialect for the source confirming this.
type responsesWireTool struct {
	Type        string              `json:"type"`
	Name        string              `json:"name"`
	Description string              `json:"description,omitempty"`
	Parameters  json.RawMessage     `json:"parameters,omitempty"`
	Namespace   string              `json:"namespace,omitempty"`
	Tools       []responsesWireTool `json:"tools,omitempty"`
}

// functionCallOutputText extracts the human-readable text a tool result
// carries, matching FunctionCallOutputBody's two wire shapes: a plain
// string, or an array of content items (only "text"-typed items are kept -
// mirrors FunctionCallOutputBody::to_text's own lossy behavior on images).
func functionCallOutputText(raw json.RawMessage) string {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var items []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &items); err != nil {
		return ""
	}
	var sb strings.Builder
	for i, it := range items {
		if it.Type != "" && it.Type != "text" && it.Type != "output_text" {
			continue
		}
		if i > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(it.Text)
	}
	return sb.String()
}

func (responsesDialect) ParseRequest(r *http.Request) (*ChatRequest, error) {
	defer r.Body.Close()

	var body struct {
		Model        string                   `json:"model"`
		Instructions string                   `json:"instructions"`
		Input        []responsesWireInputItem `json:"input"`
		Tools        []responsesWireTool      `json:"tools,omitempty"`
		Reasoning    json.RawMessage          `json:"reasoning"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("decode responses request: %w", err)
	}
	if body.Model == "" {
		return nil, fmt.Errorf("missing model field")
	}

	messages := make([]ChatMessage, 0, len(body.Input)+1)
	if body.Instructions != "" {
		messages = append(messages, ChatMessage{Role: "system", Content: body.Instructions})
	}
	for _, item := range body.Input {
		switch item.Type {
		case "", "message":
			var text strings.Builder
			for _, c := range item.Content {
				text.WriteString(c.Text)
			}
			messages = append(messages, ChatMessage{Role: item.Role, Content: text.String()})
		case "function_call":
			args := json.RawMessage(item.Arguments)
			if !json.Valid(args) {
				return nil, fmt.Errorf("function_call %q arguments is not valid JSON: %s", item.CallID, item.Arguments)
			}
			messages = append(messages, ChatMessage{
				Role: "assistant",
				ToolCalls: []ToolCall{{
					ID:        item.CallID,
					Name:      namespacedToolName(item.Namespace, item.Name),
					Arguments: args,
				}},
			})
		case "function_call_output":
			messages = append(messages, ChatMessage{
				Role:       "tool",
				Content:    functionCallOutputText(item.Output),
				ToolCallID: item.CallID,
			})
		default:
			// reasoning and anything else: skipped, not erroring, matching
			// this dialect's original text-only tolerance.
		}
	}

	tools := make([]ToolSpec, 0, len(body.Tools))
	for _, t := range body.Tools {
		if t.Type == "namespace" {
			for _, nested := range t.Tools {
				if nested.Type != "" && nested.Type != "function" {
					continue
				}
				nested.Name = namespacedToolName(t.Name, nested.Name)
				tools = append(tools, ToolSpec{Type: "function", Function: ToolSpecFunction{Name: nested.Name, Description: nested.Description, Parameters: nested.Parameters}})
			}
			continue
		}
		if t.Type != "" && t.Type != "function" {
			continue
		}
		tools = append(tools, ToolSpec{
			Type: "function",
			Function: ToolSpecFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		})
	}

	provider, model := splitProviderModel(body.Model)
	return &ChatRequest{
		Provider: provider,
		Model:    model,
		Messages: messages,
		Tools:    tools,
		Effort:   responsesReasoningEffort(body.Reasoning),
		// The Responses client always sends "Accept: text/event-stream"
		// and requires SSE framing regardless of a "stream" field on the
		// request body - same shape of requirement as chat_completions'
		// codex-fork client (see ChatRequest.Stream doc comment).
		Stream: true,
	}, nil
}

// responsesReasoningEffort extracts and normalizes reasoning.effort without
// rejecting malformed or unexpected reasoning values.
func responsesReasoningEffort(raw json.RawMessage) string {
	var reasoning map[string]json.RawMessage
	if json.Unmarshal(raw, &reasoning) != nil {
		return ""
	}
	var effort string
	if json.Unmarshal(reasoning["effort"], &effort) != nil {
		return ""
	}
	switch effort {
	case "none", "low", "medium", "high", "xhigh", "max":
		return effort
	case "minimal":
		return "low"
	default:
		return ""
	}
}

// WriteResponse frames the whole (non-streamed-from-the-backend) reply as
// the minimum SSE event sequence codex-rs's Responses parser accepts:
// one response.output_item.done per output item, then a mandatory
// response.completed - omitting it produces "stream closed before
// response.completed" client-side (verified against codex-rs's own
// process_sse test fixtures, codex-api/src/sse/responses.rs).
func (responsesDialect) WriteResponse(w http.ResponseWriter, resp *ChatResponse) error {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, canFlush := w.(http.Flusher)

	writeEvent := func(kind string, payload map[string]any) error {
		payload["type"] = kind
		data, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("marshal SSE event %s: %w", kind, err)
		}
		if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", kind, data); err != nil {
			return fmt.Errorf("write SSE event %s: %w", kind, err)
		}
		if canFlush {
			flusher.Flush()
		}
		return nil
	}

	if err := writeEvent("response.output_item.done", map[string]any{
		"item": map[string]any{
			"type": "message",
			"role": "assistant",
			"content": []map[string]any{
				{"type": "output_text", "text": resp.Content},
			},
		},
	}); err != nil {
		return err
	}

	for _, tc := range resp.ToolCalls {
		name, namespace := splitNamespacedToolName(tc.Name)
		args := string(tc.Arguments)
		if !json.Valid(tc.Arguments) {
			args = "{}"
		}
		item := map[string]any{
			"type":      "function_call",
			"name":      name,
			"arguments": args,
			"call_id":   tc.ID,
		}
		if namespace != "" {
			item["namespace"] = namespace
		}
		if err := writeEvent("response.output_item.done", map[string]any{"item": item}); err != nil {
			return err
		}
	}

	respID := fmt.Sprintf("resp-gachbothar-%d", time.Now().UnixNano())
	response := map[string]any{"id": respID}
	if resp.Model != "" {
		response["model"] = resp.Model
	}
	if resp.Usage != nil {
		response["usage"] = map[string]any{"input_tokens": resp.Usage.InputTokens, "input_tokens_details": map[string]any{"cached_tokens": resp.Usage.CachedInputTokens}, "output_tokens": resp.Usage.OutputTokens, "total_tokens": resp.Usage.InputTokens + resp.Usage.OutputTokens}
	}
	return writeEvent("response.completed", map[string]any{"response": response})
}

func namespacedToolName(namespace, name string) string {
	if namespace == "" {
		return name
	}
	return namespace + "__" + name
}

func splitNamespacedToolName(name string) (string, string) {
	if !strings.HasPrefix(name, "mcp__") {
		return name, ""
	}
	rest := strings.TrimPrefix(name, "mcp__")
	i := strings.Index(rest, "__")
	if i < 0 {
		return name, ""
	}
	return rest[i+2:], "mcp__" + rest[:i]
}

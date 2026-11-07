package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// chatCompletionsDialect implements the OpenAI-style /v1/chat/completions
// dialect.
type chatCompletionsDialect struct{}

func (chatCompletionsDialect) Name() string { return "chat_completions" }

func (chatCompletionsDialect) RoutePath() string { return "/v1/chat/completions" }

// openAIWireMessage matches the chat_completions wire shape for messages.
// We decode into this and convert by hand so that ChatMessage's tool fields
// (tagged json:"-") stay out of the direct marshal path.
type openAIWireMessage struct {
	Role       string               `json:"role"`
	Content    openAIWireContent    `json:"content"`
	ToolCalls  []openAIWireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string               `json:"tool_call_id,omitempty"`
}

// openAIWireContent accepts both shapes the chat_completions wire uses for
// message content: a plain string (most clients), or an array of content
// parts (pi's client, and any caller using the multi-part message format -
// confirmed live 30 Sep 2026: pi sent content as
// [{"type":"text","text":"..."}] and a bare `Content string` field failed
// to unmarshal it entirely, 400ing every request). Only text parts are
// read; non-text parts (e.g. images) are silently skipped rather than
// erroring, since this dialect has no image support to route them to.
type openAIWireContent string

func (c *openAIWireContent) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		*c = openAIWireContent(s)
		return nil
	}

	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(data, &parts); err != nil {
		return fmt.Errorf("content must be a string or an array of content parts: %w", err)
	}
	var sb strings.Builder
	for _, p := range parts {
		sb.WriteString(p.Text)
	}
	*c = openAIWireContent(sb.String())
	return nil
}

type openAIWireToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function openAIWireToolFunction `json:"function"`
}

type openAIWireToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// parseOpenAIWireToolCalls converts the chat_completions wire shape for tool
// calls (function.arguments as a JSON-encoded string) into the internal
// representation (Arguments as a decoded JSON object literal - see ToolCall's
// doc comment in types.go). Shared by chatCompletionsDialect.ParseRequest and
// ZenProvider.callChat, which both receive tool calls in this same wire shape.
func parseOpenAIWireToolCalls(wire []openAIWireToolCall) ([]ToolCall, error) {
	if len(wire) == 0 {
		return nil, nil
	}
	out := make([]ToolCall, 0, len(wire))
	for _, tc := range wire {
		args := json.RawMessage(tc.Function.Arguments)
		if !json.Valid(args) {
			return nil, fmt.Errorf("tool call %q arguments is not valid JSON: %s", tc.ID, tc.Function.Arguments)
		}
		out = append(out, ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: args,
		})
	}
	return out, nil
}

func (chatCompletionsDialect) ParseRequest(r *http.Request) (*ChatRequest, error) {
	defer r.Body.Close()

	var body struct {
		Model    string              `json:"model"`
		Messages []openAIWireMessage `json:"messages"`
		Stream   bool                `json:"stream"`
		Tools    []ToolSpec          `json:"tools,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("decode chat_completions request: %w", err)
	}
	if body.Model == "" {
		return nil, fmt.Errorf("missing model field")
	}

	messages := make([]ChatMessage, 0, len(body.Messages))
	for i, m := range body.Messages {
		cm := ChatMessage{
			Role:       m.Role,
			Content:    string(m.Content),
			ToolCallID: m.ToolCallID,
		}
		if len(m.ToolCalls) > 0 {
			toolCalls, err := parseOpenAIWireToolCalls(m.ToolCalls)
			if err != nil {
				return nil, fmt.Errorf("decode chat_completions request: message %d %w", i, err)
			}
			cm.ToolCalls = toolCalls
		}
		messages = append(messages, cm)
	}

	provider, model := splitProviderModel(body.Model)
	return &ChatRequest{
		Provider: provider,
		Model:    model,
		Messages: messages,
		Stream:   body.Stream,
		Tools:    body.Tools,
	}, nil
}

func (chatCompletionsDialect) WriteResponse(w http.ResponseWriter, resp *ChatResponse) error {
	if resp.Stream {
		return writeSSEChatCompletion(w, resp.Content, resp.Model, resp.ToolCalls, resp.Usage)
	}

	w.Header().Set("Content-Type", "application/json")

	finishReason := "stop"
	msg := map[string]any{
		"role":    "assistant",
		"content": resp.Content,
	}
	if len(resp.ToolCalls) > 0 {
		finishReason = "tool_calls"
		msg["tool_calls"] = buildOpenAIToolCalls(resp.ToolCalls)
		if resp.Content == "" {
			msg["content"] = nil
		}
	}

	out := struct {
		Choices []struct {
			Message      map[string]any `json:"message"`
			FinishReason string         `json:"finish_reason"`
		} `json:"choices"`
		Usage map[string]any `json:"usage,omitempty"`
	}{
		Choices: []struct {
			Message      map[string]any `json:"message"`
			FinishReason string         `json:"finish_reason"`
		}{
			{Message: msg, FinishReason: finishReason},
		},
	}
	if resp.Usage != nil {
		out.Usage = openAIUsage(resp.Usage)
	}
	return json.NewEncoder(w).Encode(out)
}

func openAIUsage(usage *Usage) map[string]any {
	return map[string]any{
		"prompt_tokens":     usage.InputTokens,
		"completion_tokens": usage.OutputTokens,
		"total_tokens":      usage.InputTokens + usage.OutputTokens,
		"prompt_tokens_details": map[string]any{
			"cached_tokens": usage.CachedInputTokens,
		},
	}
}

// buildOpenAIToolCalls returns the wire-shaped tool_calls array, with each
// function.arguments re-serialized from the internal decoded object into a
// JSON string.
func buildOpenAIToolCalls(toolCalls []ToolCall) []map[string]any {
	out := make([]map[string]any, 0, len(toolCalls))
	for _, tc := range toolCalls {
		args := string(tc.Arguments)
		if !json.Valid(tc.Arguments) {
			args = "{}"
		}
		out = append(out, map[string]any{
			"id":   tc.ID,
			"type": "function",
			"function": map[string]any{
				"name":      tc.Name,
				"arguments": args,
			},
		})
	}
	return out
}

// writeSSEChatCompletion frames the whole (non-streamed-from-the-backend)
// response as a single SSE chunk, the minimum a strict chat_completions
// streaming client will accept: one delta chunk carrying the full content,
// a finish_reason chunk, then [DONE]. Not real token-by-token streaming -
// gachbothar's v1 scope stays "one shot to the backend" (see DESIGN.md) -
// this only satisfies callers that require SSE framing on the wire.
//
// id/created/model are set on every chunk because codex-fork's
// chat_completions client silently drops a chunk missing them (no error,
// no agent_message, turn just completes empty) rather than erroring -
// confirmed live 22 Sep 2026: gachbothar's own request/response logging
// showed content="pong" actually being sent, but codex surfaced nothing
// until these fields were added.
func writeSSEChatCompletion(w http.ResponseWriter, content, model string, toolCalls []ToolCall, usage *Usage) error {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, canFlush := w.(http.Flusher)
	id := fmt.Sprintf("chatcmpl-gachbothar-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	writeChunk := func(delta map[string]any, finishReason any, includeUsage bool) error {
		chunk := map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{
				{"index": 0, "delta": delta, "finish_reason": finishReason},
			},
		}
		if includeUsage && usage != nil {
			chunk["usage"] = openAIUsage(usage)
		}
		data, err := json.Marshal(chunk)
		if err != nil {
			return fmt.Errorf("marshal SSE chunk: %w", err)
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			return fmt.Errorf("write SSE chunk: %w", err)
		}
		if canFlush {
			flusher.Flush()
		}
		return nil
	}

	delta := map[string]any{"role": "assistant", "content": content}
	finishReason := "stop"
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
		delta["tool_calls"] = buildOpenAIToolCalls(toolCalls)
		if content == "" {
			delta["content"] = nil
		}
	}

	if err := writeChunk(delta, nil, false); err != nil {
		return err
	}
	if err := writeChunk(map[string]any{}, finishReason, true); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, "data: [DONE]\n\n"); err != nil {
		return fmt.Errorf("write SSE terminator: %w", err)
	}
	if canFlush {
		flusher.Flush()
	}
	return nil
}

func splitProviderModel(raw string) (provider, model string) {
	parts := strings.SplitN(raw, "/", 2)
	if len(parts) == 1 {
		return parts[0], ""
	}
	return parts[0], parts[1]
}

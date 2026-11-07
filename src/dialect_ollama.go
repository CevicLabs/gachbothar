package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// ollamaNativeDialect implements Ollama's native /api/chat dialect.
type ollamaNativeDialect struct{}

func (ollamaNativeDialect) Name() string { return "ollama-native" }

func (ollamaNativeDialect) RoutePath() string { return "/api/chat" }

func (ollamaNativeDialect) ParseRequest(r *http.Request) (*ChatRequest, error) {
	defer r.Body.Close()

	var body struct {
		Model    string                 `json:"model"`
		Messages []ollamaWireMessage    `json:"messages"`
		Tools    []ToolSpec             `json:"tools"`
		Stream   bool                   `json:"stream"`
		Options  map[string]interface{} `json:"options"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("decode ollama-native request: %w", err)
	}
	if body.Model == "" {
		return nil, fmt.Errorf("missing model field")
	}

	provider, model := splitProviderModel(body.Model)
	messages := make([]ChatMessage, 0, len(body.Messages))
	for _, wm := range body.Messages {
		m := ChatMessage{Role: wm.Role, Content: wm.Content, ToolCallID: wm.ToolCallID}
		for _, tc := range wm.ToolCalls {
			m.ToolCalls = append(m.ToolCalls, ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
		}
		messages = append(messages, m)
	}
	// Ollama's num_predict reads 0 and negatives as "no limit", so both stay
	// unset here; only a fraction is not a number of tokens at all.
	maxTokens := 0
	if raw, ok := body.Options["num_predict"]; ok {
		limit, ok := raw.(float64)
		if !ok || limit != float64(int(limit)) {
			return nil, fmt.Errorf("options.num_predict must be a whole number: %v", raw)
		}
		if limit > 0 {
			maxTokens = int(limit)
		}
	}

	return &ChatRequest{
		Provider:  provider,
		Model:     model,
		Messages:  messages,
		Stream:    body.Stream,
		Tools:     body.Tools,
		MaxTokens: maxTokens,
	}, nil
}

func (ollamaNativeDialect) WriteResponse(w http.ResponseWriter, resp *ChatResponse) error {
	w.Header().Set("Content-Type", "application/json")

	if !resp.Stream {
		out := struct {
			Message         ollamaWireMessage `json:"message"`
			PromptEvalCount *int              `json:"prompt_eval_count,omitempty"`
			EvalCount       *int              `json:"eval_count,omitempty"`
		}{
			Message: ollamaResponseMessage(resp),
		}
		if resp.Usage != nil {
			promptTokens, outputTokens := resp.Usage.InputTokens, resp.Usage.OutputTokens
			out.PromptEvalCount = &promptTokens
			out.EvalCount = &outputTokens
		}
		return json.NewEncoder(w).Encode(out)
	}

	// Ollama's streaming wire format is newline-delimited JSON objects, not
	// SSE. gachbothar only ever makes one non-streaming call to the backend
	// (see writeSSEChatCompletion's comment on the chat_completions dialect
	// for the same tradeoff), so this fakes the minimum framing a strict
	// streaming client needs: one chunk carrying the full content, then a
	// closing chunk with done:true.
	type ollamaStreamChunk struct {
		Model           string            `json:"model"`
		CreatedAt       string            `json:"created_at"`
		Message         ollamaWireMessage `json:"message"`
		Done            bool              `json:"done"`
		PromptEvalCount *int              `json:"prompt_eval_count,omitempty"`
		EvalCount       *int              `json:"eval_count,omitempty"`
	}

	enc := json.NewEncoder(w)
	flusher, canFlush := w.(http.Flusher)
	now := time.Now().UTC().Format(time.RFC3339Nano)

	if err := enc.Encode(ollamaStreamChunk{
		Model:     resp.Model,
		CreatedAt: now,
		Message:   ollamaResponseMessage(resp),
		Done:      false,
	}); err != nil {
		return fmt.Errorf("write ollama-native stream chunk: %w", err)
	}
	if canFlush {
		flusher.Flush()
	}

	doneChunk := ollamaStreamChunk{
		Model:     resp.Model,
		CreatedAt: now,
		Message:   ollamaWireMessage{Role: "assistant", Content: ""},
		Done:      true,
	}
	if resp.Usage != nil {
		promptTokens, outputTokens := resp.Usage.InputTokens, resp.Usage.OutputTokens
		doneChunk.PromptEvalCount = &promptTokens
		doneChunk.EvalCount = &outputTokens
	}
	if err := enc.Encode(doneChunk); err != nil {
		return fmt.Errorf("write ollama-native stream terminator: %w", err)
	}
	if canFlush {
		flusher.Flush()
	}
	return nil
}

func ollamaResponseMessage(resp *ChatResponse) ollamaWireMessage {
	m := ollamaWireMessage{Role: "assistant", Content: resp.Content}
	for _, tc := range resp.ToolCalls {
		m.ToolCalls = append(m.ToolCalls, ollamaWireToolCall{ID: tc.ID, Function: ollamaWireToolFunction{Name: tc.Name, Arguments: tc.Arguments}})
	}
	return m
}

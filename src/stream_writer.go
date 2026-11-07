package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type streamWriter struct {
	w                             http.ResponseWriter
	dialect, model, id, createdAt string
	created                       int64
	started                       bool
}

func newStreamWriter(w http.ResponseWriter, dialect, model string) *streamWriter {
	now := time.Now()
	return &streamWriter{w: w, dialect: dialect, model: model, id: fmt.Sprintf("chatcmpl-gachbothar-%d", now.UnixNano()), created: now.Unix(), createdAt: now.UTC().Format(time.RFC3339Nano)}
}
func (s *streamWriter) write(v any, event string) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if !s.started {
		if s.dialect == "ollama-native" {
			s.w.Header().Set("Content-Type", "application/x-ndjson")
		} else {
			s.w.Header().Set("Content-Type", "text/event-stream")
			s.w.Header().Set("Cache-Control", "no-cache")
		}
		s.started = true
	}
	switch s.dialect {
	case "ollama-native":
		_, err = fmt.Fprintf(s.w, "%s\n", data)
	case "responses":
		_, err = fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", event, data)
	default:
		_, err = fmt.Fprintf(s.w, "data: %s\n\n", data)
	}
	if err == nil {
		if f, ok := s.w.(http.Flusher); ok {
			f.Flush()
		}
	}
	return err
}
func (s *streamWriter) chatChunk(delta map[string]any, reason any, usage *Usage) error {
	chunk := map[string]any{"id": s.id, "object": "chat.completion.chunk", "created": s.created, "model": s.model, "choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": reason}}}
	if usage != nil {
		chunk["usage"] = openAIUsage(usage)
	}
	return s.write(chunk, "")
}
func (s *streamWriter) WriteText(text string) error {
	switch s.dialect {
	case "responses":
		// Codex ignores a delta unless a message item is already open.
		if !s.started {
			item := map[string]any{"type": "message", "role": "assistant", "content": []any{}}
			if err := s.write(map[string]any{"type": "response.output_item.added", "item": item}, "response.output_item.added"); err != nil {
				return err
			}
		}
		return s.write(map[string]any{"type": "response.output_text.delta", "delta": text}, "response.output_text.delta")
	case "ollama-native":
		return s.write(map[string]any{"model": s.model, "created_at": s.createdAt, "message": map[string]any{"role": "assistant", "content": text}, "done": false}, "")
	case "chat_completions":
		delta := map[string]any{"content": text}
		if !s.started {
			delta["role"] = "assistant"
		}
		return s.chatChunk(delta, nil, nil)
	default:
		return fmt.Errorf("unsupported streaming dialect %q", s.dialect)
	}
}
func (s *streamWriter) Finish(d DialectAdapter, resp *ChatResponse) error {
	if !s.started && resp.Content != "" {
		return d.WriteResponse(s.w, resp)
	}
	switch s.dialect {
	case "responses":
		return d.WriteResponse(s.w, resp)
	case "chat_completions":
		reason := "stop"
		if len(resp.ToolCalls) != 0 {
			reason = "tool_calls"
			calls := buildOpenAIToolCalls(resp.ToolCalls)
			for i := range calls {
				calls[i]["index"] = i
			}
			delta := map[string]any{"tool_calls": calls}
			if !s.started {
				delta["role"] = "assistant"
			}
			if err := s.chatChunk(delta, nil, nil); err != nil {
				return err
			}
		}
		if err := s.chatChunk(map[string]any{}, reason, resp.Usage); err != nil {
			return err
		}
		_, err := fmt.Fprint(s.w, "data: [DONE]\n\n")
		if f, ok := s.w.(http.Flusher); ok {
			f.Flush()
		}
		return err
	case "ollama-native":
		msg := map[string]any{"role": "assistant", "content": ""}
		if len(resp.ToolCalls) > 0 {
			calls := make([]ollamaWireToolCall, 0, len(resp.ToolCalls))
			for _, tc := range resp.ToolCalls {
				calls = append(calls, ollamaWireToolCall{ID: tc.ID, Function: ollamaWireToolFunction{Name: tc.Name, Arguments: tc.Arguments}})
			}
			msg["tool_calls"] = calls
		}
		chunk := map[string]any{"model": s.model, "created_at": s.createdAt, "message": msg, "done": true}
		if resp.Usage != nil {
			chunk["prompt_eval_count"] = resp.Usage.InputTokens
			chunk["eval_count"] = resp.Usage.OutputTokens
		}
		return s.write(chunk, "")
	}
	return fmt.Errorf("unsupported streaming dialect %q", s.dialect)
}

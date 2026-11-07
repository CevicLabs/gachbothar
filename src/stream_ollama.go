package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
)

// Stream forwards Ollama NDJSON content as it arrives. Tool calls and usage are
// accumulated until the terminal record for conversion by the dialect.
func (p *OllamaProvider) Stream(ctx context.Context, req *ChatRequest, emit func(string) error) (*ChatResponse, error) {
	current := *req
	for attempt := 0; ; attempt++ {
		raw, err := p.marshalChatRequest(&current)
		if err != nil {
			return nil, err
		}
		var body map[string]any
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.UseNumber()
		if err = decoder.Decode(&body); err != nil {
			return nil, err
		}
		body["stream"] = true
		raw, err = json.Marshal(body)
		if err != nil {
			return nil, err
		}
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/chat", bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		request.Header.Set("Content-Type", "application/json")
		p.setAuthHeader(request)
		response, err := streamDo(p.client, request)
		if err != nil {
			return nil, err
		}
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			b, readErr := io.ReadAll(response.Body)
			response.Body.Close()
			if readErr != nil {
				return nil, readErr
			}
			if response.StatusCode == http.StatusBadRequest && current.Effort != "" && attempt == 0 {
				log.Printf("ollama model %q rejected reasoning effort %q; retrying without effort", req.Model, req.Effort)
				current.Effort = ""
				continue
			}
			// Preserve the established auto-pull behavior; that uncommon first request
			// remains buffered until a subsequent request uses the now-present model.
			if p.autoPull && response.StatusCode == http.StatusNotFound && isModelNotFound(b, req.Model) {
				return p.Call(req)
			}
			return nil, fmt.Errorf("ollama backend %q returned %d: %s", p.name, response.StatusCode, b)
		}
		defer response.Body.Close()
		dec := json.NewDecoder(response.Body)
		result := &ChatResponse{}
		for {
			var raw json.RawMessage
			if err := dec.Decode(&raw); err != nil {
				if err == io.EOF {
					return nil, fmt.Errorf("ollama stream ended before done")
				}
				return nil, fmt.Errorf("read ollama stream: %w", err)
			}
			var chunk struct {
				Message ollamaWireMessage `json:"message"`
				Done    bool              `json:"done"`
				Error   string            `json:"error"`
			}
			if err := json.Unmarshal(raw, &chunk); err != nil {
				return nil, fmt.Errorf("parse ollama stream: %w", err)
			}
			if chunk.Error != "" {
				return nil, fmt.Errorf("ollama stream: %s", chunk.Error)
			}
			if chunk.Message.Content != "" {
				result.Content += chunk.Message.Content
				if err := emit(chunk.Message.Content); err != nil {
					return nil, err
				}
			}
			for _, call := range chunk.Message.ToolCalls {
				id := call.ID
				if id == "" {
					id = fmt.Sprintf("call_%d", len(result.ToolCalls))
				}
				result.ToolCalls = append(result.ToolCalls, ToolCall{ID: id, Name: call.Function.Name, Arguments: call.Function.Arguments})
			}
			if chunk.Done {
				result.Usage = ollamaUsageFromResponse(raw)
				return result, nil
			}
		}
	}
}

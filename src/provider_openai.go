package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// OpenAIProvider calls OpenAI-compatible backends such as Ollama /v1 and
// llama.cpp. The configured base_url already includes its version path, so
// requests go to POST {base_url}/chat/completions without further rewriting.
//
// The wire helpers and the buffered/streamed chat-completions parsing are the
// ones Zen's chat protocol uses, shared instead of copied. This provider only
// adds its own auth rule: Authorization is sent as Bearer only when
// api_key_env names a set, non-empty environment variable (local Ollama and
// llama.cpp need no key at all).
type OpenAIProvider struct {
	name    string
	baseURL string
	apiEnv  string
	client  *http.Client
}

// NewOpenAIProvider constructs an OpenAI-compatible provider adapter.
func NewOpenAIProvider(name, baseURL, apiKeyEnv string) *OpenAIProvider {
	return &OpenAIProvider{
		name:    name,
		baseURL: strings.TrimRight(baseURL, "/"),
		apiEnv:  apiKeyEnv,
		client:  &http.Client{Timeout: 120 * time.Second},
	}
}

// Name returns the provider's registered name.
func (p *OpenAIProvider) Name() string { return p.name }

// apiKey resolves the configured environment variable. ok is false both when
// no api_key_env is configured and when that variable is unset or empty.
func (p *OpenAIProvider) apiKey() (string, bool) {
	if p.apiEnv == "" {
		return "", false
	}
	key := os.Getenv(p.apiEnv)
	return key, key != ""
}

func (p *OpenAIProvider) setAuth(header http.Header) {
	if key, ok := p.apiKey(); ok {
		header.Set("Authorization", "Bearer "+key)
	}
}

// Call sends a non-streaming chat request.
func (p *OpenAIProvider) Call(req *ChatRequest) (*ChatResponse, error) {
	body, err := openAIChatRequestBody(req)
	if err != nil {
		return nil, err
	}

	respBody, status, err := p.post(body)
	if status == http.StatusBadRequest && body["reasoning_effort"] != nil {
		log.Printf("openai model %q rejected reasoning effort %q; retrying without effort", req.Model, req.Effort)
		delete(body, "reasoning_effort")
		respBody, status, err = p.post(body)
	}
	if err != nil {
		return nil, err
	}
	return parseOpenAIChatResponse("openai chat", respBody)
}

// Stream uses the shared OpenAI chat SSE parser with usage included in the
// final stream event. streamSSE logs the model from req and returns a typed
// non-2xx error, which Stream wraps so callers see the provider name.
func (p *OpenAIProvider) Stream(ctx context.Context, req *ChatRequest, emit func(string) error) (*ChatResponse, error) {
	body, err := openAIChatRequestBody(req)
	if err != nil {
		return nil, err
	}
	body["stream_options"] = map[string]any{"include_usage": true}

	resp, err := streamSSE(ctx, p.client, p.baseURL+"/chat/completions", body, req, "chat", p.setAuth, emit)
	if err != nil {
		var failed *streamStatusError
		if errors.As(err, &failed) {
			return nil, fmt.Errorf("openai backend %q returned %d: %s", p.name, failed.status, failed.body)
		}
		return nil, err
	}
	return resp, nil
}

// openAIChatRequestBody builds the request body shared by Zen's chat protocol
// and OpenAI-compatible backends: the OpenAI chat wire shape, with tool call
// arguments encoded as JSON strings.
func openAIChatRequestBody(req *ChatRequest) (map[string]any, error) {
	messages := make([]openAIWireMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		wm := openAIWireMessage{
			Role:       m.Role,
			Content:    openAIWireContent(m.Content),
			ToolCallID: m.ToolCallID,
		}
		for _, tc := range m.ToolCalls {
			args := string(tc.Arguments)
			if !json.Valid(tc.Arguments) {
				args = "{}"
			}
			wm.ToolCalls = append(wm.ToolCalls, openAIWireToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: openAIWireToolFunction{
					Name:      tc.Name,
					Arguments: args,
				},
			})
		}
		messages = append(messages, wm)
	}

	body := struct {
		Model     string              `json:"model"`
		Messages  []openAIWireMessage `json:"messages"`
		Stream    bool                `json:"stream"`
		Tools     []ToolSpec          `json:"tools,omitempty"`
		Effort    string              `json:"reasoning_effort,omitempty"`
		MaxTokens int                 `json:"max_tokens,omitempty"`
	}{
		Model:     req.Model,
		Messages:  messages,
		Stream:    false,
		Tools:     req.Tools,
		Effort:    req.Effort,
		MaxTokens: req.MaxTokens,
	}

	return requestMap(body)
}

// parseOpenAIChatResponse reads a buffered chat-completions reply for both
// providers, reusing the shared tool-call and usage helpers.
func parseOpenAIChatResponse(kind string, respBody []byte) (*ChatResponse, error) {
	var out struct {
		Usage   json.RawMessage `json:"usage"`
		Choices []struct {
			Message openAIWireMessage `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, fmt.Errorf("parse %s response: %w", kind, err)
	}
	if len(out.Choices) == 0 {
		return nil, fmt.Errorf("%s response had no choices", kind)
	}

	msg := out.Choices[0].Message
	toolCalls, err := parseOpenAIWireToolCalls(msg.ToolCalls)
	if err != nil {
		return nil, fmt.Errorf("parse %s response: %w", kind, err)
	}
	result := &ChatResponse{Content: string(msg.Content), ToolCalls: toolCalls}
	if len(out.Usage) > 0 {
		if err := mergeChatUsageStrict(&result.Usage, out.Usage); err != nil {
			return nil, fmt.Errorf("parse %s response: %w", kind, err)
		}
	}
	return result, nil
}

func (p *OpenAIProvider) post(body any) ([]byte, int, error) {
	bodyJSON, err := json.Marshal(body)
	if err != nil {
		return nil, 0, fmt.Errorf("marshal openai request: %w", err)
	}

	url := p.baseURL + "/chat/completions"
	httpReq, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(bodyJSON))
	if err != nil {
		return nil, 0, fmt.Errorf("create request to %q: %w", url, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	p.setAuth(httpReq.Header)

	httpResp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, 0, fmt.Errorf("call openai backend %q: %w", p.name, err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	failed := httpResp.StatusCode < 200 || httpResp.StatusCode >= 300
	if err != nil && failed {
		return respBody, httpResp.StatusCode, fmt.Errorf("openai backend %q returned %d: %s (body read failed: %w)", p.name, httpResp.StatusCode, respBody, err)
	}
	if err != nil {
		return nil, httpResp.StatusCode, fmt.Errorf("read openai response: %w", err)
	}
	if failed {
		return respBody, httpResp.StatusCode, fmt.Errorf("openai backend %q returned %d: %s", p.name, httpResp.StatusCode, respBody)
	}
	return respBody, httpResp.StatusCode, nil
}

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// AnthropicProvider forwards normalized requests to the Anthropic Messages API.
type AnthropicProvider struct {
	name    string
	baseURL string
	apiKey  string
	client  *http.Client
}

func NewAnthropicProvider(name, baseURL, apiKeyEnv string) *AnthropicProvider {
	return &AnthropicProvider{
		name: name, baseURL: strings.TrimRight(baseURL, "/"), apiKey: os.Getenv(apiKeyEnv),
		client: &http.Client{Timeout: 120 * time.Second},
	}
}

func (p *AnthropicProvider) Name() string { return p.name }

type anthropicBlock struct {
	Type         string                 `json:"type"`
	Text         string                 `json:"text,omitempty"`
	ID           string                 `json:"id,omitempty"`
	Name         string                 `json:"name,omitempty"`
	Input        json.RawMessage        `json:"input,omitempty"`
	ToolUseID    string                 `json:"tool_use_id,omitempty"`
	Content      *string                `json:"content,omitempty"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicCacheControl struct {
	Type string `json:"type"`
}

type anthropicOutputConfig struct {
	Effort string `json:"effort"`
}

type anthropicMessage struct {
	Role    string           `json:"role"`
	Content []anthropicBlock `json:"content"`
}

type anthropicTool struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description,omitempty"`
	InputSchema  json.RawMessage        `json:"input_schema"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

// Call sends a non-streaming Messages request, preserving tool calls and results.
func (p *AnthropicProvider) Call(req *ChatRequest) (*ChatResponse, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("anthropic provider %q: no API key set (check api_key_env)", p.name)
	}
	body, err := anthropicRequestBody(req)
	if err != nil {
		return nil, err
	}

	raw, status, err := p.sendMessages(body)
	if err != nil {
		return nil, err
	}
	if status == http.StatusBadRequest && req.Effort != "" && req.Effort != "none" {
		log.Printf("anthropic model %q rejected reasoning effort %q; retrying without effort", req.Model, req.Effort)
		delete(body, "output_config")
		raw, status, err = p.sendMessages(body)
		if err != nil {
			return nil, err
		}
	}
	if status < 200 || status >= 300 {
		return nil, fmt.Errorf("anthropic backend %q returned %d: %s", p.name, status, raw)
	}
	var out struct {
		Content []anthropicBlock `json:"content"`
		Usage   struct {
			InputTokens              int `json:"input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			OutputTokens             int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("parse anthropic response: %w", err)
	}
	result := &ChatResponse{}
	result.Usage = &Usage{InputTokens: out.Usage.InputTokens + out.Usage.CacheCreationInputTokens + out.Usage.CacheReadInputTokens, CachedInputTokens: out.Usage.CacheReadInputTokens, OutputTokens: out.Usage.OutputTokens}
	for _, block := range out.Content {
		switch block.Type {
		case "text":
			result.Content += block.Text
		case "tool_use":
			result.ToolCalls = append(result.ToolCalls, ToolCall{ID: block.ID, Name: block.Name, Arguments: block.Input})
		}
	}
	return result, nil
}

func anthropicRequestBody(req *ChatRequest) (map[string]any, error) {
	var systemParts []string
	messages := make([]anthropicMessage, 0, len(req.Messages))
	previousTool := false
	for _, m := range req.Messages {
		if m.Role == "system" || m.Role == "developer" {
			systemParts = append(systemParts, m.Content)
			previousTool = false
			continue
		}
		if m.Role == "tool" {
			block := anthropicBlock{Type: "tool_result", ToolUseID: m.ToolCallID, Content: &m.Content}
			if previousTool {
				last := &messages[len(messages)-1]
				last.Content = append(last.Content, block)
			} else {
				messages = append(messages, anthropicMessage{Role: "user", Content: []anthropicBlock{block}})
			}
			previousTool = true
			continue
		}
		previousTool = false
		blocks := make([]anthropicBlock, 0, 1+len(m.ToolCalls))
		if m.Content != "" {
			blocks = append(blocks, anthropicBlock{Type: "text", Text: m.Content})
		}
		for _, call := range m.ToolCalls {
			input := call.Arguments
			if len(input) == 0 {
				input = json.RawMessage(`{}`)
			}
			blocks = append(blocks, anthropicBlock{Type: "tool_use", ID: call.ID, Name: call.Name, Input: input})
		}
		messages = append(messages, anthropicMessage{Role: m.Role, Content: blocks})
	}
	tools := make([]anthropicTool, 0, len(req.Tools))
	for _, tool := range req.Tools {
		schema := tool.Function.Parameters
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		tools = append(tools, anthropicTool{Name: tool.Function.Name, Description: tool.Function.Description, InputSchema: schema})
	}
	var system any
	if len(systemParts) > 0 {
		blocks := make([]anthropicBlock, 0, len(systemParts))
		for _, part := range systemParts {
			blocks = append(blocks, anthropicBlock{Type: "text", Text: part})
		}
		blocks[len(blocks)-1].CacheControl = &anthropicCacheControl{Type: "ephemeral"}
		system = blocks
	}
	if len(tools) > 0 {
		tools[len(tools)-1].CacheControl = &anthropicCacheControl{Type: "ephemeral"}
	}
	if len(messages) > 0 && len(messages[len(messages)-1].Content) > 0 {
		messages[len(messages)-1].Content[len(messages[len(messages)-1].Content)-1].CacheControl = &anthropicCacheControl{Type: "ephemeral"}
	}
	var outputConfig *anthropicOutputConfig
	if req.Effort != "" && req.Effort != "none" {
		outputConfig = &anthropicOutputConfig{Effort: req.Effort}
	}
	body := struct {
		Model        string                 `json:"model"`
		System       any                    `json:"system,omitempty"`
		Messages     []anthropicMessage     `json:"messages"`
		MaxTokens    int                    `json:"max_tokens"`
		Tools        []anthropicTool        `json:"tools,omitempty"`
		OutputConfig *anthropicOutputConfig `json:"output_config,omitempty"`
	}{req.Model, system, messages, tokenLimitOr(req.MaxTokens, 8192), tools, outputConfig}

	return requestMap(body)
}

func (p *AnthropicProvider) sendMessages(body any) ([]byte, int, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, 0, fmt.Errorf("marshal anthropic request: %w", err)
	}
	url := p.baseURL + "/v1/messages"
	httpReq, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(encoded))
	if err != nil {
		return nil, 0, fmt.Errorf("create anthropic request to %q: %w", url, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", p.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, 0, fmt.Errorf("call anthropic backend %q: %w", p.name, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, fmt.Errorf("read anthropic response: %w", err)
	}
	return raw, resp.StatusCode, nil
}

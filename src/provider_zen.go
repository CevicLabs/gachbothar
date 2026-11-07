package main

import (
	"bytes"
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

// ZenProvider calls OpenCode Zen's paid-tier chat endpoints. It covers two
// of Zen's four wire protocols:
//
//   - "chat" (the default): openai-chat, POST {base_url}/zen/v1/chat/completions,
//     auth via "Authorization: Bearer <key>".
//   - "messages": anthropic-messages, POST {base_url}/zen/v1/messages, auth
//     via "x-api-key: <key>" - Zen checks that header specifically for this
//     protocol and ignores a bare Authorization: Bearer, confirmed live
//     22 Sep 2026 against qwen3.5-plus (401 "Missing API key" without it).
//     Sending both headers is harmless, so "messages" sends both.
//
// Free-tier models are out of scope: they require a live opencode serve
// session (x-opencode-session tied to a real session, not a self-minted
// UUID), which gachbothar does not run or proxy to.
type ZenProvider struct {
	name     string
	baseURL  string
	apiKey   string
	protocol string
	client   *http.Client
}

// NewZenProvider constructs a Zen provider adapter. protocol must be "chat",
// "messages", or "" (defaults to "chat").
func NewZenProvider(name, baseURL, apiKeyEnv, protocol string) (*ZenProvider, error) {
	if protocol == "" {
		protocol = "chat"
	}
	if protocol != "chat" && protocol != "messages" {
		return nil, fmt.Errorf("zen provider %q: unknown protocol %q (want \"chat\" or \"messages\")", name, protocol)
	}
	apiKey := os.Getenv(apiKeyEnv)
	return &ZenProvider{
		name:     name,
		baseURL:  baseURL,
		apiKey:   apiKey,
		protocol: protocol,
		client:   &http.Client{Timeout: 120 * time.Second},
	}, nil
}

// Name returns the provider's registered name.
func (p *ZenProvider) Name() string { return p.name }

// Call sends a non-streaming chat request to Zen.
func (p *ZenProvider) Call(req *ChatRequest) (*ChatResponse, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("zen provider %q: no API key set (check api_key_env)", p.name)
	}

	call := p.callChat
	sent := req.Effort != ""
	if p.protocol == "messages" {
		call = p.callMessages
		sent = sent && req.Effort != "none"
	}
	resp, err := call(req)
	var failed *zenStatusError
	if !sent || !errors.As(err, &failed) || failed.status != http.StatusBadRequest {
		return resp, err
	}
	log.Printf("zen model %q rejected reasoning effort %q; retrying without effort", req.Model, req.Effort)
	withoutEffort := *req
	withoutEffort.Effort = ""
	return call(&withoutEffort)
}

// zenStatusError is a reply from Zen outside 2xx.
type zenStatusError struct {
	name   string
	status int
	body   []byte
}

func (e *zenStatusError) Error() string {
	return fmt.Sprintf("zen backend %q returned %d: %s", e.name, e.status, e.body)
}

func (p *ZenProvider) callChat(req *ChatRequest) (*ChatResponse, error) {
	body, err := openAIChatRequestBody(req)
	if err != nil {
		return nil, err
	}

	respBody, err := p.post("/zen/v1/chat/completions", body, func(h http.Header) {
		h.Set("Authorization", "Bearer "+p.apiKey)
	})
	if err != nil {
		return nil, err
	}
	return parseOpenAIChatResponse("zen chat", respBody)
}

func (p *ZenProvider) callMessages(req *ChatRequest) (*ChatResponse, error) {
	body, err := zenMessagesRequestBody(req)
	if err != nil {
		return nil, err
	}

	respBody, err := p.post("/zen/v1/messages", body, func(h http.Header) {
		h.Set("Authorization", "Bearer "+p.apiKey)
		h.Set("x-api-key", p.apiKey)
		h.Set("anthropic-version", "2023-06-01")
	})
	if err != nil {
		return nil, err
	}

	var out struct {
		Content []anthropicBlock `json:"content"`
		Usage   *struct {
			InputTokens              *int `json:"input_tokens"`
			CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     *int `json:"cache_read_input_tokens"`
			OutputTokens             *int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, fmt.Errorf("parse zen messages response: %w", err)
	}
	result := &ChatResponse{}
	if out.Usage != nil && (out.Usage.InputTokens != nil || out.Usage.CacheCreationInputTokens != nil || out.Usage.CacheReadInputTokens != nil || out.Usage.OutputTokens != nil) {
		usage := &Usage{}
		if out.Usage.InputTokens != nil {
			usage.InputTokens += *out.Usage.InputTokens
		}
		if out.Usage.CacheCreationInputTokens != nil {
			usage.InputTokens += *out.Usage.CacheCreationInputTokens
		}
		if out.Usage.CacheReadInputTokens != nil {
			usage.InputTokens += *out.Usage.CacheReadInputTokens
			usage.CachedInputTokens = *out.Usage.CacheReadInputTokens
		}
		if out.Usage.OutputTokens != nil {
			usage.OutputTokens = *out.Usage.OutputTokens
		}
		result.Usage = usage
	}
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

func zenMessagesRequestBody(req *ChatRequest) (map[string]any, error) {
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
				messages[len(messages)-1].Content = append(messages[len(messages)-1].Content, block)
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
		inputSchema := tool.Function.Parameters
		if len(inputSchema) == 0 {
			inputSchema = json.RawMessage(`{"type":"object"}`)
		}
		tools = append(tools, anthropicTool{Name: tool.Function.Name, Description: tool.Function.Description, InputSchema: inputSchema})
	}

	var outputConfig *anthropicOutputConfig
	if req.Effort != "" && req.Effort != "none" {
		outputConfig = &anthropicOutputConfig{Effort: req.Effort}
	}
	body := struct {
		Model        string                 `json:"model"`
		System       string                 `json:"system,omitempty"`
		Messages     []anthropicMessage     `json:"messages"`
		MaxTokens    int                    `json:"max_tokens"`
		Tools        []anthropicTool        `json:"tools,omitempty"`
		OutputConfig *anthropicOutputConfig `json:"output_config,omitempty"`
	}{
		Model:        req.Model,
		System:       strings.Join(systemParts, "\n\n"),
		Messages:     messages,
		MaxTokens:    tokenLimitOr(req.MaxTokens, 4096),
		Tools:        tools,
		OutputConfig: outputConfig,
	}

	return requestMap(body)
}

func (p *ZenProvider) post(path string, body any, setHeaders func(http.Header)) ([]byte, error) {
	bodyJSON, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal zen request: %w", err)
	}

	url := p.baseURL + path
	httpReq, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(bodyJSON))
	if err != nil {
		return nil, fmt.Errorf("create request to %q: %w", url, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	setHeaders(httpReq.Header)

	httpResp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("call zen backend %q: %w", p.name, err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("read zen response: %w", err)
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, &zenStatusError{name: p.name, status: httpResp.StatusCode, body: respBody}
	}
	return respBody, nil
}

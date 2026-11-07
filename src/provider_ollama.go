package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// OllamaProvider calls Ollama's native /api/chat endpoint.
// It is used for both local and cloud Ollama providers.
type OllamaProvider struct {
	name        string
	baseURL     string
	apiKeyEnv   string
	client      *http.Client
	autoPull    bool
	pullClient  *http.Client
	pullLocks   map[string]*sync.Mutex
	pullLocksMu sync.Mutex
}

// NewOllamaProvider constructs an Ollama provider adapter.
func NewOllamaProvider(name, baseURL, apiKeyEnv string, autoPull bool, pullTimeoutSeconds int) *OllamaProvider {
	timeout := 120 * time.Second
	pullTimeout := 600 * time.Second
	if pullTimeoutSeconds > 0 {
		pullTimeout = time.Duration(pullTimeoutSeconds) * time.Second
	}
	return &OllamaProvider{
		name:       name,
		baseURL:    baseURL,
		apiKeyEnv:  apiKeyEnv,
		client:     &http.Client{Timeout: timeout},
		autoPull:   autoPull,
		pullClient: &http.Client{Timeout: pullTimeout},
		pullLocks:  make(map[string]*sync.Mutex),
	}
}

// Name returns the provider's registered name.
func (p *OllamaProvider) Name() string { return p.name }

// lockForModel returns a mutex scoped to this provider+model pair, creating it
// if necessary. Concurrent callers for the same missing model must trigger
// exactly one pull.
func (p *OllamaProvider) lockForModel(model string) *sync.Mutex {
	p.pullLocksMu.Lock()
	defer p.pullLocksMu.Unlock()
	mu, ok := p.pullLocks[model]
	if !ok {
		mu = &sync.Mutex{}
		p.pullLocks[model] = mu
	}
	return mu
}

// ollamaWireMessage is the local request/response wire shape for Ollama's
// /api/chat. We build this by hand from ChatMessage so that ChatMessage's
// tool fields (tagged json:"-") never leak into the wire.
type ollamaWireMessage struct {
	Role       string               `json:"role"`
	Content    string               `json:"content"`
	ToolCalls  []ollamaWireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string               `json:"tool_call_id,omitempty"`
}

type ollamaWireToolCall struct {
	ID       string                 `json:"id,omitempty"`
	Function ollamaWireToolFunction `json:"function"`
}

type ollamaWireToolFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// Call sends a non-streaming chat request to the Ollama backend.
func (p *OllamaProvider) Call(req *ChatRequest) (*ChatResponse, error) {
	resp, status, body, err := p.doChat(req)
	if err == nil {
		return resp, nil
	}

	// Only attempt a pull for a 404 whose body says the model is missing,
	// and only if this provider is explicitly configured for auto_pull.
	if !p.autoPull || status != http.StatusNotFound || !isModelNotFound(body, req.Model) {
		return nil, err
	}

	// Serialize concurrent pulls for the same model on this provider.
	mu := p.lockForModel(req.Model)
	mu.Lock()
	defer mu.Unlock()

	// Another goroutine may have pulled while we were waiting. Check the
	// store first with a lightweight probe so a redundant pull is not fired
	// and so the original chat is retried only once in the single-threaded
	// case.
	if p.verifyModel(req.Model) == nil {
		resp, _, _, err = p.doChat(req)
		if err != nil {
			return nil, err
		}
		return resp, nil
	}

	start := time.Now()
	pullErr := p.pullModel(req.Model)
	elapsed := time.Since(start)
	if pullErr != nil {
		log.Printf("ollama auto_pull failed for model %q on provider %q after %v: %v", req.Model, p.name, elapsed, pullErr)
		return nil, fmt.Errorf("auto_pull failed for model %q on provider %q: %w", req.Model, p.name, pullErr)
	}

	// The pull reported success, but a pull can lie: confirm the model is
	// actually in the local store before retrying.
	if verifyErr := p.verifyModel(req.Model); verifyErr != nil {
		log.Printf("ollama auto_pull verification failed for model %q on provider %q after %v: %v", req.Model, p.name, elapsed, verifyErr)
		return nil, fmt.Errorf("auto_pull verification failed for model %q on provider %q: %w", req.Model, p.name, verifyErr)
	}

	log.Printf("ollama auto_pull succeeded for model %q on provider %q in %v", req.Model, p.name, elapsed)

	// Retry the original chat exactly once.
	resp, _, _, err = p.doChat(req)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// doChat performs one /api/chat request and returns the parsed response along
// with the HTTP status and body so the caller can decide whether to pull.
func (p *OllamaProvider) doChat(req *ChatRequest) (*ChatResponse, int, []byte, error) {
	resp, status, body, err := p.doChatOnce(req)
	if status != http.StatusBadRequest || req.Effort == "" {
		return resp, status, body, err
	}
	log.Printf("ollama model %q rejected reasoning effort %q; retrying without effort", req.Model, req.Effort)
	withoutEffort := *req
	withoutEffort.Effort = ""
	return p.doChatOnce(&withoutEffort)
}

func (p *OllamaProvider) doChatOnce(req *ChatRequest) (*ChatResponse, int, []byte, error) {
	bodyJSON, err := p.marshalChatRequest(req)
	if err != nil {
		return nil, 0, nil, err
	}

	url := p.baseURL + "/api/chat"
	httpReq, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(bodyJSON))
	if err != nil {
		return nil, 0, nil, fmt.Errorf("create request to %q: %w", url, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	p.setAuthHeader(httpReq)

	httpResp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, 0, nil, fmt.Errorf("call ollama backend %q: %w", p.name, err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, httpResp.StatusCode, nil, fmt.Errorf("read ollama response: %w", err)
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, httpResp.StatusCode, respBody, fmt.Errorf("ollama backend %q returned %d: %s", p.name, httpResp.StatusCode, string(respBody))
	}

	var out struct {
		Message struct {
			Role       string               `json:"role"`
			Content    string               `json:"content"`
			ToolCalls  []ollamaWireToolCall `json:"tool_calls,omitempty"`
			ToolCallID string               `json:"tool_call_id,omitempty"`
		} `json:"message"`
	}
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, httpResp.StatusCode, respBody, fmt.Errorf("parse ollama response: %w", err)
	}

	resp := &ChatResponse{Content: out.Message.Content, Usage: ollamaUsageFromResponse(respBody)}
	if len(out.Message.ToolCalls) > 0 {
		resp.ToolCalls = make([]ToolCall, 0, len(out.Message.ToolCalls))
		for i, tc := range out.Message.ToolCalls {
			id := tc.ID
			if id == "" {
				id = fmt.Sprintf("call_%d", i)
			}
			resp.ToolCalls = append(resp.ToolCalls, ToolCall{
				ID:        id,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
	}
	return resp, httpResp.StatusCode, respBody, nil
}

// ollamaThink maps an effort to Ollama's think field, which takes low, medium,
// high, max or a boolean and refuses anything else. nil leaves the field out.
func ollamaThink(effort string) any {
	switch effort {
	case "":
		return nil
	case "none":
		return false
	case "xhigh":
		return "high"
	}
	return effort
}

// ollamaRequestOptions carries the client's output-token limit. nil means the
// client set none, so the whole options object is left off the request.
type ollamaOptions struct {
	NumPredict int `json:"num_predict,omitempty"`
}

func ollamaRequestOptions(req *ChatRequest) *ollamaOptions {
	if req.MaxTokens == 0 {
		return nil
	}
	return &ollamaOptions{NumPredict: req.MaxTokens}
}

func (p *OllamaProvider) marshalChatRequest(req *ChatRequest) ([]byte, error) {
	wireMessages := make([]ollamaWireMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		om := ollamaWireMessage{
			Role:    m.Role,
			Content: m.Content,
		}
		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			om.ToolCalls = make([]ollamaWireToolCall, 0, len(m.ToolCalls))
			for _, tc := range m.ToolCalls {
				om.ToolCalls = append(om.ToolCalls, ollamaWireToolCall{
					Function: ollamaWireToolFunction{
						Name:      tc.Name,
						Arguments: tc.Arguments,
					},
				})
			}
		}
		if m.Role == "tool" {
			// Ollama native tool-result messages have no tool_call_id field.
			om.ToolCallID = ""
		}
		wireMessages = append(wireMessages, om)
	}

	body := struct {
		Model    string              `json:"model"`
		Messages []ollamaWireMessage `json:"messages"`
		Stream   bool                `json:"stream"`
		Tools    []ToolSpec          `json:"tools,omitempty"`
		Think    any                 `json:"think,omitempty"`
		Options  *ollamaOptions      `json:"options,omitempty"`
	}{
		Model:    req.Model,
		Messages: wireMessages,
		Stream:   false,
		Tools:    req.Tools,
		Think:    ollamaThink(req.Effort),
		Options:  ollamaRequestOptions(req),
	}

	bodyJSON, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal ollama request: %w", err)
	}
	return bodyJSON, nil
}

func (p *OllamaProvider) setAuthHeader(httpReq *http.Request) {
	if p.apiKeyEnv != "" {
		if key := os.Getenv(p.apiKeyEnv); key != "" {
			httpReq.Header.Set("Authorization", "Bearer "+key)
		}
	}
}

// ollamaUsageFromResponse converts Ollama's optional evaluation counts into
// normalized usage. A partial set of counts is still useful usage information.
func ollamaUsageFromResponse(body []byte) *Usage {
	var counts struct {
		PromptEvalCount *int `json:"prompt_eval_count"`
		EvalCount       *int `json:"eval_count"`
	}
	if json.Unmarshal(body, &counts) != nil || counts.PromptEvalCount == nil && counts.EvalCount == nil {
		return nil
	}
	usage := &Usage{}
	if counts.PromptEvalCount != nil {
		usage.InputTokens = *counts.PromptEvalCount
	}
	if counts.EvalCount != nil {
		usage.OutputTokens = *counts.EvalCount
	}
	return usage
}

// isModelNotFound detects Ollama's specific 404 body for a missing model.
// The real daemon returns {"error":"model 'x' not found"}. The test fake
// prints the model with %q, which yields malformed JSON, so the check is a
// defensive substring match rather than strict JSON parsing: the body must
// contain "not found" and mention the requested model name.
func isModelNotFound(body []byte, model string) bool {
	if len(body) == 0 {
		return false
	}
	msg := strings.ToLower(string(body))
	return strings.Contains(msg, "not found") && strings.Contains(msg, strings.ToLower(model))
}

// pullModel streams a POST to /api/pull and drains the response body. It
// returns an error if the HTTP round-trip or the JSON streaming itself fails;
// it does NOT verify that the model actually landed in the store.
func (p *OllamaProvider) pullModel(model string) error {
	url := p.baseURL + "/api/pull"
	payload, err := json.Marshal(map[string]any{"model": model, "stream": true})
	if err != nil {
		return fmt.Errorf("marshal pull request: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), p.pullClient.Timeout)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("create pull request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	p.setAuthHeader(httpReq)

	start := time.Now()
	resp, err := p.pullClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("pull request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("pull returned %d: %s", resp.StatusCode, string(body))
	}

	// Ollama streams the pull as NDJSON. Drain it so a long download does not
	// look idle, but log progress lines at a bounded rate.
	scanner := bufio.NewScanner(resp.Body)
	lastLog := time.Time{}
	lineCount := 0
	for scanner.Scan() {
		lineCount++
		if time.Since(lastLog) >= 5*time.Second {
			lastLog = time.Now()
			log.Printf("ollama pull %q on provider %q in progress (%d lines)", model, p.name, lineCount)
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read pull stream: %w", err)
	}

	log.Printf("ollama pull %q on provider %q completed after %v (%d lines)", model, p.name, time.Since(start), lineCount)
	return nil
}

// verifyModel asks the Ollama daemon whether the model is now in the local
// store. It prefers /api/show, falling back to /api/tags if show is not
// implemented.
func (p *OllamaProvider) verifyModel(model string) error {
	payload, err := json.Marshal(map[string]string{"model": model})
	if err != nil {
		return fmt.Errorf("marshal show request: %w", err)
	}
	url := p.baseURL + "/api/show"
	httpReq, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("create show request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	p.setAuthHeader(httpReq)

	resp, err := p.pullClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("show request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		return nil
	}
	if resp.StatusCode != http.StatusNotFound {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("show returned %d: %s", resp.StatusCode, string(body))
	}

	// /api/show returned 404; try /api/tags as a fallback.
	tagsReq, err := http.NewRequest(http.MethodGet, p.baseURL+"/api/tags", nil)
	if err != nil {
		return fmt.Errorf("create tags request: %w", err)
	}
	p.setAuthHeader(tagsReq)

	tagsResp, err := p.pullClient.Do(tagsReq)
	if err != nil {
		return fmt.Errorf("tags request: %w", err)
	}
	defer tagsResp.Body.Close()

	if tagsResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(tagsResp.Body)
		return fmt.Errorf("tags returned %d: %s", tagsResp.StatusCode, string(body))
	}

	var list struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(tagsResp.Body).Decode(&list); err != nil {
		return fmt.Errorf("parse tags response: %w", err)
	}
	for _, m := range list.Models {
		if m.Name == model {
			return nil
		}
	}
	return fmt.Errorf("model %q not found in tags after pull", model)
}

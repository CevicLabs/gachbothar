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
	"sort"
	"strings"
	"time"
)

func requestMap(body any) (map[string]any, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	decoder := json.NewDecoder(bytes.NewReader(b))
	decoder.UseNumber()
	err = decoder.Decode(&m)
	return m, err
}

func (p *ZenProvider) Stream(ctx context.Context, req *ChatRequest, emit func(string) error) (*ChatResponse, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("zen provider %q: no API key set (check api_key_env)", p.name)
	}
	var body map[string]any
	var err error
	path := "/zen/v1/chat/completions"
	if p.protocol == "messages" {
		body, err = zenMessagesRequestBody(req)
		path = "/zen/v1/messages"
	} else {
		body, err = openAIChatRequestBody(req)
	}
	if err != nil {
		return nil, err
	}
	if p.protocol == "chat" {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	return streamSSE(ctx, p.client, p.baseURL+path, body, req, p.protocol, func(h http.Header) {
		h.Set("Authorization", "Bearer "+p.apiKey)
		if p.protocol == "messages" {
			h.Set("x-api-key", p.apiKey)
			h.Set("anthropic-version", "2023-06-01")
		}
	}, emit)
}

// streamDo applies the client's timeout to the response headers only, so a
// long answer is not cut off while the backend is still sending it.
func streamDo(client *http.Client, request *http.Request) (*http.Response, error) {
	if client.Timeout <= 0 {
		return client.Do(request)
	}
	ctx, cancel := context.WithCancel(request.Context())
	timer := time.AfterFunc(client.Timeout, cancel)
	unlimited := *client
	unlimited.Timeout = 0
	resp, err := unlimited.Do(request.WithContext(ctx))
	if err != nil {
		cancel()
		return nil, err
	}
	if !timer.Stop() {
		resp.Body.Close()
		cancel()
		return nil, fmt.Errorf("backend response headers: %w", context.DeadlineExceeded)
	}
	resp.Body = cancelOnClose{resp.Body, cancel}
	return resp, nil
}

type cancelOnClose struct {
	io.ReadCloser
	cancel context.CancelFunc
}

func (c cancelOnClose) Close() error {
	err := c.ReadCloser.Close()
	c.cancel()
	return err
}

// streamStatusError is a non-2xx reply seen while opening a backend stream.
// Providers wrap it so the caller sees both the provider and the HTTP status.
type streamStatusError struct {
	status int
	body   []byte
}

func (e *streamStatusError) Error() string {
	return fmt.Sprintf("backend returned %d: %s", e.status, e.body)
}

type streamTool struct{ id, name, args string }

func streamSSE(ctx context.Context, client *http.Client, url string, body map[string]any, req *ChatRequest, protocol string, headers func(http.Header), emit func(string) error) (*ChatResponse, error) {
	body["stream"] = true
	var resp *http.Response
	for {
		raw, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		request.Header.Set("Content-Type", "application/json")
		headers(request.Header)
		resp, err = streamDo(client, request)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			break
		}
		raw, err = io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			// Keep the status: a cut-off error body is still a non-2xx reply.
			raw = fmt.Appendf(raw, " (body read failed: %v)", err)
		}
		effortField := "reasoning_effort"
		if protocol == "messages" {
			effortField = "output_config"
		}
		if resp.StatusCode == http.StatusBadRequest && body[effortField] != nil {
			log.Printf("model %q rejected reasoning effort %q; retrying without effort", req.Model, req.Effort)
			delete(body, effortField)
			continue
		}
		return nil, &streamStatusError{status: resp.StatusCode, body: raw}
	}
	defer resp.Body.Close()
	result := &ChatResponse{}
	tools := map[int]*streamTool{}
	reader := bufio.NewReader(resp.Body)
	data := []string{}
	terminated := false
	process := func(payload string) error {
		if payload == "[DONE]" {
			terminated = true
			return nil
		}
		var ev struct {
			Type    string          `json:"type"`
			Index   int             `json:"index"`
			Error   json.RawMessage `json:"error"`
			Message struct {
				Usage json.RawMessage `json:"usage"`
			} `json:"message"`
			Block struct {
				Type  string          `json:"type"`
				ID    string          `json:"id"`
				Name  string          `json:"name"`
				Input json.RawMessage `json:"input"`
				Text  string          `json:"text"`
			} `json:"content_block"`
			Delta struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
			Usage   json.RawMessage `json:"usage"`
			Choices []struct {
				Index int `json:"index"`
				Delta struct {
					Content   string `json:"content"`
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			return fmt.Errorf("parse stream event: %w", err)
		}
		if ev.Type == "error" || len(ev.Error) > 0 && string(ev.Error) != "null" {
			return fmt.Errorf("backend stream error: %s", ev.Error)
		}
		text := ""
		if protocol == "messages" {
			if ev.Type == "message_stop" {
				terminated = true
			}
			if ev.Type == "content_block_start" {
				if ev.Block.Type == "tool_use" {
					arg := string(ev.Block.Input)
					if arg == "{}" || arg == "null" {
						arg = ""
					}
					tools[ev.Index] = &streamTool{id: ev.Block.ID, name: ev.Block.Name, args: arg}
				}
				text = ev.Block.Text
			}
			if ev.Type == "content_block_delta" {
				if ev.Delta.Type == "text_delta" {
					text = ev.Delta.Text
				}
				if ev.Delta.Type == "input_json_delta" {
					t := tools[ev.Index]
					if t == nil {
						return fmt.Errorf("tool delta without start")
					}
					t.args += ev.Delta.PartialJSON
				}
			}
			if len(ev.Message.Usage) > 0 {
				mergeMessageUsage(&result.Usage, ev.Message.Usage)
			}
			if len(ev.Usage) > 0 {
				mergeMessageUsage(&result.Usage, ev.Usage)
			}
		} else {
			for _, c := range ev.Choices {
				if c.Index != 0 {
					continue
				}
				text += c.Delta.Content
				for _, call := range c.Delta.ToolCalls {
					t := tools[call.Index]
					if t == nil {
						t = &streamTool{}
						tools[call.Index] = t
					}
					t.id += call.ID
					t.name += call.Function.Name
					t.args += call.Function.Arguments
				}
			}
			if len(ev.Usage) > 0 {
				mergeChatUsage(&result.Usage, ev.Usage)
			}
		}
		if text != "" {
			result.Content += text
			return emit(text)
		}
		return nil
	}
	for !terminated {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return nil, err
		}
		line = strings.TrimSuffix(strings.TrimSuffix(line, "\n"), "\r")
		if line == "" && len(data) > 0 {
			if e := process(strings.Join(data, "\n")); e != nil {
				return nil, e
			}
			data = nil
		} else if strings.HasPrefix(line, "data:") {
			v := strings.TrimPrefix(line, "data:")
			v = strings.TrimPrefix(v, " ")
			data = append(data, v)
		}
		if err == io.EOF {
			if len(data) > 0 {
				if e := process(strings.Join(data, "\n")); e != nil {
					return nil, e
				}
			}
			break
		}
	}
	if !terminated {
		return nil, fmt.Errorf("backend stream ended before completion")
	}
	indices := make([]int, 0, len(tools))
	for i := range tools {
		indices = append(indices, i)
	}
	sort.Ints(indices)
	for _, i := range indices {
		t := tools[i]
		if t.args == "" {
			t.args = "{}"
		}
		if !json.Valid([]byte(t.args)) {
			return nil, fmt.Errorf("invalid tool arguments")
		}
		result.ToolCalls = append(result.ToolCalls, ToolCall{ID: t.id, Name: t.name, Arguments: json.RawMessage(t.args)})
	}
	return result, nil
}
func mergeMessageUsage(dst **Usage, raw json.RawMessage) {
	var u struct {
		Input  *int `json:"input_tokens"`
		Create *int `json:"cache_creation_input_tokens"`
		Read   *int `json:"cache_read_input_tokens"`
		Output *int `json:"output_tokens"`
	}
	if json.Unmarshal(raw, &u) != nil || u.Input == nil && u.Create == nil && u.Read == nil && u.Output == nil {
		return
	}
	if *dst == nil {
		*dst = &Usage{}
	}
	if u.Input != nil || u.Create != nil || u.Read != nil {
		v := 0
		if u.Input != nil {
			v += *u.Input
		}
		if u.Create != nil {
			v += *u.Create
		}
		if u.Read != nil {
			v += *u.Read
			(*dst).CachedInputTokens = *u.Read
		}
		(*dst).InputTokens = v
	}
	if u.Output != nil {
		(*dst).OutputTokens = *u.Output
	}
}
func mergeChatUsage(dst **Usage, raw json.RawMessage) {
	var u struct {
		Input   *int `json:"prompt_tokens"`
		Output  *int `json:"completion_tokens"`
		Details *struct {
			Cached *int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	}
	if json.Unmarshal(raw, &u) != nil || u.Input == nil && u.Output == nil && (u.Details == nil || u.Details.Cached == nil) {
		return
	}
	if *dst == nil {
		*dst = &Usage{}
	}
	if u.Input != nil {
		(*dst).InputTokens = *u.Input
	}
	if u.Output != nil {
		(*dst).OutputTokens = *u.Output
	}
	if u.Details != nil && u.Details.Cached != nil {
		(*dst).CachedInputTokens = *u.Details.Cached
	}
}

// mergeChatUsageStrict is the buffered form of mergeChatUsage: a usage value
// with the wrong JSON shape or field types is an error rather than
// silently-dropped usage data. Fields may be omitted or null.
func mergeChatUsageStrict(dst **Usage, raw json.RawMessage) error {
	var shape struct {
		Input   *int `json:"prompt_tokens"`
		Output  *int `json:"completion_tokens"`
		Details *struct {
			Cached *int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	}
	if err := json.Unmarshal(raw, &shape); err != nil {
		return fmt.Errorf("parse usage: %w", err)
	}
	mergeChatUsage(dst, raw)
	return nil
}

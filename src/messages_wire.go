package main

import "encoding/json"

// Zen's messages protocol sends the Anthropic Messages wire shape.
type (
	anthropicBlock struct {
		Type         string                 `json:"type"`
		Text         string                 `json:"text,omitempty"`
		ID           string                 `json:"id,omitempty"`
		Name         string                 `json:"name,omitempty"`
		Input        json.RawMessage        `json:"input,omitempty"`
		ToolUseID    string                 `json:"tool_use_id,omitempty"`
		Content      *string                `json:"content,omitempty"`
		CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
	}

	anthropicCacheControl struct {
		Type string `json:"type"`
	}

	anthropicOutputConfig struct {
		Effort string `json:"effort"`
	}

	anthropicMessage struct {
		Role    string           `json:"role"`
		Content []anthropicBlock `json:"content"`
	}

	anthropicTool struct {
		Name         string                 `json:"name"`
		Description  string                 `json:"description,omitempty"`
		InputSchema  json.RawMessage        `json:"input_schema"`
		CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
	}
)

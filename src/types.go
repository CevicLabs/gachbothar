package main

import "encoding/json"

// ToolSpecFunction is the function definition inside a tool spec.
type ToolSpecFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ToolSpec is a tool definition. Both chat_completions and Ollama native use
// the same wire shape for definitions, so this type is marshaled/unmarshaled
// directly on both sides.
type ToolSpec struct {
	Type     string           `json:"type"`
	Function ToolSpecFunction `json:"function"`
}

// ToolCall is a single function call made by the model. Arguments is a decoded
// JSON object literal (e.g. {"city":"Dublin"}), not the OpenAI wire's
// JSON-encoded string.
type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ChatMessage is the unified internal representation of a chat turn.
//
// The tool-related fields are deliberately tagged json:"-" because
// ChatMessage is directly marshaled/unmarshaled by the two wire formats in
// dialect_ollama.go and provider_ollama.go, and those formats use different
// shapes for tool calls. Tagging them json:"-" prevents a stray direct
// json.Marshal/Unmarshal of ChatMessage from silently emitting or reading
// the wrong shape. Each dialect and provider builds its own local wire-shaped
// structs and converts by hand.
type ChatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"-"`
	ToolCallID string     `json:"-"`
}

// ChatRequest is the normalized request produced by a dialect adapter.
// Provider and Model are extracted from the caller's "model" field, which
// must be formatted as "provider/model". Stream requests incremental output;
// providers without streaming support retain the buffered response path.
type ChatRequest struct {
	Provider string
	Model    string
	Messages []ChatMessage
	Stream   bool
	Tools    []ToolSpec
	Effort   string
	// MaxTokens is the client's output-token limit; 0 means the client set
	// none.
	MaxTokens int
}

// ChatResponse is the normalized response returned by a provider adapter.
// Stream is copied from the originating ChatRequest by main.go so the
// dialect adapter's WriteResponse can pick the right framing.
type ChatResponse struct {
	Content   string
	Stream    bool
	Model     string
	ToolCalls []ToolCall
	Usage     *Usage
}

// Usage reports token counts for providers that expose them.
type Usage struct {
	InputTokens       int
	CachedInputTokens int
	OutputTokens      int
}

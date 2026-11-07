package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// ProviderConfig holds the per-provider settings read from providers.json.
// Type picks the ProviderAdapter constructor from the registry in
// provider.go (empty defaults to "ollama", so existing config files with no
// "type" field keep working unchanged). Protocol is only meaningful for
// type "zen": "chat" (openai-chat, the default) or "messages"
// (anthropic-messages - needed for e.g. the qwen3.5-* family, which Zen
// serves over that protocol despite the model name giving no hint of it).
type ProviderConfig struct {
	BaseURL            string `json:"base_url"`
	APIKeyEnv          string `json:"api_key_env"`
	Type               string `json:"type,omitempty"`
	Protocol           string `json:"protocol,omitempty"`
	AutoPull           bool   `json:"auto_pull,omitempty"`
	PullTimeoutSeconds int    `json:"pull_timeout_seconds,omitempty"`
	// Models are the names GET /v1/models lists, without the provider prefix.
	Models []string `json:"models,omitempty"`
}

// Config is the top-level configuration shape. Host and Port are optional -
// see main.go for the real precedence (HOST/PORT env vars win over these,
// which win over the built-in defaults). Host defaults to 127.0.0.1, not
// all interfaces - binding wide open with no caller auth was a real,
// live exposure, measured 22 Sep 2026, not a hypothetical one.
type Config struct {
	Providers map[string]ProviderConfig `json:"providers"`
	Host      string                    `json:"host,omitempty"`
	Port      string                    `json:"port,omitempty"`
}

// LoadConfig reads providers.json from path and unmarshals it into Config.
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config %q: %w", path, err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config %q: %w", path, err)
	}
	return &cfg, nil
}

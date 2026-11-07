package main

import (
	"context"
	"fmt"
)

// ProviderAdapter forwards a normalized ChatRequest to a backend and returns
// a normalized ChatResponse.
type ProviderAdapter interface {
	Name() string
	Call(req *ChatRequest) (*ChatResponse, error)
}

// StreamingProvider is optional; unsupported providers retain buffered replies.
type StreamingProvider interface {
	Stream(ctx context.Context, req *ChatRequest, emit func(string) error) (*ChatResponse, error)
}

// providerBuilders maps a config "type" string to the constructor for that
// ProviderAdapter. This is the actual modularity point: adding a new
// upstream (a new wire protocol gachbothar speaks OUT to a backend) means
// adding one file and one registry entry here, never touching BuildProviders
// itself - the same pattern dialect.go already uses for incoming protocols.
var providerBuilders = map[string]func(name string, pc ProviderConfig) (ProviderAdapter, error){
	"anthropic": func(name string, pc ProviderConfig) (ProviderAdapter, error) {
		return NewAnthropicProvider(name, pc.BaseURL, pc.APIKeyEnv), nil
	},
	"ollama": func(name string, pc ProviderConfig) (ProviderAdapter, error) {
		return NewOllamaProvider(name, pc.BaseURL, pc.APIKeyEnv, pc.AutoPull, pc.PullTimeoutSeconds), nil
	},
	"zen": func(name string, pc ProviderConfig) (ProviderAdapter, error) {
		return NewZenProvider(name, pc.BaseURL, pc.APIKeyEnv, pc.Protocol)
	},
	"openai": func(name string, pc ProviderConfig) (ProviderAdapter, error) {
		return NewOpenAIProvider(name, pc.BaseURL, pc.APIKeyEnv), nil
	},
}

// BuildProviders creates a provider registry from the loaded configuration.
// A config entry with no "type" defaults to "ollama" for backward
// compatibility with providers.json files written before Type existed.
func BuildProviders(cfg *Config) (map[string]ProviderAdapter, error) {
	providers := make(map[string]ProviderAdapter, len(cfg.Providers))
	for name, pc := range cfg.Providers {
		if pc.BaseURL == "" {
			return nil, fmt.Errorf("provider %q missing base_url", name)
		}
		typ := pc.Type
		if typ == "" {
			typ = "ollama"
		}
		build, ok := providerBuilders[typ]
		if !ok {
			return nil, fmt.Errorf("provider %q has unknown type %q", name, typ)
		}
		provider, err := build(name, pc)
		if err != nil {
			return nil, fmt.Errorf("build provider %q: %w", name, err)
		}
		providers[name] = provider
	}
	return providers, nil
}

// tokenLimitOr uses the client's output-token limit when one was set and a
// provider's fixed default otherwise.
func tokenLimitOr(maxTokens, fallback int) int {
	if maxTokens != 0 {
		return maxTokens
	}
	return fallback
}

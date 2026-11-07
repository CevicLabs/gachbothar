package main

import (
	"strings"
	"testing"
)

func TestBuildProvidersRejectsUnknownAnthropicType(t *testing.T) {
	cfg := &Config{Providers: map[string]ProviderConfig{
		"anthropic": {Type: "anthropic", BaseURL: "https://api.anthropic.com", APIKeyEnv: "ANTHROPIC_API_KEY"},
	}}
	_, err := BuildProviders(cfg)
	if err == nil {
		t.Fatal("expected an error for the removed anthropic provider type")
	}
	if !strings.Contains(err.Error(), `unknown type "anthropic"`) {
		t.Fatalf("error = %v, want the existing unknown-type wording", err)
	}
}

package main

import (
	"encoding/json"
	"log"
	"net/http"
	"sort"
)

// modelsPath is where Codex looks for its model list, under a base_url ending in /v1.
const modelsPath = "/v1/models"

// truncationPolicy and modelEntry follow Codex's ModelInfo
// (codex-rs/protocol/src/openai_models.rs). Fields without a pointer or
// omitempty are required there, or decoding the whole list fails, so they stay
// in the JSON even when null or empty.
type truncationPolicy struct {
	Mode  string `json:"mode"`
	Limit int    `json:"limit"`
}

type modelEntry struct {
	Slug                       string           `json:"slug"`
	DisplayName                string           `json:"display_name"`
	Description                *string          `json:"description"`
	SupportedReasoningLevels   []string         `json:"supported_reasoning_levels"`
	ShellType                  string           `json:"shell_type"`
	Visibility                 string           `json:"visibility"`
	SupportedInAPI             bool             `json:"supported_in_api"`
	Priority                   int              `json:"priority"`
	AvailabilityNux            *string          `json:"availability_nux"`
	Upgrade                    *string          `json:"upgrade"`
	SupportVerbosity           bool             `json:"support_verbosity"`
	DefaultVerbosity           *string          `json:"default_verbosity"`
	ApplyPatchToolType         *string          `json:"apply_patch_tool_type"`
	TruncationPolicy           truncationPolicy `json:"truncation_policy"`
	ExperimentalSupportedTools []string         `json:"experimental_supported_tools"`
	BaseInstructions           string           `json:"base_instructions"`
}

// listModels names every model listed in the config as provider/model, ordered
// by provider, then as listed. Priority follows that order.
func listModels(cfg *Config) []modelEntry {
	names := make([]string, 0, len(cfg.Providers))
	for name := range cfg.Providers {
		names = append(names, name)
	}
	sort.Strings(names)

	entries := []modelEntry{}
	for _, name := range names {
		for _, model := range cfg.Providers[name].Models {
			entries = append(entries, modelEntry{
				Slug:                       name + "/" + model,
				DisplayName:                model + " (gachbothar)",
				SupportedReasoningLevels:   []string{},
				ShellType:                  "default",
				Visibility:                 "list",
				SupportedInAPI:             true,
				Priority:                   len(entries),
				TruncationPolicy:           truncationPolicy{Mode: "tokens", Limit: 10000},
				ExperimentalSupportedTools: []string{},
			})
		}
	}
	return entries
}

// modelsHandler answers Codex's GET /v1/models?client_version=... with
// {"models":[...]}, which is Codex's own shape, not OpenAI's {"data":[...]}.
func modelsHandler(cfg *Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{"models": listModels(cfg)}); err != nil {
			log.Printf("write models response: %v", err)
		}
	}
}

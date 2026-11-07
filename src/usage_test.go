package main

import (
	"reflect"
	"testing"
)

// Gate for T1: every provider reports usage, and the Chat Completions and Ollama
// dialects write it in their own shapes.

func TestOllamaProviderReportsUsage(t *testing.T) {
	resp := usageCall(t, usageOllama(t, `{"message":{"role":"assistant","content":"ok"},"done":true,"prompt_eval_count":11,"eval_count":7}`))
	if resp.Content != "ok" {
		t.Errorf("content = %q", resp.Content)
	}
	usageWant(t, resp.Usage, Usage{InputTokens: 11, OutputTokens: 7})
}

func TestZenChatReportsUsage(t *testing.T) {
	resp := usageCall(t, usageZen(t, "chat", `{"choices":[{"message":{"role":"assistant","content":"ok"}}],
		"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25,"prompt_tokens_details":{"cached_tokens":8}}}`))
	usageWant(t, resp.Usage, Usage{InputTokens: 20, CachedInputTokens: 8, OutputTokens: 5})
}

func TestZenMessagesReportsUsage(t *testing.T) {
	resp := usageCall(t, usageZen(t, "messages", `{"content":[{"type":"text","text":"ok"}],
		"usage":{"input_tokens":3,"cache_creation_input_tokens":4,"cache_read_input_tokens":10,"output_tokens":6}}`))
	usageWant(t, resp.Usage, Usage{InputTokens: 17, CachedInputTokens: 10, OutputTokens: 6})
}

func TestChatCompletionsWritesUsage(t *testing.T) {
	out := usageJSON(t, usageWrite(t, chatCompletionsDialect{}, &ChatResponse{Content: "ok", Usage: usageSample}))
	want := map[string]any{
		"prompt_tokens": 17.0, "completion_tokens": 6.0, "total_tokens": 23.0,
		"prompt_tokens_details": map[string]any{"cached_tokens": 10.0},
	}
	if !reflect.DeepEqual(out["usage"], any(want)) {
		t.Errorf("usage = %#v", out["usage"])
	}
}

func TestOllamaDialectWritesUsage(t *testing.T) {
	out := usageJSON(t, usageWrite(t, ollamaNativeDialect{}, &ChatResponse{Content: "ok", Usage: usageSample}))
	if out["prompt_eval_count"] != 17.0 || out["eval_count"] != 6.0 {
		t.Errorf("out = %#v", out)
	}
}

package main

import (
	"reflect"
	"strings"
	"testing"
)

func TestNoBackendUsageMeansNoUsage(t *testing.T) {
	if r := usageCall(t, usageOllama(t, `{"message":{"role":"assistant","content":"ok"},"done":true}`)); r.Usage != nil {
		t.Errorf("ollama: %+v", r.Usage)
	}
	if r := usageCall(t, usageZen(t, "chat", `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)); r.Usage != nil {
		t.Errorf("zen chat: %+v", r.Usage)
	}
	if r := usageCall(t, usageZen(t, "messages", `{"content":[{"type":"text","text":"ok"}]}`)); r.Usage != nil {
		t.Errorf("zen messages: %+v", r.Usage)
	}
}

// Ollama leaves prompt_eval_count out when it reused a cached prompt.
func TestOllamaOutputCountAloneStillCounts(t *testing.T) {
	r := usageCall(t, usageOllama(t, `{"message":{"role":"assistant","content":"ok"},"eval_count":7}`))
	usageWant(t, r.Usage, Usage{OutputTokens: 7})
}

func TestZenChatWithoutCacheDetails(t *testing.T) {
	r := usageCall(t, usageZen(t, "chat", `{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":20,"completion_tokens":5}}`))
	usageWant(t, r.Usage, Usage{InputTokens: 20, OutputTokens: 5})
}

func TestZenMessagesWithoutCacheFields(t *testing.T) {
	r := usageCall(t, usageZen(t, "messages", `{"content":[],"usage":{"input_tokens":3,"output_tokens":6}}`))
	usageWant(t, r.Usage, Usage{InputTokens: 3, OutputTokens: 6})
}

func TestUsageComesWithToolCalls(t *testing.T) {
	o := usageCall(t, usageOllama(t, `{"message":{"role":"assistant","tool_calls":[{"function":{"name":"n","arguments":{}}}]},"prompt_eval_count":1,"eval_count":2}`))
	if len(o.ToolCalls) != 1 {
		t.Errorf("ollama calls %+v", o.ToolCalls)
	}
	usageWant(t, o.Usage, Usage{InputTokens: 1, OutputTokens: 2})
	z := usageCall(t, usageZen(t, "messages", `{"content":[{"type":"tool_use","id":"a","name":"n","input":{}}],"usage":{"input_tokens":3,"output_tokens":6}}`))
	if len(z.ToolCalls) != 1 {
		t.Errorf("zen calls %+v", z.ToolCalls)
	}
	usageWant(t, z.Usage, Usage{InputTokens: 3, OutputTokens: 6})
	c := usageJSON(t, usageWrite(t, chatCompletionsDialect{}, &ChatResponse{ToolCalls: []ToolCall{{ID: "a", Name: "n"}}, Usage: usageSample}))
	if u, _ := c["usage"].(map[string]any); u == nil || u["total_tokens"] != 23.0 {
		t.Errorf("chat completions: %#v", c["usage"])
	}
}

func TestChatCompletionsWithoutUsageStaysAsItWas(t *testing.T) {
	if out := usageJSON(t, usageWrite(t, chatCompletionsDialect{}, &ChatResponse{Content: "ok"})); out["usage"] != nil {
		t.Errorf("non-stream: %#v", out)
	} else if _, ok := out["usage"]; ok {
		t.Errorf("non-stream has a usage key: %#v", out)
	}
	if body := usageWrite(t, chatCompletionsDialect{}, &ChatResponse{Content: "ok", Stream: true, Model: "m"}); strings.Contains(body, "usage") {
		t.Errorf("stream: %s", body)
	}
}

// The usage goes on the finish chunk: a client that did not ask for a separate
// usage chunk may read choices[0] on every chunk.
func TestChatCompletionsStreamCarriesUsageOnTheFinishChunk(t *testing.T) {
	body := usageWrite(t, chatCompletionsDialect{}, &ChatResponse{Content: "ok", Stream: true, Model: "m", Usage: usageSample})
	if !strings.HasSuffix(strings.TrimSpace(body), "data: [DONE]") {
		t.Errorf("no terminator: %s", body)
	}
	seen := 0
	want := map[string]any{
		"prompt_tokens": 17.0, "completion_tokens": 6.0, "total_tokens": 23.0,
		"prompt_tokens_details": map[string]any{"cached_tokens": 10.0},
	}
	for _, chunk := range usageLines(t, body) {
		choices, _ := chunk["choices"].([]any)
		if len(choices) != 1 {
			t.Errorf("chunk without one choice: %#v", chunk)
			continue
		}
		finish := choices[0].(map[string]any)["finish_reason"]
		usage, has := chunk["usage"]
		if has {
			seen++
			if finish != "stop" {
				t.Errorf("usage on a chunk with finish_reason %v", finish)
			}
			if !reflect.DeepEqual(usage, any(want)) {
				t.Errorf("usage = %#v", usage)
			}
		}
	}
	if seen != 1 {
		t.Errorf("usage on %d chunks: %s", seen, body)
	}
}

func TestOllamaDialectStreamCountsOnlyOnTheDoneChunk(t *testing.T) {
	lines := usageLines(t, usageWrite(t, ollamaNativeDialect{}, &ChatResponse{Content: "ok", Stream: true, Model: "m", Usage: usageSample}))
	if len(lines) != 2 {
		t.Fatalf("lines = %#v", lines)
	}
	if _, ok := lines[0]["eval_count"]; ok {
		t.Errorf("counts on the first chunk: %#v", lines[0])
	}
	if lines[1]["done"] != true || lines[1]["prompt_eval_count"] != 17.0 || lines[1]["eval_count"] != 6.0 {
		t.Errorf("done chunk: %#v", lines[1])
	}
}

func TestOllamaDialectWithoutUsageStaysAsItWas(t *testing.T) {
	for _, stream := range []bool{false, true} {
		body := usageWrite(t, ollamaNativeDialect{}, &ChatResponse{Content: "ok", Stream: stream, Model: "m"})
		if strings.Contains(body, "eval_count") {
			t.Errorf("stream %t: %s", stream, body)
		}
	}
}

func TestUsageReachesAClientEndToEnd(t *testing.T) {
	r := usageCall(t, usageZen(t, "chat", `{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":20,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":8}}}`))
	body := usageWrite(t, responsesDialect{}, r)
	if !strings.Contains(body, `"input_tokens":20`) || !strings.Contains(body, `"cached_tokens":8`) || !strings.Contains(body, `"output_tokens":5`) {
		t.Errorf("responses: %s", body)
	}
	r = usageCall(t, usageOllama(t, `{"message":{"content":"ok"},"prompt_eval_count":11,"eval_count":7}`))
	out := usageJSON(t, usageWrite(t, chatCompletionsDialect{}, r))
	if u, _ := out["usage"].(map[string]any); u == nil || u["total_tokens"] != 18.0 {
		t.Errorf("chat completions: %#v", out)
	}
}

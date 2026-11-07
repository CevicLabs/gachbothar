package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// T8 gate: the client's output-token limit survives translation on both
// boundaries. These tests fail before the change because ChatRequest carries
// no limit at all.

func tokensParse(t *testing.T, dialect DialectAdapter, body string) (*ChatRequest, error) {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, dialect.RoutePath(), strings.NewReader(body))
	return dialect.ParseRequest(r)
}

func tokensHTTP400(t *testing.T, dialect DialectAdapter, body string, field string) {
	t.Helper()
	if _, err := tokensParse(t, dialect, body); err == nil {
		t.Fatalf("parsed an invalid %s without an error", field)
	} else if !strings.Contains(err.Error(), field) {
		t.Errorf("error does not name %s: %v", field, err)
	}
}

func TestMaxTokensParse_ChatCompletions(t *testing.T) {
	for _, tt := range []struct {
		name, body string
		want       int
	}{
		{"prefers max_completion_tokens", `{"model":"p/m","messages":[],"max_completion_tokens":12,"max_tokens":77}`, 12},
		{"falls back to max_tokens", `{"model":"p/m","messages":[],"max_tokens":77}`, 77},
		{"unset", `{"model":"p/m","messages":[]}`, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			req, err := tokensParse(t, chatCompletionsDialect{}, tt.body)
			if err != nil {
				t.Fatal(err)
			}
			if req.MaxTokens != tt.want {
				t.Errorf("req.MaxTokens = %d, want %d", req.MaxTokens, tt.want)
			}
		})
	}
}

func TestMaxTokensParse_Responses(t *testing.T) {
	for _, tt := range []struct {
		name, body string
		want       int
	}{
		{"reads max_output_tokens", `{"model":"p/m","input":[],"max_output_tokens":77}`, 77},
		{"unset", `{"model":"p/m","input":[]}`, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			req, err := tokensParse(t, responsesDialect{}, tt.body)
			if err != nil {
				t.Fatal(err)
			}
			if req.MaxTokens != tt.want {
				t.Errorf("req.MaxTokens = %d, want %d", req.MaxTokens, tt.want)
			}
		})
	}
}

func TestMaxTokensParse_OllamaNative(t *testing.T) {
	for _, tt := range []struct {
		name, body string
		want       int
	}{
		{"reads num_predict", `{"model":"p/m","messages":[],"options":{"num_predict":77,"temperature":0.2},"stream":false}`, 77},
		{"no options at all", `{"model":"p/m","messages":[],"stream":false}`, 0},
		{"options without num_predict", `{"model":"p/m","messages":[],"options":{"temperature":0.2},"stream":false}`, 0},
		{"zero is Ollama's no limit", `{"model":"p/m","messages":[],"options":{"num_predict":0},"stream":false}`, 0},
		{"negative is Ollama's no limit", `{"model":"p/m","messages":[],"options":{"num_predict":-5},"stream":false}`, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			req, err := tokensParse(t, ollamaNativeDialect{}, tt.body)
			if err != nil {
				t.Fatal(err)
			}
			if req.MaxTokens != tt.want {
				t.Errorf("req.MaxTokens = %d, want %d", req.MaxTokens, tt.want)
			}
		})
	}
}

func TestMaxTokensParse_RejectsInvalidLimits(t *testing.T) {
	for _, bad := range []string{"0", "-5", "0.5", "77.5"} {
		tokensHTTP400(t, chatCompletionsDialect{}, `{"model":"p/m","messages":[],"max_completion_tokens":`+bad+`}`, "max_completion_tokens")
		tokensHTTP400(t, chatCompletionsDialect{}, `{"model":"p/m","messages":[],"max_tokens":`+bad+`}`, "max_tokens")
		tokensHTTP400(t, responsesDialect{}, `{"model":"p/m","input":[],"max_output_tokens":`+bad+`}`, "max_output_tokens")
	}
	// Ollama only refuses fractionals; zero and negative are its own "no limit".
	for _, bad := range []string{"0.5", "77.5"} {
		tokensHTTP400(t, ollamaNativeDialect{}, `{"model":"p/m","messages":[],"options":{"num_predict":`+bad+`},"stream":false}`, "options.num_predict")
	}
}

// tokensCapture records every JSON request body a fake backend receives.
type tokensCapture struct {
	bodies []map[string]any
}

func tokensReadBody(t *testing.T, r *http.Request, got *tokensCapture) {
	t.Helper()
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		t.Errorf("read backend body: %v", err)
		return
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Errorf("backend request body: %v (%s)", err, raw)
		return
	}
	got.bodies = append(got.bodies, body)
}

const tokensOpenAIReply = `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`

const tokensMessagesReply = `{"content":[{"type":"text","text":"ok"}]}`

const tokensOllamaReply = `{"message":{"role":"assistant","content":"ok"},"done":true}`

const tokensSSEChatLine = `data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}` + "\n\ndata: [DONE]\n\n"

const tokensSSEMessagesStop = "data: {\"type\":\"message_stop\"}\n\n"

type tokensProviderCase struct {
	name    string
	backend string // "chat" (OpenAI chat), "messages", "ollama"
}

var tokensProviderCases = []tokensProviderCase{
	{"openai", "chat"},
	{"zen chat", "chat"},
	{"zen messages", "messages"},
	{"ollama", "ollama"},
}

// tokensServer captures request bodies and answers the wire shape the tested
// provider expects, buffered or streamed, based on what it received.
func tokensServer(t *testing.T, backend string, got *tokensCapture) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tokensReadBody(t, r, got)
		stream, _ := got.bodies[len(got.bodies)-1]["stream"].(bool)
		w.Header().Set("Content-Type", "application/json")
		var reply string
		switch {
		case backend == "ollama":
			w.Header().Set("Content-Type", "application/x-ndjson")
			reply = tokensOllamaReply + "\n"
		case backend == "messages" && stream:
			reply = tokensSSEMessagesStop
		case backend == "messages":
			reply = tokensMessagesReply
		case stream:
			reply = tokensSSEChatLine
		default:
			reply = tokensOpenAIReply
		}
		_, _ = w.Write([]byte(reply))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func tokensNewProvider(t *testing.T, name, url string) ProviderAdapter {
	t.Helper()
	switch name {
	case "openai":
		return newTestOpenAI(t, url, "")
	case "zen chat":
		return newTestZen(t, url)
	case "zen messages":
		return newTestZenMessages(t, url)
	case "ollama":
		return NewOllamaProvider("ollama", url, "", false, 0)
	}
	t.Fatalf("unknown provider case %q", name)
	return nil
}

func tokensLimitInBody(t *testing.T, name string, body map[string]any, want float64) {
	t.Helper()
	switch name {
	case "ollama":
		options, _ := body["options"].(map[string]any)
		if got, _ := options["num_predict"].(float64); got != want {
			t.Errorf("options.num_predict = %v, want %v", options["num_predict"], want)
		}
	default:
		if got, _ := body["max_tokens"].(float64); got != want {
			t.Errorf("max_tokens = %v, want %v", body["max_tokens"], want)
		}
	}
}

func TestMaxTokensProvidersForwardLimitBufferedAndStreamed(t *testing.T) {
	for _, tc := range tokensProviderCases {
		t.Run(tc.name, func(t *testing.T) {
			got := &tokensCapture{}
			srv := tokensServer(t, tc.backend, got)
			req := &ChatRequest{
				Model:     "m",
				Effort:    "high", // checked below for ollama's think, and it must not interfere
				MaxTokens: 77,
				Messages:  []ChatMessage{{Role: "user", Content: "hi"}},
			}
			if _, err := tokensNewProvider(t, tc.name, srv.URL).Call(req); err != nil {
				t.Fatalf("Call: %v", err)
			}
			if len(got.bodies) != 1 {
				t.Fatalf("%d buffered calls, want 1", len(got.bodies))
			}
			tokensLimitInBody(t, tc.name, got.bodies[0], 77)

			streamGot := &tokensCapture{}
			streamSrv := tokensServer(t, tc.backend, streamGot)
			provider := tokensNewProvider(t, tc.name, streamSrv.URL)
			sp, ok := provider.(StreamingProvider)
			if !ok {
				t.Fatalf("provider %s does not stream", tc.name)
			}
			if _, err := sp.Stream(context.Background(), req, func(string) error { return nil }); err != nil {
				t.Fatalf("Stream: %v", err)
			}
			if len(streamGot.bodies) != 1 {
				t.Fatalf("%d streamed calls, want 1", len(streamGot.bodies))
			}
			tokensLimitInBody(t, tc.name, streamGot.bodies[0], 77)
			if tc.name == "ollama" {
				if got, wantField := streamGot.bodies[0]["think"], "high"; got != wantField {
					t.Errorf("options forwarding changed think: %v, want %v", got, wantField)
				}
			}
		})
	}
}

func TestMaxTokensProvidersDefaultsWhenUnset(t *testing.T) {
	for _, tc := range tokensProviderCases {
		t.Run(tc.name, func(t *testing.T) {
			got := &tokensCapture{}
			srv := tokensServer(t, tc.backend, got)
			req := &ChatRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}}
			if _, err := tokensNewProvider(t, tc.name, srv.URL).Call(req); err != nil {
				t.Fatalf("Call: %v", err)
			}
			if len(got.bodies) != 1 {
				t.Fatalf("%d buffered calls, want 1", len(got.bodies))
			}
			body := got.bodies[0]
			switch tc.name {
			case "openai", "zen chat":
				if _, present := body["max_tokens"]; present {
					t.Errorf("max_tokens sent with no limit: %v", body["max_tokens"])
				}
			case "ollama":
				if _, present := body["options"]; present {
					t.Errorf("options sent with no limit: %v", body["options"])
				}
			case "zen messages":
				if got, _ := body["max_tokens"].(float64); got != 4096 {
					t.Errorf("max_tokens = %v, want the 4096 default", body["max_tokens"])
				}
			}
		})
	}
}

func TestMaxTokensSurvivesEffortRetry(t *testing.T) {
	for _, tc := range []struct {
		name  string
		build func(t *testing.T, url string) ProviderAdapter
		reply string
	}{
		{"ollama", func(t *testing.T, url string) ProviderAdapter {
			return NewOllamaProvider("ollama", url, "", false, 0)
		}, tokensOllamaReply},
	} {
		t.Run(tc.name, func(t *testing.T) {
			url, calls := effortServer(t, func(n int) (int, string) {
				if n == 1 {
					return http.StatusBadRequest, `{"error":"this model does not support that"}`
				}
				return http.StatusOK, tc.reply
			})
			req := effortRequest("high")
			req.MaxTokens = 77
			if _, err := tc.build(t, url).Call(req); err != nil {
				t.Fatalf("Call: %v", err)
			}
			if len(calls.bodies) != 2 {
				t.Fatalf("%d calls, want 2", len(calls.bodies))
			}
			tokensLimitInBody(t, tc.name, calls.bodies[0], 77)
			tokensLimitInBody(t, tc.name, calls.bodies[1], 77)
		})
	}
}

func TestMaxTokensFromChatCompletionsReachesOpenAI(t *testing.T) {
	got := &tokensCapture{}
	srv := tokensServer(t, "chat", got)
	req, err := tokensParse(t, chatCompletionsDialect{}, `{"model":"openai/m","messages":[{"role":"user","content":"hi"}],"max_tokens":77}`)
	if err != nil {
		t.Fatal(err)
	}
	if req.MaxTokens != 77 {
		t.Fatalf("ParseRequest lost the limit: %d", req.MaxTokens)
	}
	if _, err := tokensNewProvider(t, "openai", srv.URL).Call(req); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if len(got.bodies) != 1 {
		t.Fatalf("%d calls, want 1", len(got.bodies))
	}
	tokensLimitInBody(t, "openai", got.bodies[0], 77)
}

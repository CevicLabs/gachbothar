package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// slowStream writes n lines, one every gap, so the whole body outlasts a short
// client timeout while each piece arrives well inside it.
func slowStream(n int, gap time.Duration, line func(i int) string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for i := 0; i < n; i++ {
			fmt.Fprint(w, line(i))
			w.(http.Flusher).Flush()
			time.Sleep(gap)
		}
	}))
}

func TestSSEStreamOutlastsClientTimeout(t *testing.T) {
	upstream := slowStream(6, 100*time.Millisecond, func(i int) string {
		if i == 5 {
			return "data: [DONE]\n\n"
		}
		return `data: {"choices":[{"index":0,"delta":{"content":"x"}}]}` + "\n\n"
	})
	defer upstream.Close()
	client := &http.Client{Timeout: 250 * time.Millisecond}
	resp, err := streamSSE(context.Background(), client, upstream.URL, map[string]any{}, &ChatRequest{Model: "m"}, "chat", func(http.Header) {}, func(string) error { return nil })
	if err != nil {
		t.Fatalf("long stream cut off: %v", err)
	}
	if resp.Content != "xxxxx" {
		t.Fatalf("content %q", resp.Content)
	}
}

func TestOllamaStreamOutlastsClientTimeout(t *testing.T) {
	upstream := slowStream(6, 100*time.Millisecond, func(i int) string {
		return fmt.Sprintf(`{"message":{"content":"x"},"done":%v}`+"\n", i == 5)
	})
	defer upstream.Close()
	p := NewOllamaProvider("test", upstream.URL, "", false, 0)
	p.client = &http.Client{Timeout: 250 * time.Millisecond}
	resp, err := p.Stream(context.Background(), &ChatRequest{Model: "m"}, func(string) error { return nil })
	if err != nil {
		t.Fatalf("long stream cut off: %v", err)
	}
	if resp.Content != "xxxxxx" {
		t.Fatalf("content %q", resp.Content)
	}
}

func TestStreamStillTimesOutWithoutHeaders(t *testing.T) {
	block := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { <-block }))
	defer upstream.Close()
	defer close(block)
	client := &http.Client{Timeout: 200 * time.Millisecond}
	start := time.Now()
	_, err := streamSSE(context.Background(), client, upstream.URL, map[string]any{}, &ChatRequest{Model: "m"}, "chat", func(http.Header) {}, func(string) error { return nil })
	if err == nil {
		t.Fatal("expected a timeout error")
	}
	if time.Since(start) > 2*time.Second {
		t.Fatalf("timeout not applied: took %v", time.Since(start))
	}
	if !strings.Contains(err.Error(), "deadline") && !strings.Contains(err.Error(), "Timeout") && !strings.Contains(err.Error(), "canceled") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Codex drops (or, in debug builds, panics on) output_text.delta unless a
// message item was opened first with response.output_item.added.
func TestResponsesStreamOpensItemBeforeDelta(t *testing.T) {
	w := httptest.NewRecorder()
	s := newStreamWriter(w, "responses", "m")
	if err := s.WriteText("a"); err != nil {
		t.Fatal(err)
	}
	if err := s.WriteText("b"); err != nil {
		t.Fatal(err)
	}
	body := w.Body.String()
	added := strings.Index(body, "event: response.output_item.added")
	delta := strings.Index(body, "event: response.output_text.delta")
	if added < 0 || delta < 0 || added > delta {
		t.Fatalf("output_item.added must precede the first delta:\n%s", body)
	}
	if strings.Count(body, "response.output_item.added") != 2 { // event line + type field, once
		t.Fatalf("item opened more than once:\n%s", body)
	}
}

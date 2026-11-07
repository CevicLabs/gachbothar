package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOllamaStreamsBeforeBackendCompletes(t *testing.T) {
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body = make([]byte, 4096)
		n, _ := r.Body.Read(body)
		if !strings.Contains(string(body[:n]), `"stream":true`) {
			t.Error("stream not requested")
		}
		fmt.Fprintln(w, `{"message":{"content":"first"},"done":false}`)
		w.(http.Flusher).Flush()
		<-release
		fmt.Fprintln(w, `{"message":{"content":"second"},"done":true,"prompt_eval_count":3,"eval_count":2}`)
	}))
	defer upstream.Close()
	p := NewOllamaProvider("test", upstream.URL, "", false, 0)
	first := make(chan string, 1)
	done := make(chan *ChatResponse, 1)
	errs := make(chan error, 1)
	go func() {
		r, err := p.Stream(context.Background(), &ChatRequest{Model: "m"}, func(s string) error { first <- s; return nil })
		if err != nil {
			errs <- err
			return
		}
		done <- r
	}()
	if got := <-first; got != "first" {
		t.Fatalf("got %q", got)
	}
	close(release)
	if got := <-first; got != "second" {
		t.Fatalf("got %q", got)
	}
	select {
	case r := <-done:
		if r.Content != "firstsecond" || r.Usage == nil || r.Usage.InputTokens != 3 || r.Usage.OutputTokens != 2 {
			t.Fatalf("bad response: %+v", r)
		}
	case err := <-errs:
		t.Fatal(err)
	}
}

func TestOllamaDialectPreservesTools(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"p/m","messages":[{"role":"assistant","content":"","tool_calls":[{"id":"a","function":{"name":"f","arguments":{"x":1}}}]}],"tools":[{"type":"function","function":{"name":"f"}}]}`))
	req, err := (ollamaNativeDialect{}).ParseRequest(r)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Tools) != 1 || len(req.Messages) != 1 || len(req.Messages[0].ToolCalls) != 1 || req.Messages[0].ToolCalls[0].Name != "f" {
		t.Fatalf("bad parse: %+v", req)
	}
	for _, stream := range []bool{false, true} {
		w := httptest.NewRecorder()
		err = (ollamaNativeDialect{}).WriteResponse(w, &ChatResponse{Stream: stream, ToolCalls: req.Messages[0].ToolCalls})
		if err != nil || !strings.Contains(w.Body.String(), `"tool_calls"`) || !strings.Contains(w.Body.String(), `"name":"f"`) {
			t.Fatalf("stream %v: %s (%v)", stream, w.Body.String(), err)
		}
	}
}

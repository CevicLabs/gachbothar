package main

import (
	"net/http"
	"testing"
)

func TestResponsesReadsEffort(t *testing.T) {
	cases := map[string]string{
		`{"effort":"low"}`:                   "low",
		`{"effort":"medium"}`:                "medium",
		`{"effort":"high","summary":"auto"}`: "high",
		`{"effort":"xhigh"}`:                 "xhigh",
		`{"effort":"max"}`:                   "max",
		`{"effort":"none"}`:                  "none",
		`{"effort":"minimal"}`:               "low",
		`{"effort":"bogus"}`:                 "",
		`{"summary":"auto"}`:                 "",
		"":                                   "",
	}
	for reasoning, want := range cases {
		if got := effortParse(t, reasoning).Effort; got != want {
			t.Errorf("reasoning %s: Effort = %q, want %q", reasoning, got, want)
		}
	}
}

func TestEffortReachesEachBackend(t *testing.T) {
	for _, route := range effortRoutes() {
		for _, effort := range []string{"", "none", "low", "medium", "high", "xhigh", "max"} {
			url, calls := effortServer(t, func(int) (int, string) { return http.StatusOK, route.reply })
			if _, err := route.build(t, url).Call(effortRequest(effort)); err != nil {
				t.Errorf("%s effort %q: %v", route.name, effort, err)
				continue
			}
			effortField(t, route.name+" effort "+effort, calls.bodies[0], route.field, route.want(effort))
		}
	}
}

func TestRefusedEffortIsDroppedAndRetriedOnce(t *testing.T) {
	for _, route := range effortRoutes() {
		url, calls := effortServer(t, func(n int) (int, string) {
			if n == 1 {
				return http.StatusBadRequest, `{"error":"this model does not support that"}`
			}
			return http.StatusOK, route.reply
		})
		resp, err := route.build(t, url).Call(effortRequest("high"))
		if err != nil {
			t.Errorf("%s: %v", route.name, err)
			continue
		}
		if resp.Content != "OK" {
			t.Errorf("%s: content = %q", route.name, resp.Content)
		}
		if len(calls.bodies) != 2 {
			t.Errorf("%s: %d calls, want 2", route.name, len(calls.bodies))
			continue
		}
		effortField(t, route.name+" first call", calls.bodies[0], route.field, route.want("high"))
		effortField(t, route.name+" second call", calls.bodies[1], route.field, nil)
	}
}

func TestOtherFailuresAreNotRetried(t *testing.T) {
	for _, route := range effortRoutes() {
		url, calls := effortServer(t, effortFails(http.StatusInternalServerError))
		if _, err := route.build(t, url).Call(effortRequest("high")); err == nil {
			t.Errorf("%s: no error on a 500", route.name)
		}
		if len(calls.bodies) != 1 {
			t.Errorf("%s: %d calls on a 500, want 1", route.name, len(calls.bodies))
		}
	}
}

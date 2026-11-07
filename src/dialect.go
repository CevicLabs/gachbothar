package main

import "net/http"

// DialectAdapter translates between a public wire protocol (e.g. OpenAI chat
// completions) and gachbothar's internal ChatRequest/ChatResponse types.
type DialectAdapter interface {
	Name() string
	RoutePath() string
	ParseRequest(r *http.Request) (*ChatRequest, error)
	WriteResponse(w http.ResponseWriter, resp *ChatResponse) error
}

var dialectRegistry = make(map[string]DialectAdapter)

// RegisterDialect adds a dialect adapter to the registry.
func RegisterDialect(d DialectAdapter) {
	dialectRegistry[d.Name()] = d
}

// GetDialect returns a registered dialect adapter by name.
func GetDialect(name string) (DialectAdapter, bool) {
	d, ok := dialectRegistry[name]
	return d, ok
}

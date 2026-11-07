package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
)

// version is set at build time by the Makefile (-X main.version=...).
var version = "dev"

func init() {
	RegisterDialect(chatCompletionsDialect{})
	RegisterDialect(ollamaNativeDialect{})
	RegisterDialect(responsesDialect{})
}

func main() {
	configPath := os.Getenv("PROVIDERS_CONFIG")
	if configPath == "" {
		configPath = "providers.json"
	}

	cfg, err := LoadConfig(configPath)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	providers, err := BuildProviders(cfg)
	if err != nil {
		log.Fatalf("failed to build providers: %v", err)
	}

	for _, dialect := range dialectRegistry {
		d := dialect // capture loop variable for closure
		http.HandleFunc(d.RoutePath(), chatHandler(d, providers))
	}

	http.HandleFunc(modelsPath, modelsHandler(cfg))

	// Precedence: env var > config file > default. Default host is
	// 127.0.0.1, not all interfaces - see Config's own doc comment for why.
	// Binding to 0.0.0.0 is still possible, it just has to be asked for
	// explicitly (GACHBOTHAR_HOST=0.0.0.0 or "host": "0.0.0.0" in providers.json),
	// never silently the default.
	host := os.Getenv("GACHBOTHAR_HOST")
	if host == "" {
		host = cfg.Host
	}
	if host == "" {
		host = "127.0.0.1"
	}
	port := os.Getenv("PORT")
	if port == "" {
		port = cfg.Port
	}
	if port == "" {
		port = "8099"
	}
	addr := host + ":" + port
	log.Printf("gachbothar %s listening on %s", version, addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

// finishResponse fills in what the dialects read from the request. A model the
// provider named wins: with "auto" it is the one that really answered.
func finishResponse(resp *ChatResponse, req *ChatRequest) {
	resp.Stream = req.Stream
	if resp.Model == "" {
		resp.Model = req.Model
	}
}

// chatHandler keeps provider and dialect selection independent.
func chatHandler(d DialectAdapter, providers map[string]ProviderAdapter) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		req, err := d.ParseRequest(r)
		if err != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
			return
		}
		provider, ok := providers[req.Provider]
		if !ok {
			http.Error(w, fmt.Sprintf("unknown provider %q", req.Provider), http.StatusBadRequest)
			return
		}
		if stream, ok := provider.(StreamingProvider); ok && req.Stream {
			writer := newStreamWriter(w, d.Name(), req.Model)
			resp, err := stream.Stream(r.Context(), req, writer.WriteText)
			if err != nil {
				if !writer.started {
					http.Error(w, fmt.Sprintf("provider error: %v", err), http.StatusBadGateway)
				} else {
					log.Printf("provider stream error for %s: %v", d.Name(), err)
				}
				return
			}
			finishResponse(resp, req)
			if err := writer.Finish(d, resp); err != nil {
				log.Printf("write response for %s: %v", d.Name(), err)
			}
			return
		}
		resp, err := provider.Call(req)
		if err != nil {
			http.Error(w, fmt.Sprintf("provider error: %v", err), http.StatusBadGateway)
			return
		}
		finishResponse(resp, req)
		if err := d.WriteResponse(w, resp); err != nil {
			log.Printf("write response for %s: %v", d.Name(), err)
		}
	}
}

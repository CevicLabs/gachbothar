# gachbothar: design

gachbothar is a Go standard-library HTTP router. Dialect adapters parse inbound
requests into `ChatRequest` and serialize normalized output; provider adapters
translate requests and responses for each backend. Models use `provider/model`.
The router selects a provider by prefix and keeps provider and dialect logic separate.

## Dialects and providers

- Responses `/v1/responses` emits SSE for Codex and maps namespaced tools.
- Chat Completions `/v1/chat/completions` serves JSON or SSE.
- Ollama native `/api/chat` serves JSON or newline-delimited streaming JSON.
- Ollama (native), OpenAI-compatible (`openai`: Ollama `/v1`, llama.cpp, vLLM and
  similar), and OpenCode Zen chat or Messages providers
  support incremental backend responses for streaming requests. Non-streaming
  requests use the buffered provider path.
- Providers normalize tool calls and optional token usage. Reasoning effort is
  forwarded where supported, with the established one-time HTTP 400 retry.
- `GET /v1/models` lists only configured names in Codex's model-list shape.

Streaming crosses an optional `StreamingProvider` boundary: providers emit text
as received and return completed tool calls and usage; dialect writers translate
those pieces into the caller's framing. A stream that fails after output starts
closes without emitting a successful completion. Ollama auto-pull falls back to
a buffered first response. The existing provider HTTP timeout still applies.

## Configuration and security

`PROVIDERS_CONFIG` selects the JSON config (default `./providers.json`).
The `providers` map selects provider type, base URL, key environment variable,
and optional model list. API keys are never stored in the config example.
`GACHBOTHAR_HOST` and `PORT` override config; the default bind is 127.0.0.1:8099.
See `packaging/providers.json.example`.

## Layout

Go source and tests in `src/` (module root is the repo root), docs in `docs/`,
service unit, logrotate and config example in `packaging/`. `make` builds
`build/gachbothar`; `sudo make install` puts it in `/usr/local` (see `Makefile`).

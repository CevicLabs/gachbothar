# gachbothar

*Gach bóthar* is Irish for "every road" (say it roughly *gakh BOH-har*):
one address, every model backend.

A small HTTP router that lets one client talk to many model backends. Clients
speak the protocol they already know; gachbothar translates to whatever the
backend speaks.

- **One static binary.** Go standard library only, no dependencies, no runtime.
- **Three client protocols in:** OpenAI Responses (`/v1/responses`, what Codex
  uses), OpenAI Chat Completions (`/v1/chat/completions`) and Ollama native
  (`/api/chat`). Plus `GET /v1/models`.
- **Backends out:** any OpenAI-compatible server (Ollama `/v1`, llama.cpp,
  vLLM, ...), Ollama native, and OpenCode Zen (chat or Messages protocol).
- **Tool calls** in both directions, including multi-turn tool use.
- **Streaming** end to end (SSE or Ollama's newline-delimited JSON).
- **Token usage normalised**, including cached input tokens, so cost and cache
  hit rates can be compared across backends.
- **Reasoning effort** forwarded where the backend supports it, with a single
  retry without it when a model rejects the field.

Models are addressed as `provider/model`, where `provider` is a name from your
config: `ollama/qwen3:8b`, `llamacpp/gpt-oss-20b`, or
`zen/claude-sonnet-5-5` (Claude models go through OpenCode Zen).

## Security

gachbothar has **no caller authentication**. Anyone who can reach the port can
use your backends and your API keys. It binds to `127.0.0.1` by default; only
bind wider (`GACHBOTHAR_HOST`) on a network you trust, or put it behind a
reverse proxy that authenticates.

API keys are never stored in the config. The config names an environment
variable (`api_key_env`) and the key is read from the environment.

## Quick start

Needs Go 1.25 or newer to build.

```sh
make                      # builds build/gachbothar
cp packaging/providers.json.example providers.json   # edit to taste
./build/gachbothar         # listens on 127.0.0.1:8099
```

Try it:

```sh
curl -s http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ollama/qwen3:8b","messages":[{"role":"user","content":"Say OK."}]}'
```

## Install as a service (Linux, systemd)

```sh
make && sudo make install
```

| What | Where |
|---|---|
| Binary | `/usr/local/bin/gachbothar` (previous one kept as `gachbothar.prev`) |
| Config | `/usr/local/etc/gachbothar/providers.json` (never overwritten by install) |
| Keys | `/usr/local/etc/gachbothar/gachbothar.env`, mode 0600, `NAME=value` per line |
| Log | `/var/log/gachbothar/gachbothar.log`, rotated weekly |
| Unit | `/etc/systemd/system/gachbothar.service` |

The service runs as the user who ran `sudo` (override with
`sudo make install SVCUSER=someone`). After editing the config:
`sudo systemctl restart gachbothar`. `sudo make uninstall` removes the binary and
unit and keeps config and logs.

## Configuration

`providers.json` (path from `PROVIDERS_CONFIG`, default `./providers.json`):

```json
{
  "host": "127.0.0.1",
  "port": "8099",
  "providers": {
    "ollama":   { "type": "openai", "base_url": "http://127.0.0.1:11434/v1", "models": ["qwen3:8b"] },
    "llamacpp": { "type": "openai", "base_url": "http://127.0.0.1:8080/v1" },
    "zen": { "type": "zen", "base_url": "https://opencode.ai", "api_key_env": "ZEN_API_KEY" }
  }
}
```

| Field | Meaning |
|---|---|
| `type` | `openai`, `ollama` (default) or `zen` |
| `base_url` | backend address; for `openai` include the version path (`.../v1`) |
| `api_key_env` | environment variable holding the key; empty for local backends |
| `models` | names listed by `GET /v1/models` (any name is still routed) |
| `protocol` | `zen` only: `chat` (default) or `messages` |
| `auto_pull`, `pull_timeout_seconds` | `ollama` only: pull a missing model on first use |

Environment: `GACHBOTHAR_HOST` and `PORT` override `host` and `port`.

For Ollama, prefer `type: "openai"` with `/v1`: Ollama's native `/api/chat`
does not report cached tokens, its OpenAI-compatible endpoint does.

## Using it from Codex

```toml
model = "ollama/qwen3:8b"
model_provider = "gachbothar"

[model_providers.gachbothar]
name = "gachbothar"
base_url = "http://127.0.0.1:8099/v1"
wire_api = "responses"
requires_openai_auth = false
```

## Development

```sh
make test        # go test ./...
go vet ./...
```

Source is in `src/`, design notes in `docs/design.md`. Adding a backend is one
file plus one entry in the `providerBuilders` registry (`src/provider.go`);
adding a client protocol is one `Dialect` implementation.

## Licence

MIT, see `LICENSE`.

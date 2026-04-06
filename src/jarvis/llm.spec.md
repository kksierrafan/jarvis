## LLM Backend Spec

This specification documents the LLM gateway's support for multiple backend API formats, enabling faster inference on different platforms.

### Overview

The LLM gateway supports two backend API formats:
- **Ollama** (default on non-Apple-Silicon): Uses Ollama's `/api/chat` and `/api/generate` endpoints.
- **OpenAI-compatible** (default on Apple Silicon macOS): Uses the `/v1/chat/completions` endpoint, compatible with MLX, LM Studio, vLLM, and other OpenAI-compatible servers.

### Motivation

Ollama wraps llama.cpp but adds overhead. On Apple Silicon, MLX (Apple's ML framework) is purpose-built for the unified memory architecture and achieves significantly faster inference. MLX exposes an OpenAI-compatible API, so the gateway needs to support both formats.

### Configuration

Three new configuration fields in `config.py`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm_backend` | string | `"auto"` | Backend API format: `"auto"`, `"ollama"`, or `"openai"` |
| `openai_base_url` | string | `"http://127.0.0.1:8080"` | Base URL for OpenAI-compatible server |
| `openai_chat_model` | string | `""` | Model name sent to OpenAI-compatible API (empty = server decides) |

**Backend resolution** (`"auto"` mode):
- Always resolves to `"ollama"` on all platforms. Ollama v0.19+ natively uses MLX on Apple Silicon, so no separate backend is needed for fast inference.
- Users who want a standalone OpenAI-compatible server (LM Studio, vLLM, `mlx_lm.server`) can set `"openai"` explicitly.

### API Format Differences

| Aspect | Ollama | OpenAI-compatible |
|--------|--------|-------------------|
| Chat endpoint | `{base_url}/api/chat` | `{base_url}/v1/chat/completions` |
| Generate endpoint | `{base_url}/api/generate` | `{base_url}/v1/chat/completions` (chat format) |
| Request options | `{"options": {"num_ctx": 4096}}` | Not included (server-side config) |
| Non-streaming response | `{"message": {"content": "..."}}` | `{"choices": [{"message": {"content": "..."}}]}` |
| Tool calls (response) | `message.tool_calls` | `choices[0].message.tool_calls` |
| Streaming format | NDJSON lines | SSE (`data: {...}` lines) |
| Thinking field | `message.thinking` | Not standard (ignored) |
| Qwen3 think toggle | `"think": false` in payload | Not applicable |

### Internal Normalisation

All LLM gateway functions (`call_llm_direct`, `call_llm_streaming`, `chat_with_messages`) accept an `api_format` parameter (`"ollama"` or `"openai"`). Internally, they:

1. Build the correct endpoint URL for the format.
2. Build the correct request payload (Ollama includes `options.num_ctx`; OpenAI does not).
3. **Normalise responses**: OpenAI responses are reshaped into the Ollama internal format (`{"message": {"content": ..., "tool_calls": ...}}`).

This normalisation means **callers do not need to handle format differences**. The reply engine, enrichment, profile selection, and intent judge all receive responses in a uniform shape.

### Affected Components

Each component that calls the LLM gateway receives the resolved `api_format`:

| Component | How it gets api_format |
|-----------|----------------------|
| Reply Engine (`engine.py`) | Resolves from `cfg.llm_backend` at entry; passes to all LLM calls |
| Profile Selection (`profiles.py`) | Receives `api_format` parameter from engine |
| Enrichment (`enrichment.py`) | Receives `api_format` parameter from engine |
| Intent Judge (`intent_judge.py`) | Resolves from config in `create_intent_judge()`; stored on `IntentJudgeConfig` |
| Conversation Memory (`conversation.py`) | Embeddings stay on Ollama (see below) |

### Embeddings

**Embeddings always use Ollama** regardless of `llm_backend`. Rationale:
- Embedding models (e.g., `nomic-embed-text`) are small and fast on Ollama across all platforms.
- MLX and most OpenAI-compatible servers focus on chat/generation, not embeddings.
- This avoids requiring users to run embedding models on their OpenAI-compatible server.

The `ollama_base_url` and `ollama_embed_model` config fields continue to control embedding requests unchanged.

### Resolve Helper

`resolve_llm_backend(backend_setting: str) -> str` in `config.py`:
- Input: `"auto"`, `"ollama"`, or `"openai"`
- Output: `"ollama"` or `"openai"` (never `"auto"`)
- Platform detection matches the existing `_is_apple_silicon()` pattern used by `whisper_backend`.

`get_llm_chat_config(cfg) -> tuple[str, str, str]` in `config.py`:
- Returns `(base_url, chat_model, api_format)` based on the resolved backend.
- Backend `"ollama"` → `(cfg.ollama_base_url, cfg.ollama_chat_model, "ollama")`
- Backend `"openai"` → `(cfg.openai_base_url, cfg.openai_chat_model, "openai")`

### Migration

Config migration v2: no automatic migration needed. Existing configs without the new fields use defaults (`llm_backend: "auto"`, which resolves to the appropriate backend for the platform).

### Platform Support Summary

| Platform | Default backend | Notes |
|----------|----------------|-------|
| macOS ARM64 (Apple Silicon) | Ollama | Ollama v0.19+ natively uses MLX for faster inference |
| macOS Intel | Ollama | Standard Ollama setup |
| Windows | Ollama | Standard Ollama setup |
| Linux | Ollama | Standard Ollama setup; GPU users may benefit from vLLM in future |

All platforms default to Ollama. Users can opt into an OpenAI-compatible backend by setting `llm_backend: "openai"` for use with LM Studio, vLLM, or standalone `mlx_lm.server`.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**tiy-core** is a Rust library providing a unified LLM API and stateful Agent runtime. It abstracts over multiple LLM providers (OpenAI, Anthropic, Google, Ollama, xAI, Groq, OpenRouter, MiniMax, Kimi Coding, ZAI, Zenmux) with a common streaming interface, tool/function calling, and an agentic conversation loop.

## Build & Test Commands

```bash
cargo build                          # Build
cargo test                           # Run all tests
cargo test test_agent_state_new      # Run a single test by name
cargo test --test test_agent         # Run tests in a specific file
cargo test -- --nocapture            # Run tests with output visible
cargo check                          # Check without building
cargo fmt                            # Format code
cargo clippy                         # Lint
cargo run --example basic_usage      # Run example (requires API keys)
```

## Architecture

### Core Layers

1. **Types** (`src/types/`) — Provider-agnostic data model.
   - `Message` is a tagged enum: `User` / `Assistant` / `ToolResult`
   - `ContentBlock` enum: `Text`, `Thinking`, `ToolCall`, `Image`
   - `Model` uses builder pattern; `base_url: Option<String>` (providers supply defaults when None); `api: Option<Api>` selects wire protocol; `compat: Option<OpenAICompletionsCompat>` controls protocol-level compatibility flags
   - `Api` enum identifies wire protocol (e.g., `OpenAICompletions`, `OpenAIResponses`, `AnthropicMessages`, `GoogleGenerativeAi`, `GoogleVertex`, `Ollama`)
   - `Provider` enum identifies the service/company (e.g., `OpenAI`, `Anthropic`, `Google`, `XAI`, `Groq`, `Zenmux`). Many variants exist as type data without a corresponding provider module.

2. **Provider** (`src/provider/`) — Two-tier architecture:

   **Protocol providers** (implement HTTP/SSE stream parsing):
   - `openai_completions` — OpenAI Chat Completions (`/chat/completions`)
   - `openai_responses` — OpenAI Responses API (`/responses`)
   - `anthropic` — Anthropic Messages API (`/messages`)
   - `google` — Google Generative AI + Vertex AI (dual-mode, single module)
   - `ollama` — Ollama (OpenAI-compatible wrapper, defaults to `localhost:11434`)

   **Delegation providers** (inject API key/compat, then call a protocol provider):
   | Module | Delegates To | Env Var |
   |---|---|---|
   | `xai` | `openai_completions` | `XAI_API_KEY` |
   | `groq` | `openai_completions` | `GROQ_API_KEY` |
   | `openrouter` | `openai_completions` | `OPENROUTER_API_KEY` |
   | `zai` | `openai_completions` | `ZAI_API_KEY` |
   | `minimax` | `anthropic` | `MINIMAX_API_KEY` |
   | `kimi_coding` | `anthropic` | `KIMI_API_KEY` |
   | `zenmux` | adaptive (see below) | `ZENMUX_API_KEY` |

   Delegation providers expose `default_compat()` static methods (for OpenAI-compatible ones) that return provider-specific `OpenAICompletionsCompat` flags. These are injected onto the model when `model.compat.is_none()`.

3. **Stream** (`src/stream/`) — `EventStream<T, R>` is a generic async stream backed by `parking_lot::Mutex<VecDeque>`. Implements `futures::Stream` with a separate `result()` future for the final message. `AssistantMessageEventStream` emits `AssistantMessageEvent` variants: `Start`, `TextDelta`, `ThinkingDelta`, `ToolCallDelta`, `Done`, `Error`, etc.

4. **Agent** (`src/agent/`) — Stateful conversation manager. `Agent` wraps `AgentState` (thread-safe with `parking_lot::RwLock`/`AtomicBool`) and runs an autonomous loop: stream LLM → check tool calls → execute via `ToolExecutor` callback → loop. Supports steering (interrupt), follow-up message queues, event subscription (observer pattern), abort, and configurable max turns (default 25). Tool execution can be parallel (default) or sequential.

### Supporting Modules

- **Transform** (`src/transform/`) — Cross-provider message transformation: thinking block conversion (thinking→text with `[Reasoning]` wrapper when switching providers), tool call ID normalization, orphan tool call resolution (inserts synthetic error results).
- **Thinking** (`src/thinking/`) — `ThinkingLevel` enum (Off/Minimal/Low/Medium/High/XHigh) and provider-specific thinking option structs.
- **Validation** (`src/validation/`) — Tool parameter JSON Schema validation via `jsonschema` crate.
- **Models** (`src/models/`) — `ModelRegistry` with predefined models and a global static instance. Use `get_model("openai", "gpt-4o")` to look up. Predefined models are a small stub set (a few models per provider for OpenAI, Anthropic, Google).

### Key Design Patterns

- **Provider registry**: `ProviderRegistry` is keyed by `Provider::as_str()` string (e.g., `"openai"`, `"xai"`, `"zenmux"`). Global static accessed via `register_provider()` / `get_provider()` / `get_registered_providers()`.
- **Base URL resolution**: 3-level fallback: `StreamOptions.base_url` > `model.base_url` > provider's `DEFAULT_BASE_URL` constant.
- **API key resolution**: `StreamOptions.api_key` → provider's `default_api_key` → environment variable (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
- **SSE parsing**: Provider modules parse Server-Sent Events manually from HTTP byte streams, buffering incomplete lines. The OpenAI Responses provider extracts event type from the JSON `type` field (not SSE `event:` line) to handle proxies that strip SSE event headers.
- **Thread safety**: All mutable state uses `parking_lot` locks (not `std::sync`) for non-poisoning behavior.
- **Builders**: `Model::builder()`, `AssistantMessage::builder()`, `Tool::builder()` with required-field validation in `build()`.

### Zenmux Adaptive Routing

Zenmux is a unique multi-protocol proxy provider. When `base_url` is None, empty, or starts with `https://zenmux.ai`, it routes based on model ID:
- `"google"` or `"gemini"` in ID → `GoogleProvider` with `Api::GoogleVertex`, base `https://zenmux.ai/api/vertex-ai`
- `"openai"` or `"gpt"` in ID → `OpenAIResponsesProvider` with `Api::OpenAIResponses`, base `https://zenmux.ai/api/v1`
- anything else → `AnthropicProvider` with `Api::AnthropicMessages`, base `https://zenmux.ai/api/anthropic/v1`

When a custom (non-zenmux) base URL is provided, it forces `OpenAICompletions` protocol.

### Google Dual-Mode URL Format

The `google` module handles two URL formats based on `model.api`:
- `GoogleGenerativeAi` (default): `{base}/models/{id}:streamGenerateContent?alt=sse` with `x-goog-api-key` header
- `GoogleVertex`: `{base}/v1/publishers/google/models/{id}:streamGenerateContent?alt=sse` with `Authorization: Bearer` header

### Type Relationships

```
Agent → AgentState (Arc-wrapped, thread-safe)
      → AgentConfig (model, thinking_level, tool_execution mode)
      → ToolExecutor (async callback: name, id, args → AgentToolResult)

LLMProvider.stream() → AssistantMessageEventStream

ProviderRegistry: HashMap<String, ArcProvider> keyed by Provider::as_str()
ModelRegistry: HashMap<provider, HashMap<model_id, Model>>
```

## Adding a New Provider

**Protocol provider** (new wire format):
1. Create `src/provider/<name>.rs` implementing `LLMProvider` trait (3 methods: `provider_type`, `stream`, `stream_simple`)
2. Add `pub mod <name>;` to `src/provider/mod.rs`
3. Add wire-format request/response structs (private to the module)
4. Implement SSE stream parsing in a `run_stream` async function spawned via `tokio::spawn`
5. Push events to `AssistantMessageEventStream` (Start → deltas → Done/Error)
6. Add integration test in `tests/test_provider_<name>.rs` using `wiremock` for HTTP mocking

**Delegation provider** (wraps existing protocol):
1. Create `src/provider/<name>.rs` with `new()`, `with_api_key()`, and key resolution from env var
2. Implement `LLMProvider::stream()` by injecting `compat` settings and delegating to the protocol provider's `stream()`
3. Optionally expose `default_compat()` static method for OpenAI-compatible providers
4. Add tests in `tests/test_delegation_providers.rs`

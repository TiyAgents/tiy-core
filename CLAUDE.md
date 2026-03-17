# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**tiy-core** is a Rust library providing a unified LLM API and stateful Agent runtime. It abstracts over multiple LLM providers (OpenAI, Anthropic, Google, Ollama, xAI, Groq, OpenRouter, MiniMax, Kimi Coding, Zenmux, etc.) with a common streaming interface, tool/function calling, and an agentic conversation loop.

## Build & Test Commands

```bash
# Build
cargo build

# Run all tests
cargo test

# Run a single test by name
cargo test test_agent_state_new

# Run tests in a specific file
cargo test --test test_agent

# Run tests with output visible
cargo test -- --nocapture

# Run examples (requires API keys in env)
cargo run --example basic_usage
cargo run --example agent_example

# Check without building
cargo check

# Format code
cargo fmt

# Lint
cargo clippy
```

## Architecture

### Core Layers

The library has four main layers, from low-level to high-level:

1. **Types** (`src/types/`) — Provider-agnostic data model. `Message` is a tagged enum (User/Assistant/ToolResult). `AssistantMessage` contains `Vec<ContentBlock>` where `ContentBlock` is an enum of Text, Thinking, ToolCall, and Image. `UserContent` is either plain text or a vec of `ContentBlock`s. Both `Model` and `AssistantMessage` use the builder pattern.

2. **Provider** (`src/provider/`) — Each LLM API has its own module implementing the `LLMProvider` trait (two methods: `stream` and `stream_simple`). Providers are registered in a `ProviderRegistry` (keyed by `Api` enum variant). There is a global static registry accessed via `register_provider()` / `get_provider()`. Each provider module is self-contained with its own wire-format request/response types and message conversion logic.

3. **Stream** (`src/stream/`) — `EventStream<T, R>` is a generic async stream backed by a lock-free queue (`parking_lot::Mutex<VecDeque>`). It implements `futures::Stream` and supports a separate `result()` future for the final message. `AssistantMessageEventStream` is the concrete type alias emitting `AssistantMessageEvent` variants (Start, TextDelta, ThinkingDelta, ToolCallDelta, Done, Error, etc.).

4. **Agent** (`src/agent/`) — Stateful conversation manager. `Agent` wraps `AgentState` (thread-safe with `parking_lot::RwLock`/`AtomicBool`) and runs an autonomous loop: stream LLM → check tool calls → execute via `ToolExecutor` callback → loop. Supports steering (interrupt) and follow-up message queues, event subscription (observer pattern), abort, and configurable max turns (default 25). Tool execution can be parallel (default) or sequential.

### Supporting Modules

- **Transform** (`src/transform/`) — Cross-provider message transformation. Handles thinking block conversion between providers (thinking→text with `[Reasoning]` wrapper when switching providers), tool call ID normalization, and orphan tool call resolution (inserts synthetic error results).

- **Thinking** (`src/thinking/`) — `ThinkingLevel` enum (Off/Minimal/Low/Medium/High/XHigh) and provider-specific thinking option structs (`OpenAIThinkingOptions`, `AnthropicThinkingOptions`, `GoogleThinkingOptions`).

- **Validation** (`src/validation/`) — Tool parameter JSON Schema validation.

- **Models** (`src/models/`) — `ModelRegistry` with predefined models (OpenAI, Anthropic, Google) and a global static instance. Use `get_model("openai", "gpt-4o")` to look up predefined models.

### Key Design Patterns

- **Thread safety**: All mutable state uses `parking_lot` locks (not `std::sync`) for non-poisoning behavior. `AgentState` fields are individually locked for fine-grained concurrency.
- **Provider resolution**: Agent first checks an explicitly set provider, then falls back to the global `ProviderRegistry` by `Api` type.
- **API key resolution**: Priority order is `StreamOptions.api_key` → provider's `default_api_key` → environment variable (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
- **SSE parsing**: Provider modules parse Server-Sent Events manually from the HTTP byte stream, buffering incomplete lines.
- **Builders**: `Model::builder()`, `AssistantMessage::builder()`, and `Tool::builder()` use the builder pattern with required-field validation in `build()`.

### Type Relationships

```
Agent → AgentState (Arc-wrapped, thread-safe)
      → AgentConfig (model, thinking_level, tool_execution mode)
      → ToolExecutor (async callback: name, id, args → AgentToolResult)

AgentMessage ↔ Message (bidirectional conversion via From)
AgentTool → Tool (via as_tool())

LLMProvider.stream() → AssistantMessageEventStream (EventStream<AssistantMessageEvent, AssistantMessage>)

ProviderRegistry: HashMap<String, ArcProvider> (global static via once_cell::Lazy)
ModelRegistry: HashMap<provider, HashMap<model_id, Model>> (global static)
```

## Adding a New Provider

1. Create `src/provider/<name>.rs` implementing `LLMProvider` trait
2. Add `pub mod <name>;` to `src/provider/mod.rs`
3. Add wire-format request/response structs (private to the module)
4. Implement SSE stream parsing in a `run_stream` async function spawned via `tokio::spawn`
5. Push events to the `AssistantMessageEventStream` (Start → deltas → Done/Error)
6. Add integration test in `tests/test_provider_<name>.rs`

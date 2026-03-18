# Protocol Layer

[English](./README.md) | [中文](#协议层)

Wire-format implementations that handle the actual HTTP/SSE communication with LLM APIs. Most users should use the [Provider](../provider/README.md) layer instead of interacting with protocols directly.

## Overview

The protocol layer sits between the high-level [Provider](../provider/) facades and the raw HTTP wire. Each protocol module knows how to:

1. Build the provider-specific JSON request body
2. Send the HTTP request with correct headers and authentication
3. Parse the SSE (Server-Sent Events) stream
4. Emit normalized `AssistantMessageEvent` variants into an `AssistantMessageEventStream`

```
Provider (facade) ──delegates──▶ Protocol (wire-format) ──HTTP/SSE──▶ LLM API
```

## LLMProtocol Trait

All protocols implement the `LLMProtocol` trait defined in `traits.rs`:

```rust
#[async_trait]
pub trait LLMProtocol: Send + Sync {
    /// Identifies which provider this implementation handles.
    fn provider_type(&self) -> Provider;

    /// Stream completion with full options.
    fn stream(&self, model: &Model, context: &Context, options: StreamOptions)
        -> AssistantMessageEventStream;

    /// Stream completion with simplified options.
    fn stream_simple(&self, model: &Model, context: &Context, options: SimpleStreamOptions)
        -> AssistantMessageEventStream;
}
```

## Base Protocols

Four base protocol implementations handle the distinct wire formats:

| Module | Struct | API Endpoint | SSE Event Flow |
|---|---|---|---|
| `openai_completions` | `OpenAICompletionsProtocol` | `POST /chat/completions` | `data: {choices[0].delta}` chunks, `[DONE]` sentinel |
| `openai_responses` | `OpenAIResponsesProtocol` | `POST /responses` | Typed events: `response.output_item.added` → `response.output_text.delta` / `response.function_call_arguments.delta` → `response.output_item.done` → `response.completed` |
| `anthropic` | `AnthropicProtocol` | `POST /messages` | `message_start` → `content_block_start` → `content_block_delta` → `content_block_stop` → `message_delta` → `message_stop` |
| `google` | `GoogleProtocol` | `POST /models/{id}:streamGenerateContent?alt=sse` | JSON chunks with `candidates[].content.parts[]` |

### Default Base URLs

| Protocol | Default Base URL | Env Var |
|---|---|---|
| OpenAI Completions | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| OpenAI Responses | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| Anthropic | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` |
| Google (GenAI) | `https://generativelanguage.googleapis.com/v1beta` | `GOOGLE_API_KEY` / `GEMINI_API_KEY` |
| Google (Vertex AI) | `https://us-central1-aiplatform.googleapis.com` | `GOOGLE_API_KEY` |

### Google Dual-Mode

The `google` module handles two URL formats based on `model.api`:

- **`GoogleGenerativeAi`** (default): `{base}/models/{id}:streamGenerateContent?alt=sse` with `x-goog-api-key` header
- **`GoogleVertex`**: `{base}/v1/publishers/google/models/{id}:streamGenerateContent?alt=sse` with `Authorization: Bearer` header

## Shared Infrastructure (`common.rs`)

Common utilities shared across all protocol implementations:

| Function | Purpose |
|---|---|
| `resolve_base_url` | 3-level fallback: `options.base_url` > `model.base_url` > provider default |
| `apply_on_payload` | Serialize request body, optionally passing through `on_payload` hook for mutation |
| `validate_url_or_error` | Validate base URL against `SecurityConfig.url` policy (SSRF protection) |
| `apply_custom_headers` | Inject custom headers, skipping protected ones per `HeaderPolicy` |
| `handle_error_response` | Read error body (bounded), log, emit `Error` event |
| `check_sse_buffer_overflow` | Abort stream if SSE line buffer exceeds configured limit |
| `debug_preview` | Truncate body string for debug logging |

## Delegation Macros (`delegation.rs`)

Two macros reduce boilerplate for creating delegation providers:

### `define_openai_delegation_provider!`

Generates a provider that delegates to `OpenAICompletionsProtocol`. Three variants:

```rust
// Variant 1: No compat injection (e.g., OpenRouter)
define_openai_delegation_provider! {
    name: OpenRouterProvider,
    doc: "OpenRouter provider.",
    provider_type: Provider::OpenRouter,
    env_var: "OPENROUTER_API_KEY",
}

// Variant 2: Static compat (e.g., xAI, ZAI)
define_openai_delegation_provider! {
    name: XAIProvider,
    doc: "xAI provider.",
    provider_type: Provider::XAI,
    env_var: "XAI_API_KEY",
    default_compat: || OpenAICompletionsCompat { ... },
}

// Variant 3: Model-aware compat (e.g., Groq)
define_openai_delegation_provider! {
    name: GroqProvider,
    doc: "Groq provider.",
    provider_type: Provider::Groq,
    env_var: "GROQ_API_KEY",
    model_aware_compat: |model_id: &str| OpenAICompletionsCompat { ... },
}
```

### `define_anthropic_delegation_provider!`

Generates a provider that delegates to `AnthropicProtocol`:

```rust
define_anthropic_delegation_provider! {
    name: KimiCodingProvider,
    doc: "Kimi Coding provider.",
    provider_type: Provider::KimiCoding,
    env_var: "KIMI_API_KEY",
}
```

## Provider Registry (`registry.rs`)

A global thread-safe registry maps `Provider::as_str()` keys to `ArcProtocol` instances:

```rust
use std::sync::Arc;
use tiy_core::provider::{register_provider, get_provider};
use tiy_core::provider::openai::OpenAIProvider;
use tiy_core::types::Provider;

// Register
register_provider(Arc::new(OpenAIProvider::new()));

// Lookup
let provider = get_provider(&Provider::OpenAI).unwrap();
```

Registry API:

| Function | Description |
|---|---|
| `register_provider(ArcProtocol)` | Register a provider globally |
| `get_provider(&Provider)` | Look up by `Provider` enum |
| `get_registered_providers()` | List all registered provider name strings |
| `clear_providers()` | Remove all registered providers |

## File Structure

```
protocol/
├── mod.rs                  # Module declarations, re-exports LLMProtocol + registry
├── traits.rs               # LLMProtocol trait, BoxedProtocol, ArcProtocol
├── registry.rs             # ProtocolRegistry + global static + convenience functions
├── common.rs               # Shared infrastructure (URL, headers, errors, SSE)
├── delegation.rs           # Macros for generating delegation providers
├── openai_completions.rs   # OpenAI Chat Completions wire format
├── openai_responses.rs     # OpenAI Responses API wire format
├── anthropic.rs            # Anthropic Messages wire format
└── google.rs               # Google GenAI + Vertex AI wire format
```

## Adding a New Protocol

> Only needed when a completely new HTTP/SSE wire format is required.

1. Create `src/protocol/<name>.rs` implementing `LLMProtocol` (3 methods)
2. Add `pub mod <name>;` to `mod.rs`
3. Define private request/response structs for the wire format
4. Implement SSE stream parsing in an async `run_stream` function spawned via `tokio::spawn`
5. Push events to `AssistantMessageEventStream`: `Start` → deltas → `Done`/`Error`
6. Add tests using `wiremock` for HTTP mocking

For delegation providers (reusing an existing protocol), see [Provider README](../provider/README.md).

---

<a name="协议层"></a>
# 协议层

[English](#protocol-layer) | [中文](./README.md)

处理与 LLM API 实际 HTTP/SSE 通信的线路格式实现。大多数用户应使用 [Provider](../provider/README.md) 层，而非直接与协议交互。

## 概述

协议层位于上层 [Provider](../provider/) 门面与原始 HTTP 线路之间。每个协议模块负责：

1. 构建提供商特定的 JSON 请求体
2. 发送带有正确请求头和认证信息的 HTTP 请求
3. 解析 SSE（Server-Sent Events）流
4. 将事件规范化为 `AssistantMessageEvent` 并推送到 `AssistantMessageEventStream`

```
Provider（门面）──委托──▶ Protocol（线路格式）──HTTP/SSE──▶ LLM API
```

## LLMProtocol Trait

所有协议都实现定义在 `traits.rs` 中的 `LLMProtocol` trait：

```rust
#[async_trait]
pub trait LLMProtocol: Send + Sync {
    /// 标识此实现对应的提供商。
    fn provider_type(&self) -> Provider;

    /// 使用完整选项进行流式补全。
    fn stream(&self, model: &Model, context: &Context, options: StreamOptions)
        -> AssistantMessageEventStream;

    /// 使用简化选项进行流式补全。
    fn stream_simple(&self, model: &Model, context: &Context, options: SimpleStreamOptions)
        -> AssistantMessageEventStream;
}
```

## 基础协议

四个基础协议实现处理不同的线路格式：

| 模块 | 结构体 | API 端点 | SSE 事件流 |
|---|---|---|---|
| `openai_completions` | `OpenAICompletionsProtocol` | `POST /chat/completions` | `data: {choices[0].delta}` 块，`[DONE]` 终止标记 |
| `openai_responses` | `OpenAIResponsesProtocol` | `POST /responses` | 类型化事件：`response.output_item.added` → `response.output_text.delta` / `response.function_call_arguments.delta` → `response.output_item.done` → `response.completed` |
| `anthropic` | `AnthropicProtocol` | `POST /messages` | `message_start` → `content_block_start` → `content_block_delta` → `content_block_stop` → `message_delta` → `message_stop` |
| `google` | `GoogleProtocol` | `POST /models/{id}:streamGenerateContent?alt=sse` | JSON 块，包含 `candidates[].content.parts[]` |

### 默认 Base URL

| 协议 | 默认 Base URL | 环境变量 |
|---|---|---|
| OpenAI Completions | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| OpenAI Responses | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| Anthropic | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` |
| Google (GenAI) | `https://generativelanguage.googleapis.com/v1beta` | `GOOGLE_API_KEY` / `GEMINI_API_KEY` |
| Google (Vertex AI) | `https://us-central1-aiplatform.googleapis.com` | `GOOGLE_API_KEY` |

### Google 双模式

`google` 模块根据 `model.api` 处理两种 URL 格式：

- **`GoogleGenerativeAi`**（默认）：`{base}/models/{id}:streamGenerateContent?alt=sse`，使用 `x-goog-api-key` 请求头
- **`GoogleVertex`**：`{base}/v1/publishers/google/models/{id}:streamGenerateContent?alt=sse`，使用 `Authorization: Bearer` 请求头

## 共享基础设施（`common.rs`）

所有协议实现共享的通用工具函数：

| 函数 | 用途 |
|---|---|
| `resolve_base_url` | 三级回退：`options.base_url` > `model.base_url` > 提供商默认值 |
| `apply_on_payload` | 序列化请求体，可选通过 `on_payload` 钩子进行修改 |
| `validate_url_or_error` | 根据 `SecurityConfig.url` 策略验证 URL（SSRF 防护） |
| `apply_custom_headers` | 注入自定义请求头，跳过受保护的头（按 `HeaderPolicy`） |
| `handle_error_response` | 读取错误响应体（有上限），记录日志，发出 `Error` 事件 |
| `check_sse_buffer_overflow` | 当 SSE 行缓冲区超出限制时中止流 |
| `debug_preview` | 截断请求体字符串用于调试日志 |

## 委托宏（`delegation.rs`）

两个宏减少了创建委托提供商的样板代码：

### `define_openai_delegation_provider!`

生成委托到 `OpenAICompletionsProtocol` 的提供商。三个变体：

```rust
// 变体 1：无兼容性注入（如 OpenRouter）
define_openai_delegation_provider! {
    name: OpenRouterProvider,
    doc: "OpenRouter provider.",
    provider_type: Provider::OpenRouter,
    env_var: "OPENROUTER_API_KEY",
}

// 变体 2：静态兼容性（如 xAI、ZAI）
define_openai_delegation_provider! {
    name: XAIProvider,
    doc: "xAI provider.",
    provider_type: Provider::XAI,
    env_var: "XAI_API_KEY",
    default_compat: || OpenAICompletionsCompat { ... },
}

// 变体 3：模型感知兼容性（如 Groq）
define_openai_delegation_provider! {
    name: GroqProvider,
    doc: "Groq provider.",
    provider_type: Provider::Groq,
    env_var: "GROQ_API_KEY",
    model_aware_compat: |model_id: &str| OpenAICompletionsCompat { ... },
}
```

### `define_anthropic_delegation_provider!`

生成委托到 `AnthropicProtocol` 的提供商：

```rust
define_anthropic_delegation_provider! {
    name: KimiCodingProvider,
    doc: "Kimi Coding provider.",
    provider_type: Provider::KimiCoding,
    env_var: "KIMI_API_KEY",
}
```

## 提供商注册表（`registry.rs`）

全局线程安全注册表，以 `Provider::as_str()` 为键映射到 `ArcProtocol` 实例：

```rust
use std::sync::Arc;
use tiy_core::provider::{register_provider, get_provider};
use tiy_core::provider::openai::OpenAIProvider;
use tiy_core::types::Provider;

// 注册
register_provider(Arc::new(OpenAIProvider::new()));

// 查找
let provider = get_provider(&Provider::OpenAI).unwrap();
```

注册表 API：

| 函数 | 描述 |
|---|---|
| `register_provider(ArcProtocol)` | 全局注册提供商 |
| `get_provider(&Provider)` | 通过 `Provider` 枚举查找 |
| `get_registered_providers()` | 列出所有已注册的提供商名称 |
| `clear_providers()` | 移除所有已注册的提供商 |

## 文件结构

```
protocol/
├── mod.rs                  # 模块声明，re-export LLMProtocol + registry
├── traits.rs               # LLMProtocol trait、BoxedProtocol、ArcProtocol
├── registry.rs             # ProtocolRegistry + 全局静态实例 + 便捷函数
├── common.rs               # 共享基础设施（URL、请求头、错误处理、SSE）
├── delegation.rs           # 生成委托提供商的宏
├── openai_completions.rs   # OpenAI Chat Completions 线路格式
├── openai_responses.rs     # OpenAI Responses API 线路格式
├── anthropic.rs            # Anthropic Messages 线路格式
└── google.rs               # Google GenAI + Vertex AI 线路格式
```

## 添加新协议

> 仅在需要全新的 HTTP/SSE 线路格式时才需要。

1. 创建 `src/protocol/<name>.rs`，实现 `LLMProtocol`（3 个方法）
2. 在 `mod.rs` 中添加 `pub mod <name>;`
3. 定义线路格式的私有请求/响应结构体
4. 在通过 `tokio::spawn` 启动的异步 `run_stream` 函数中实现 SSE 流解析
5. 向 `AssistantMessageEventStream` 推送事件：`Start` → 增量事件 → `Done`/`Error`
6. 使用 `wiremock` 进行 HTTP mock 测试

对于委托提供商（复用现有协议），请参见 [Provider README](../provider/README.md)。

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

**`finish_reason` tolerance:** Many OpenAI-compatible providers omit `finish_reason` when tool calls are present. The protocol module handles this in two stages: first, the incomplete-stream detector only flags a missing `finish_reason` as incomplete when the `[DONE]` sentinel is also absent; second, if `[DONE]` was received but `finish_reason` is missing and the output contains tool calls, `StopReason::ToolUse` is inferred automatically so the agent loop continues executing tools.

| `openai_responses` | `OpenAIResponsesProtocol` | `POST /responses` | Typed events: `response.output_item.added` → `response.output_text.delta` / `response.function_call_arguments.delta` → `response.output_item.done` → `response.completed` |
| `anthropic` | `AnthropicProtocol` | `POST /messages` | `message_start` → `content_block_start` → `content_block_delta` → `content_block_stop` → `message_delta` → `message_stop` |

**Adaptive thinking `display` field:** For models that require explicit opt-in (currently Opus 4.7), the `thinking` parameter includes an optional `display` field (`"summarized"` or `"omitted"`) controlling how thinking content appears in the response. This is set via `StreamOptions.thinking_display` (`ThinkingDisplay` enum, defaults to `Summarized`). For older models, the field is omitted from the wire format.

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
| `send_request_with_retry` | Retry transient HTTP failures with exponential backoff and `Retry-After` support |
| `handle_error_response` | Read error body (bounded), log, emit `Error` event |
| `emit_terminal_error` | Convert transport/protocol failures into a terminal `Error` event and close the stream |
| `check_sse_buffer_overflow` | Abort stream if SSE line buffer exceeds configured limit |
| `emit_incomplete_stream_error` | Emit a structured `[incomplete_stream]` terminal error with provider and detail |
| `parse_incomplete_stream_error` | Parse an incomplete-stream error back into `(provider, detail)` for upper-layer retry |
| `debug_preview` | Truncate body string for debug logging |

### Incomplete Stream Detection

When an SSE stream terminates abnormally (connection drop, missing terminal event, unclosed content blocks), each protocol module calls a dedicated detection function after the stream loop exits:

| Module | Function | Key Signals |
|--------|----------|-------------|
| `anthropic` | `incomplete_anthropic_stream_detail` | Missing `message_delta`/`message_stop`; unclosed content blocks; unfinished tool-arg JSON; trailing partial SSE frame |
| `openai_completions` | `incomplete_openai_completions_stream_detail` | Missing `finish_reason` (only if `[DONE]` also absent); missing `[DONE]`; unfinished tool-arg JSON; trailing frame |
| `openai_responses` | `incomplete_openai_responses_stream_detail` | Missing `response.completed`/`response.done`; unfinished output items; unfinished tool-arg JSON; trailing frame |
| `google` | `incomplete_google_stream_detail` | Missing candidate `finish_reason`; trailing partial frame |

If a function returns `Some(detail)`, `emit_incomplete_stream_error()` pushes a terminal `Error` event with a `[incomplete_stream]<provider>: <detail>` message. The agent layer can then call `parse_incomplete_stream_error()` on the error message to extract the provider and detail for retry or recovery logic.

### Environment Variables

Protocol-layer behaviour can be tuned through the following environment variables:

| Variable | Values | Default | Used By | Purpose |
|----------|--------|---------|---------|---------|
| `TIY_CACHE_RETENTION` | `long`, `none` | (unset → `Short`) | `anthropic`, `openai_responses` | Controls prompt-caching retention policy |

### Retry Semantics

Protocol implementations perform transparent retries only before any semantic assistant output has been emitted. This covers transient request/setup failures such as:

- HTTP `408`, `429`, `500`, `502`, `503`, `504`
- `reqwest::Error` where `is_timeout()` or `is_connect()` is `true`
- pre-semantic streamed-body interruptions reported as transport/body errors

When the provider sends `Retry-After`, the retry delay honours that header. `max_retry_delay_ms` caps the delay, while `Some(0)` disables the cap.

Once a stream has emitted semantic events such as text deltas, thinking deltas, or tool-call deltas, the protocol layer no longer retries transparently. Any later transport failure is emitted as a terminal `Error` event so upper layers can decide how to recover without risking duplicate output or repeated tool side effects.

## File Structure

```
protocol/
├── mod.rs                  # Module declarations, re-exports LLMProtocol + backward-compat registry re-exports
├── traits.rs               # LLMProtocol trait, BoxedProtocol, ArcProtocol
├── common.rs               # Shared infrastructure (URL, headers, errors, SSE)
├── openai_completions.rs   # OpenAI Chat Completions wire format
├── openai_responses.rs     # OpenAI Responses API wire format
├── anthropic.rs            # Anthropic Messages wire format
└── google.rs               # Google GenAI + Vertex AI wire format
```

> **Note:** The provider registry (`registry.rs`) and delegation macros (`delegation.rs`) now live in the [Provider](../provider/README.md) layer, alongside the facades that use them. `protocol/mod.rs` re-exports registry symbols for backward compatibility.

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

**`finish_reason` 容错：** 许多 OpenAI 兼容提供商在存在工具调用时省略 `finish_reason`。协议模块分两阶段处理：首先，不完整流检测器仅在 `[DONE]` 终止标记也缺失时才将缺失的 `finish_reason` 标记为不完整；其次，若已收到 `[DONE]` 但 `finish_reason` 缺失且输出包含工具调用，则自动推断 `StopReason::ToolUse`，使 Agent 循环继续执行工具。

| `openai_responses` | `OpenAIResponsesProtocol` | `POST /responses` | 类型化事件：`response.output_item.added` → `response.output_text.delta` / `response.function_call_arguments.delta` → `response.output_item.done` → `response.completed` |
| `anthropic` | `AnthropicProtocol` | `POST /messages` | `message_start` → `content_block_start` → `content_block_delta` → `content_block_stop` → `message_delta` → `message_stop` |

**自适应思维 `display` 字段：** 对于需要显式选择的模型（目前为 Opus 4.7），`thinking` 参数包含可选的 `display` 字段（`"summarized"` 或 `"omitted"`），用于控制思维内容在响应中的呈现方式。通过 `StreamOptions.thinking_display`（`ThinkingDisplay` 枚举，默认为 `Summarized`）设置。旧模型在线路格式中省略此字段。

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
| `send_request_with_retry` | 对瞬时 HTTP 失败执行指数退避重试，并支持 `Retry-After` |
| `handle_error_response` | 读取错误响应体（有上限），记录日志，发出 `Error` 事件 |
| `emit_terminal_error` | 将传输/协议失败转换为终止 `Error` 事件并关闭流 |
| `check_sse_buffer_overflow` | 当 SSE 行缓冲区超出限制时中止流 |
| `emit_incomplete_stream_error` | 发出结构化的 `[incomplete_stream]` 终止错误，包含提供商和详细信息 |
| `parse_incomplete_stream_error` | 将不完整流错误反向解析为 `(提供商, 详细信息)`，供上层重试使用 |
| `debug_preview` | 截断请求体字符串用于调试日志 |

### 不完整流检测

当 SSE 流异常终止（连接断开、缺少终止事件、未闭合的内容块等）时，各协议模块在流循环退出后调用专用的检测函数：

| 模块 | 函数 | 关键信号 |
|------|------|----------|
| `anthropic` | `incomplete_anthropic_stream_detail` | 缺少 `message_delta`/`message_stop`；未闭合的内容块；未完成的工具参数 JSON；尾部部分 SSE 帧 |
| `openai_completions` | `incomplete_openai_completions_stream_detail` | 缺少 `finish_reason`（仅在 `[DONE]` 也缺失时）；缺少 `[DONE]`；未完成的工具参数 JSON；尾部帧 |
| `openai_responses` | `incomplete_openai_responses_stream_detail` | 缺少 `response.completed`/`response.done`；未完成的输出项；未完成的工具参数 JSON；尾部帧 |
| `google` | `incomplete_google_stream_detail` | 缺少候选 `finish_reason`；尾部部分帧 |

若函数返回 `Some(detail)`，`emit_incomplete_stream_error()` 会推送一个包含 `[incomplete_stream]<提供商>: <详细信息>` 消息的终止 `Error` 事件。Agent 层可调用 `parse_incomplete_stream_error()` 从错误消息中提取提供商和详细信息，用于重试或恢复逻辑。

### 环境变量

协议层行为可通过以下环境变量调整：

| 变量 | 取值 | 默认值 | 使用模块 | 用途 |
|------|------|--------|----------|------|
| `TIY_CACHE_RETENTION` | `long`、`none` | （未设置 → `Short`） | `anthropic`、`openai_responses` | 控制提示缓存保留策略 |

### 重试语义

协议层只会在“尚未发出任何有语义的 assistant 输出”之前进行透明重试。覆盖的典型场景包括：

- HTTP `408`、`429`、`500`、`502`、`503`、`504`
- `reqwest::Error` 中 `is_timeout()` 或 `is_connect()` 为 `true`
- 在首个语义事件之前发生的 streamed-body/transport 抖动

如果 provider 返回了 `Retry-After`，协议层会优先遵守它。`max_retry_delay_ms` 用于限制退避上限，`Some(0)` 表示关闭这个上限。

一旦流已经发出了文本增量、thinking 增量或 tool call 增量等语义事件，协议层就不再做透明重试。此后若发生传输错误，会直接发出终止 `Error` 事件，让上层显式决定如何恢复，从而避免重复输出或重复执行工具副作用。

## 文件结构

```
protocol/
├── mod.rs                  # 模块声明，re-export LLMProtocol + 向后兼容的 registry re-export
├── traits.rs               # LLMProtocol trait、BoxedProtocol、ArcProtocol
├── common.rs               # 共享基础设施（URL、请求头、错误处理、SSE）
├── openai_completions.rs   # OpenAI Chat Completions 线路格式
├── openai_responses.rs     # OpenAI Responses API 线路格式
├── anthropic.rs            # Anthropic Messages 线路格式
└── google.rs               # Google GenAI + Vertex AI 线路格式
```

> **注意：** 提供商注册表（`registry.rs`）和委托宏（`delegation.rs`）现已迁移至 [Provider](../provider/README.md) 层，与使用它们的门面文件放在一起。`protocol/mod.rs` 为向后兼容继续 re-export 注册表符号。

## 添加新协议

> 仅在需要全新的 HTTP/SSE 线路格式时才需要。

1. 创建 `src/protocol/<name>.rs`，实现 `LLMProtocol`（3 个方法）
2. 在 `mod.rs` 中添加 `pub mod <name>;`
3. 定义线路格式的私有请求/响应结构体
4. 在通过 `tokio::spawn` 启动的异步 `run_stream` 函数中实现 SSE 流解析
5. 向 `AssistantMessageEventStream` 推送事件：`Start` → 增量事件 → `Done`/`Error`
6. 使用 `wiremock` 进行 HTTP mock 测试

对于委托提供商（复用现有协议），请参见 [Provider README](../provider/README.md)。

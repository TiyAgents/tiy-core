//! Tests for Anthropic Messages provider using wiremock for HTTP mocking.

use futures::StreamExt;
use serde_json::json;
use tiy_core::protocol::anthropic::AnthropicProtocol;
use tiy_core::protocol::LLMProtocol;
use tiy_core::types::*;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ============================================================================
// Helper functions
// ============================================================================

fn make_model(base_url: &str) -> Model {
    Model::builder()
        .id("claude-3-5-sonnet")
        .name("Claude 3.5 Sonnet")
        .api(Api::AnthropicMessages)
        .provider(Provider::Anthropic)
        .base_url(base_url)
        .input(vec![InputType::Text, InputType::Image])
        .context_window(200000)
        .max_tokens(8192)
        .build()
        .unwrap()
}

fn make_context(system_prompt: &str, user_msg: &str) -> Context {
    let mut ctx = Context::with_system_prompt(system_prompt);
    ctx.add_message(Message::User(UserMessage::text(user_msg)));
    ctx
}

fn make_options(api_key: &str) -> StreamOptions {
    StreamOptions {
        api_key: Some(api_key.to_string()),
        ..Default::default()
    }
}

/// Build an Anthropic SSE response body from a list of (event_type, json_data) pairs.
fn anthropic_sse(events: Vec<(&str, &str)>) -> String {
    events
        .iter()
        .map(|(event_type, data)| format!("event: {}\ndata: {}\n\n", event_type, data))
        .collect::<String>()
}

// ============================================================================
// Provider unit tests
// ============================================================================

#[test]
fn test_provider_type() {
    let provider = AnthropicProtocol::new();
    assert_eq!(provider.provider_type(), Provider::Anthropic);
}

// ============================================================================
// Streaming integration tests with wiremock
// ============================================================================

#[tokio::test]
async fn test_stream_simple_text_response() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        (
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_01",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0
                    }
                }
            })
            .to_string(),
        ),
        (
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            })
            .to_string(),
        ),
        ("ping", &json!({"type": "ping"}).to_string()),
        (
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"}
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world!"}
            })
            .to_string(),
        ),
        (
            "content_block_stop",
            &json!({
                "type": "content_block_stop",
                "index": 0
            })
            .to_string(),
        ),
        (
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 15}
            })
            .to_string(),
        ),
        ("message_stop", &json!({"type": "message_stop"}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let options = make_options("test-key");

    let mut stream = provider.stream(&model, &context, options);

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Should have: Start, TextStart, TextDelta("Hello"), TextDelta(" world!"), TextEnd
    assert!(!events.is_empty());

    // Check Start event
    assert!(matches!(&events[0], AssistantMessageEvent::Start { .. }));

    // Check that text deltas are present
    let text_deltas: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AssistantMessageEvent::TextDelta { .. }))
        .collect();
    assert!(!text_deltas.is_empty());

    // Verify via result too
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Hello world!");
}

#[tokio::test]
async fn test_stream_with_tool_call() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        (
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_02",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet",
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0
                    }
                }
            })
            .to_string(),
        ),
        (
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather"
                }
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": " \"Tokyo\"}"}
            })
            .to_string(),
        ),
        (
            "content_block_stop",
            &json!({
                "type": "content_block_stop",
                "index": 0
            })
            .to_string(),
        ),
        (
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 25}
            })
            .to_string(),
        ),
        ("message_stop", &json!({"type": "message_stop"}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_model(&server.uri());
    let mut context = make_context("You are helpful.", "What's the weather in Tokyo?");
    context.set_tools(vec![Tool::new(
        "get_weather",
        "Get weather",
        json!({"type": "object", "properties": {"city": {"type": "string"}}}),
    )]);
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert!(result.has_tool_calls());
    let tool_calls = result.tool_calls();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].name, "get_weather");
    assert_eq!(tool_calls[0].id, "toolu_01");
    assert_eq!(tool_calls[0].arguments["city"], "Tokyo");
}

#[tokio::test]
async fn test_stream_http_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_string(
            r#"{"error": {"type": "authentication_error", "message": "Invalid API key"}}"#,
        ))
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let options = make_options("invalid-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert!(result.error_message.is_some());
    assert!(result.error_message.unwrap().contains("401"));
}

#[tokio::test]
async fn test_stream_with_thinking() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        (
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_03",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet",
                    "usage": {
                        "input_tokens": 15,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0
                    }
                }
            })
            .to_string(),
        ),
        // Thinking block
        (
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""}
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me think"}
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": " about this..."}
            })
            .to_string(),
        ),
        (
            "content_block_stop",
            &json!({
                "type": "content_block_stop",
                "index": 0
            })
            .to_string(),
        ),
        // Text block
        (
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""}
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "The answer is 42."}
            })
            .to_string(),
        ),
        (
            "content_block_stop",
            &json!({
                "type": "content_block_stop",
                "index": 1
            })
            .to_string(),
        ),
        (
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 30}
            })
            .to_string(),
        ),
        ("message_stop", &json!({"type": "message_stop"}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "What is the meaning of life?");
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "The answer is 42.");
    assert!(result
        .thinking_content()
        .contains("Let me think about this..."));
}

#[tokio::test]
async fn test_stream_usage_tracking() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        (
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_04",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 30,
                        "cache_creation_input_tokens": 20
                    }
                }
            })
            .to_string(),
        ),
        (
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi"}
            })
            .to_string(),
        ),
        (
            "content_block_stop",
            &json!({
                "type": "content_block_stop",
                "index": 0
            })
            .to_string(),
        ),
        (
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 50}
            })
            .to_string(),
        ),
        ("message_stop", &json!({"type": "message_stop"}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    // input_tokens from message_start
    assert_eq!(result.usage.input, 100);
    // output_tokens updated by message_delta
    assert_eq!(result.usage.output, 50);
    // cache fields from message_start
    assert_eq!(result.usage.cache_read, 30);
    assert_eq!(result.usage.cache_write, 20);
    // total = input + output + cache_read + cache_write
    assert_eq!(result.usage.total_tokens, 100 + 50 + 30 + 20);
}

#[tokio::test]
async fn test_stream_length_stop_reason() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        (
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_05",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0
                    }
                }
            })
            .to_string(),
        ),
        (
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "truncated"}
            })
            .to_string(),
        ),
        (
            "content_block_stop",
            &json!({
                "type": "content_block_stop",
                "index": 0
            })
            .to_string(),
        ),
        (
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": "max_tokens"},
                "usage": {"output_tokens": 100}
            })
            .to_string(),
        ),
        ("message_stop", &json!({"type": "message_stop"}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Length);
    assert_eq!(result.text_content(), "truncated");
}

#[tokio::test]
async fn test_stream_sse_error_event() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        (
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_06",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0
                    }
                }
            })
            .to_string(),
        ),
        (
            "error",
            &json!({
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "Overloaded"
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert!(result.error_message.is_some());
    assert!(result.error_message.unwrap().contains("Overloaded"));
}

// ============================================================================
// Additional coverage: with_api_key, default, redacted thinking, signature delta,
// stop_sequence, unknown event type, stream_simple, DONE handling
// ============================================================================

#[test]
fn test_provider_with_api_key() {
    let provider = AnthropicProtocol::with_api_key("sk-ant-test");
    assert_eq!(provider.provider_type(), Provider::Anthropic);
}

#[test]
fn test_provider_default() {
    let provider = AnthropicProtocol::default();
    assert_eq!(provider.provider_type(), Provider::Anthropic);
}

#[tokio::test]
async fn test_stream_simple_delegates_correctly() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        ("message_start", &json!({"type":"message_start","message":{"id":"msg_s","type":"message","role":"assistant","model":"claude-3-5-sonnet","usage":{"input_tokens":5,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}).to_string()),
        ("content_block_start", &json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}).to_string()),
        ("content_block_delta", &json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"simple"}}).to_string()),
        ("content_block_stop", &json!({"type":"content_block_stop","index":0}).to_string()),
        ("message_delta", &json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}).to_string()),
        ("message_stop", &json!({"type":"message_stop"}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let stream = provider.stream_simple(
        &model,
        &context,
        SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("key".into()),
                ..Default::default()
            },
            reasoning: None,
            thinking_budget_tokens: None,
        },
    );
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "simple");
}

#[tokio::test]
async fn test_stream_stop_sequence_reason() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        ("message_start", &json!({"type":"message_start","message":{"id":"msg_sq","type":"message","role":"assistant","model":"claude-3-5-sonnet","usage":{"input_tokens":5,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}).to_string()),
        ("content_block_start", &json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}).to_string()),
        ("content_block_delta", &json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"stopped"}}).to_string()),
        ("content_block_stop", &json!({"type":"content_block_stop","index":0}).to_string()),
        ("message_delta", &json!({"type":"message_delta","delta":{"stop_reason":"stop_sequence"},"usage":{"output_tokens":1}}).to_string()),
        ("message_stop", &json!({"type":"message_stop"}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "stopped");
}

#[tokio::test]
async fn test_stream_redacted_thinking() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        ("message_start", &json!({"type":"message_start","message":{"id":"msg_rt","type":"message","role":"assistant","model":"claude-3-5-sonnet","usage":{"input_tokens":5,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}).to_string()),
        ("content_block_start", &json!({"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking"}}).to_string()),
        ("content_block_stop", &json!({"type":"content_block_stop","index":0}).to_string()),
        ("content_block_start", &json!({"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}).to_string()),
        ("content_block_delta", &json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}).to_string()),
        ("content_block_stop", &json!({"type":"content_block_stop","index":1}).to_string()),
        ("message_delta", &json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}).to_string()),
        ("message_stop", &json!({"type":"message_stop"}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "think about it");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "answer");
}

#[tokio::test]
async fn test_stream_with_signature_delta() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        ("message_start", &json!({"type":"message_start","message":{"id":"msg_sig","type":"message","role":"assistant","model":"claude-3-5-sonnet","usage":{"input_tokens":5,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}).to_string()),
        ("content_block_start", &json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}).to_string()),
        ("content_block_delta", &json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"deep thought"}}).to_string()),
        ("content_block_delta", &json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_abc123"}}).to_string()),
        ("content_block_stop", &json!({"type":"content_block_stop","index":0}).to_string()),
        ("content_block_start", &json!({"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}).to_string()),
        ("content_block_delta", &json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"result"}}).to_string()),
        ("content_block_stop", &json!({"type":"content_block_stop","index":1}).to_string()),
        ("message_delta", &json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}).to_string()),
        ("message_stop", &json!({"type":"message_stop"}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "think deeply");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "result");
    assert!(result.thinking_content().contains("deep thought"));
}

#[tokio::test]
async fn test_stream_unknown_event_type_ignored() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        ("message_start", &json!({"type":"message_start","message":{"id":"msg_unk","type":"message","role":"assistant","model":"claude-3-5-sonnet","usage":{"input_tokens":5,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}).to_string()),
        ("custom_unknown_event", &json!({"type":"custom_unknown","data":"ignored"}).to_string()),
        ("content_block_start", &json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}).to_string()),
        ("content_block_delta", &json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}).to_string()),
        ("content_block_stop", &json!({"type":"content_block_stop","index":0}).to_string()),
        ("message_delta", &json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}).to_string()),
        ("message_stop", &json!({"type":"message_stop"}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "ok");
}

#[tokio::test]
async fn test_stream_done_in_data() {
    let server = MockServer::start().await;

    // Some implementations send [DONE] in data field
    let sse_body = anthropic_sse(vec![
        ("message_start", &json!({"type":"message_start","message":{"id":"msg_d","type":"message","role":"assistant","model":"claude-3-5-sonnet","usage":{"input_tokens":5,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}).to_string()),
        ("content_block_start", &json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}).to_string()),
        ("content_block_delta", &json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}).to_string()),
        ("content_block_stop", &json!({"type":"content_block_stop","index":0}).to_string()),
        ("message_delta", &json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}).to_string()),
        ("message_stop", &json!({"type":"message_stop"}).to_string()),
    ]);
    // Append a [DONE] line
    let sse_body = sse_body + "data: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "hi");
}

// ============================================================================
// Message conversion coverage: multi-turn conversations
// ============================================================================

#[tokio::test]
async fn test_stream_multiturn_with_tool_calls_and_results() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        ("message_start", &json!({"type":"message_start","message":{"id":"msg_mt","type":"message","role":"assistant","model":"claude-3-5-sonnet","usage":{"input_tokens":50,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}).to_string()),
        ("content_block_start", &json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}).to_string()),
        ("content_block_delta", &json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"continued"}}).to_string()),
        ("content_block_stop", &json!({"type":"content_block_stop","index":0}).to_string()),
        ("message_delta", &json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}).to_string()),
        ("message_stop", &json!({"type":"message_stop"}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut ctx = Context::with_system_prompt("system");
    ctx.add_message(Message::User(UserMessage::text("hello")));

    // Previous assistant with thinking + text
    let asst = AssistantMessage::builder()
        .api(Api::AnthropicMessages)
        .provider(Provider::Anthropic)
        .model("claude-3-5-sonnet")
        .content(vec![
            ContentBlock::Thinking(ThinkingContent {
                thinking: "Let me consider...".to_string(),
                thinking_signature: Some("sig_1".to_string()),
                redacted: false,
            }),
            ContentBlock::Text(TextContent {
                text: "response".to_string(),
                text_signature: None,
            }),
        ])
        .stop_reason(StopReason::Stop)
        .build()
        .unwrap();
    ctx.add_message(Message::Assistant(asst));

    ctx.add_message(Message::User(UserMessage::text("now search")));

    // Assistant with tool call
    let asst2 = AssistantMessage::builder()
        .api(Api::AnthropicMessages)
        .provider(Provider::Anthropic)
        .model("claude-3-5-sonnet")
        .content(vec![ContentBlock::ToolCall(ToolCall {
            id: "tc_1".to_string(),
            name: "search".to_string(),
            arguments: json!({"q": "test"}),
            thought_signature: None,
        })])
        .stop_reason(StopReason::ToolUse)
        .build()
        .unwrap();
    ctx.add_message(Message::Assistant(asst2));

    // Tool result
    ctx.add_message(Message::ToolResult(ToolResultMessage::text(
        "tc_1",
        "search",
        "found result",
        false,
    )));

    // Errored assistant (should be skipped)
    let asst_err = AssistantMessage::builder()
        .api(Api::AnthropicMessages)
        .provider(Provider::Anthropic)
        .model("claude-3-5-sonnet")
        .content(vec![ContentBlock::Text(TextContent {
            text: "err".to_string(),
            text_signature: None,
        })])
        .stop_reason(StopReason::Error)
        .build()
        .unwrap();
    ctx.add_message(Message::Assistant(asst_err));

    // Aborted assistant (should also be skipped)
    let asst_abort = AssistantMessage::builder()
        .api(Api::AnthropicMessages)
        .provider(Provider::Anthropic)
        .model("claude-3-5-sonnet")
        .content(vec![ContentBlock::Text(TextContent {
            text: "abort".to_string(),
            text_signature: None,
        })])
        .stop_reason(StopReason::Aborted)
        .build()
        .unwrap();
    ctx.add_message(Message::Assistant(asst_abort));

    ctx.add_message(Message::User(UserMessage::text("continue")));
    ctx.set_tools(vec![Tool::new(
        "search",
        "Search",
        json!({"type":"object","properties":{"q":{"type":"string"}}}),
    )]);

    let model = make_model(&server.uri());
    let provider = AnthropicProtocol::new();
    let options = make_options("key");

    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "continued");
}

#[tokio::test]
async fn test_stream_with_redacted_thinking_in_context() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        ("message_start", &json!({"type":"message_start","message":{"id":"msg_rd","type":"message","role":"assistant","model":"claude-3-5-sonnet","usage":{"input_tokens":20,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}).to_string()),
        ("content_block_start", &json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}).to_string()),
        ("content_block_delta", &json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}).to_string()),
        ("content_block_stop", &json!({"type":"content_block_stop","index":0}).to_string()),
        ("message_delta", &json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}).to_string()),
        ("message_stop", &json!({"type":"message_stop"}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut ctx = Context::with_system_prompt("test");
    ctx.add_message(Message::User(UserMessage::text("hello")));

    // Previous assistant with redacted thinking
    let asst = AssistantMessage::builder()
        .api(Api::AnthropicMessages)
        .provider(Provider::Anthropic)
        .model("claude-3-5-sonnet")
        .content(vec![
            ContentBlock::Thinking(ThinkingContent {
                thinking: "redacted_data".to_string(),
                thinking_signature: None,
                redacted: true,
            }),
            ContentBlock::Text(TextContent {
                text: "prev response".to_string(),
                text_signature: None,
            }),
        ])
        .stop_reason(StopReason::Stop)
        .build()
        .unwrap();
    ctx.add_message(Message::Assistant(asst));
    ctx.add_message(Message::User(UserMessage::text("go on")));

    let model = make_model(&server.uri());
    let provider = AnthropicProtocol::new();
    let options = make_options("key");

    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "ok");
}

#[tokio::test]
async fn test_stream_with_image_user_content() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        ("message_start", &json!({"type":"message_start","message":{"id":"msg_img","type":"message","role":"assistant","model":"claude-3-5-sonnet","usage":{"input_tokens":30,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}).to_string()),
        ("content_block_start", &json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}).to_string()),
        ("content_block_delta", &json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I see an image"}}).to_string()),
        ("content_block_stop", &json!({"type":"content_block_stop","index":0}).to_string()),
        ("message_delta", &json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}).to_string()),
        ("message_stop", &json!({"type":"message_stop"}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut ctx = Context::with_system_prompt("test");
    // User message with blocks including an image
    ctx.add_message(Message::User(UserMessage {
        role: Role::User,
        content: UserContent::Blocks(vec![
            ContentBlock::Text(TextContent {
                text: "What is this?".to_string(),
                text_signature: None,
            }),
            ContentBlock::Image(ImageContent {
                mime_type: "image/png".to_string(),
                data: "iVBORw0KGgo=".to_string(),
            }),
        ]),
        timestamp: 0,
    }));

    let model = make_model(&server.uri());
    let provider = AnthropicProtocol::new();
    let options = make_options("key");

    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "I see an image");
}

#[tokio::test]
async fn test_stream_http_error_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(529).set_body_string(
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        ))
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Error);
}

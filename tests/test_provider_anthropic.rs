//! Tests for Anthropic Messages provider using wiremock for HTTP mocking.

use serde_json::json;
use tiy_core::types::*;
use tiy_core::provider::LLMProvider;
use tiy_core::provider::anthropic::AnthropicProvider;
use futures::StreamExt;
use wiremock::{MockServer, Mock, ResponseTemplate};
use wiremock::matchers::{method, path};

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
    events.iter()
        .map(|(event_type, data)| format!("event: {}\ndata: {}\n\n", event_type, data))
        .collect::<String>()
}

// ============================================================================
// Provider unit tests
// ============================================================================

#[test]
fn test_provider_api_type() {
    let provider = AnthropicProvider::new();
    assert_eq!(provider.api_type(), Api::AnthropicMessages);
}

// ============================================================================
// Streaming integration tests with wiremock
// ============================================================================

#[tokio::test]
async fn test_stream_simple_text_response() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        ("message_start", &json!({
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
        }).to_string()),
        ("content_block_start", &json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }).to_string()),
        ("ping", &json!({"type": "ping"}).to_string()),
        ("content_block_delta", &json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"}
        }).to_string()),
        ("content_block_delta", &json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " world!"}
        }).to_string()),
        ("content_block_stop", &json!({
            "type": "content_block_stop",
            "index": 0
        }).to_string()),
        ("message_delta", &json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 15}
        }).to_string()),
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

    let provider = AnthropicProvider::new();
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
    let text_deltas: Vec<_> = events.iter()
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
        ("message_start", &json!({
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
        }).to_string()),
        ("content_block_start", &json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "get_weather"
            }
        }).to_string()),
        ("content_block_delta", &json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}
        }).to_string()),
        ("content_block_delta", &json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": " \"Tokyo\"}"}
        }).to_string()),
        ("content_block_stop", &json!({
            "type": "content_block_stop",
            "index": 0
        }).to_string()),
        ("message_delta", &json!({
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 25}
        }).to_string()),
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

    let provider = AnthropicProvider::new();
    let model = make_model(&server.uri());
    let mut context = make_context("You are helpful.", "What's the weather in Tokyo?");
    context.set_tools(vec![
        Tool::new("get_weather", "Get weather", json!({"type": "object", "properties": {"city": {"type": "string"}}})),
    ]);
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
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_string(r#"{"error": {"type": "authentication_error", "message": "Invalid API key"}}"#),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new();
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
        ("message_start", &json!({
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
        }).to_string()),
        // Thinking block
        ("content_block_start", &json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""}
        }).to_string()),
        ("content_block_delta", &json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Let me think"}
        }).to_string()),
        ("content_block_delta", &json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": " about this..."}
        }).to_string()),
        ("content_block_stop", &json!({
            "type": "content_block_stop",
            "index": 0
        }).to_string()),
        // Text block
        ("content_block_start", &json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""}
        }).to_string()),
        ("content_block_delta", &json!({
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "The answer is 42."}
        }).to_string()),
        ("content_block_stop", &json!({
            "type": "content_block_stop",
            "index": 1
        }).to_string()),
        ("message_delta", &json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 30}
        }).to_string()),
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

    let provider = AnthropicProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "What is the meaning of life?");
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "The answer is 42.");
    assert!(result.thinking_content().contains("Let me think about this..."));
}

#[tokio::test]
async fn test_stream_usage_tracking() {
    let server = MockServer::start().await;

    let sse_body = anthropic_sse(vec![
        ("message_start", &json!({
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
        }).to_string()),
        ("content_block_start", &json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }).to_string()),
        ("content_block_delta", &json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hi"}
        }).to_string()),
        ("content_block_stop", &json!({
            "type": "content_block_stop",
            "index": 0
        }).to_string()),
        ("message_delta", &json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50}
        }).to_string()),
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

    let provider = AnthropicProvider::new();
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
        ("message_start", &json!({
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
        }).to_string()),
        ("content_block_start", &json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }).to_string()),
        ("content_block_delta", &json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "truncated"}
        }).to_string()),
        ("content_block_stop", &json!({
            "type": "content_block_stop",
            "index": 0
        }).to_string()),
        ("message_delta", &json!({
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
            "usage": {"output_tokens": 100}
        }).to_string()),
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

    let provider = AnthropicProvider::new();
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
        ("message_start", &json!({
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
        }).to_string()),
        ("error", &json!({
            "type": "error",
            "error": {
                "type": "overloaded_error",
                "message": "Overloaded"
            }
        }).to_string()),
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

    let provider = AnthropicProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert!(result.error_message.is_some());
    assert!(result.error_message.unwrap().contains("Overloaded"));
}

//! Tests for OpenAI Completions provider using wiremock for HTTP mocking.

use futures::StreamExt;
use serde_json::json;
use tiy_core::provider::openai_completions::OpenAICompletionsProvider;
use tiy_core::provider::LLMProvider;
use tiy_core::types::*;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ============================================================================
// Helper functions
// ============================================================================

fn make_model(base_url: &str) -> Model {
    Model::builder()
        .id("gpt-4o-mini")
        .name("GPT-4o Mini")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url(base_url)
        .input(vec![InputType::Text, InputType::Image])
        .context_window(128000)
        .max_tokens(16384)
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

fn sse_response(chunks: Vec<&str>) -> String {
    chunks
        .iter()
        .map(|c| format!("data: {}\n\n", c))
        .collect::<Vec<_>>()
        .join("")
        + "data: [DONE]\n\n"
}

// ============================================================================
// Provider unit tests
// ============================================================================

#[test]
fn test_provider_type() {
    let provider = OpenAICompletionsProvider::new();
    assert_eq!(provider.provider_type(), Provider::OpenAI);
}

#[test]
fn test_provider_with_api_key() {
    let provider = OpenAICompletionsProvider::with_api_key("sk-test-key");
    assert_eq!(provider.provider_type(), Provider::OpenAI);
}

// ============================================================================
// Streaming integration tests with wiremock
// ============================================================================

#[tokio::test]
async fn test_stream_simple_text_response() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": "Hello"},
                "finish_reason": null
            }]
        })
        .to_string(),
        &json!({
            "choices": [{
                "index": 0,
                "delta": {"content": " world!"},
                "finish_reason": null
            }]
        })
        .to_string(),
        &json!({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        })
        .to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAICompletionsProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let options = make_options("test-key");

    let mut stream = provider.stream(&model, &context, options);

    // EventStream is now properly async — stream.next().await works correctly
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Should have: Start, TextStart, TextDelta("Hello"), TextDelta(" world!")
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

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": ""
                        }
                    }]
                },
                "finish_reason": null
            }]
        })
        .to_string(),
        &json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": "{\"city\":"
                        }
                    }]
                },
                "finish_reason": null
            }]
        })
        .to_string(),
        &json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": " \"Tokyo\"}"
                        }
                    }]
                },
                "finish_reason": null
            }]
        })
        .to_string(),
        &json!({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15
            }
        })
        .to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAICompletionsProvider::new();
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
    assert_eq!(tool_calls[0].id, "call_abc123");
    assert_eq!(tool_calls[0].arguments["city"], "Tokyo");
}

#[tokio::test]
async fn test_stream_http_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_string(r#"{"error": {"message": "Invalid API key"}}"#),
        )
        .mount(&server)
        .await;

    let provider = OpenAICompletionsProvider::new();
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

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "reasoning_content": "Let me think"},
                "finish_reason": null
            }]
        })
        .to_string(),
        &json!({
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": " about this..."},
                "finish_reason": null
            }]
        })
        .to_string(),
        &json!({
            "choices": [{
                "index": 0,
                "delta": {"content": "The answer is 42."},
                "finish_reason": null
            }]
        })
        .to_string(),
        &json!({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        })
        .to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAICompletionsProvider::new();
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

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": "Hi"},
                "finish_reason": null
            }]
        })
        .to_string(),
        &json!({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 30},
                "completion_tokens_details": {"reasoning_tokens": 10}
            }
        })
        .to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAICompletionsProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.usage.input, 100);
    assert_eq!(result.usage.output, 50);
    assert_eq!(result.usage.total_tokens, 150);
}

#[tokio::test]
async fn test_stream_with_custom_headers() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![&json!({
        "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("x-custom-header", "custom-value"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAICompletionsProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let mut headers = std::collections::HashMap::new();
    headers.insert("x-custom-header".to_string(), "custom-value".to_string());
    let options = StreamOptions {
        api_key: Some("key".into()),
        headers: Some(headers),
        ..Default::default()
    };

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Stop);
}

#[tokio::test]
async fn test_stream_empty_response() {
    let server = MockServer::start().await;

    // Empty stream - just DONE
    let sse_body = "data: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAICompletionsProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert!(result.content.is_empty());
}

#[tokio::test]
async fn test_stream_length_stop_reason() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {"content": "truncated"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 100}
        })
        .to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAICompletionsProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Length);
}

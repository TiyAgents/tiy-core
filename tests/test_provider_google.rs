//! Tests for Google Generative AI provider using wiremock for HTTP mocking.

use serde_json::json;
use tiy_core::types::*;
use tiy_core::provider::LLMProvider;
use tiy_core::provider::google::GoogleProvider;
use futures::StreamExt;
use wiremock::{MockServer, Mock, ResponseTemplate};
use wiremock::matchers::{method, path, query_param};

// ============================================================================
// Helper functions
// ============================================================================

fn make_model(base_url: &str) -> Model {
    Model::builder()
        .id("gemini-2.0-flash")
        .name("Gemini 2.0 Flash")
        .api(Api::GoogleGenerativeAi)
        .provider(Provider::Google)
        .base_url(base_url)
        .input(vec![InputType::Text, InputType::Image])
        .context_window(1048576)
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

/// Build a Google SSE response body from JSON data chunks.
/// Google SSE format is `data: {json}\n\n` (no `event:` prefix, no `[DONE]` sentinel).
fn google_sse(chunks: Vec<&str>) -> String {
    chunks.iter()
        .map(|c| format!("data: {}\n\n", c))
        .collect::<String>()
}

// ============================================================================
// Provider unit tests
// ============================================================================

#[test]
fn test_provider_api_type() {
    let provider = GoogleProvider::new();
    assert_eq!(provider.api_type(), Api::GoogleGenerativeAi);
}

// ============================================================================
// Streaming integration tests with wiremock
// ============================================================================

#[tokio::test]
async fn test_stream_simple_text_response() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![
        &json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5
            }
        }).to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let options = make_options("test-key");

    let mut stream = provider.stream(&model, &context, options);

    // Collect all streamed events
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Should have: Start, TextStart, TextDelta("Hello"), TextEnd
    assert!(!events.is_empty());

    // Check Start event
    assert!(matches!(&events[0], AssistantMessageEvent::Start { .. }));

    // Check that text deltas are present
    let text_deltas: Vec<_> = events.iter()
        .filter(|e| matches!(e, AssistantMessageEvent::TextDelta { .. }))
        .collect();
    assert!(!text_deltas.is_empty());

    // Verify via result
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Hello");
    assert_eq!(result.usage.input, 10);
    assert_eq!(result.usage.output, 5);
}

#[tokio::test]
async fn test_stream_with_tool_call() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![
        &json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Tokyo"}
                        }
                    }],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 15
            }
        }).to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProvider::new();
    let model = make_model(&server.uri());
    let mut context = make_context("You are helpful.", "What's the weather in Tokyo?");
    context.set_tools(vec![
        Tool::new(
            "get_weather",
            "Get weather for a city",
            json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        ),
    ]);
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert!(result.has_tool_calls());
    let tool_calls = result.tool_calls();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].name, "get_weather");
    assert_eq!(tool_calls[0].arguments["city"], "Tokyo");
}

#[tokio::test]
async fn test_stream_http_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("key", "invalid-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_string(r#"{"error": {"message": "API key not valid. Please pass a valid API key."}}"#),
        )
        .mount(&server)
        .await;

    let provider = GoogleProvider::new();
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

    let sse_body = google_sse(vec![
        // First chunk: thinking content
        &json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "thought": true,
                        "text": "Let me think about this..."
                    }],
                    "role": "model"
                }
            }]
        }).to_string(),
        // Second chunk: more thinking content
        &json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "thought": true,
                        "text": " The answer involves computation."
                    }],
                    "role": "model"
                }
            }]
        }).to_string(),
        // Third chunk: actual text response with finish reason
        &json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "The answer is 42."}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20
            }
        }).to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "What is the meaning of life?");
    let options = make_options("test-key");

    let mut stream = provider.stream(&model, &context, options);

    // Collect all streamed events
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Should have thinking events followed by text events
    let thinking_deltas: Vec<_> = events.iter()
        .filter(|e| matches!(e, AssistantMessageEvent::ThinkingDelta { .. }))
        .collect();
    assert!(!thinking_deltas.is_empty(), "Expected thinking delta events");

    let text_deltas: Vec<_> = events.iter()
        .filter(|e| matches!(e, AssistantMessageEvent::TextDelta { .. }))
        .collect();
    assert!(!text_deltas.is_empty(), "Expected text delta events");

    // Verify via result
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "The answer is 42.");
    assert!(result.thinking_content().contains("Let me think about this..."));
    assert!(result.thinking_content().contains("The answer involves computation."));
}

#[tokio::test]
async fn test_stream_usage_tracking() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![
        &json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hi"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50
            }
        }).to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.usage.input, 100);
    assert_eq!(result.usage.output, 50);
    assert_eq!(result.usage.total_tokens, 150);
}

#[tokio::test]
async fn test_stream_length_stop_reason() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![
        &json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "This response was truncated because"}],
                    "role": "model"
                }
            }]
        }).to_string(),
        &json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": " it exceeded the maximum token limit."}],
                    "role": "model"
                },
                "finishReason": "MAX_TOKENS"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 100
            }
        }).to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "Write a very long essay.");
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Length);
    assert_eq!(
        result.text_content(),
        "This response was truncated because it exceeded the maximum token limit."
    );
}

#[tokio::test]
async fn test_stream_multiple_text_chunks() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![
        // First chunk: start of text
        &json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello"}],
                    "role": "model"
                }
            }]
        }).to_string(),
        // Second chunk: more text
        &json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": ", "}],
                    "role": "model"
                }
            }]
        }).to_string(),
        // Third chunk: more text
        &json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "world"}],
                    "role": "model"
                }
            }]
        }).to_string(),
        // Fourth chunk: final text with finish reason and usage
        &json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "!"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 4
            }
        }).to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Say hello world");
    let options = make_options("test-key");

    let mut stream = provider.stream(&model, &context, options);

    // Collect all streamed events
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Should have Start event
    assert!(matches!(&events[0], AssistantMessageEvent::Start { .. }));

    // Collect text deltas and verify incremental delivery
    let text_deltas: Vec<String> = events.iter()
        .filter_map(|e| match e {
            AssistantMessageEvent::TextDelta { delta, .. } => Some(delta.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text_deltas.len(), 4);
    assert_eq!(text_deltas[0], "Hello");
    assert_eq!(text_deltas[1], ", ");
    assert_eq!(text_deltas[2], "world");
    assert_eq!(text_deltas[3], "!");

    // Check TextStart and TextEnd events
    let text_starts: Vec<_> = events.iter()
        .filter(|e| matches!(e, AssistantMessageEvent::TextStart { .. }))
        .collect();
    assert_eq!(text_starts.len(), 1, "Should have exactly one TextStart event");

    let text_ends: Vec<_> = events.iter()
        .filter(|e| matches!(e, AssistantMessageEvent::TextEnd { .. }))
        .collect();
    assert_eq!(text_ends.len(), 1, "Should have exactly one TextEnd event");

    // Verify combined result
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Hello, world!");
    assert_eq!(result.usage.input, 8);
    assert_eq!(result.usage.output, 4);
    assert_eq!(result.usage.total_tokens, 12);
}

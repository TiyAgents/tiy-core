//! Tests for Google Generative AI provider using wiremock for HTTP mocking.

use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::json;
use std::sync::Arc;
use tiy_core::protocol::google::GoogleProtocol;
use tiy_core::protocol::LLMProtocol;
use tiy_core::types::*;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ============================================================================
// Helper functions
// ============================================================================

fn make_model(base_url: &str) -> Model {
    make_model_with_id(base_url, "gemini-2.0-flash")
}

fn make_model_with_id(base_url: &str, id: &str) -> Model {
    Model::builder()
        .id(id)
        .name(id)
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

fn make_options_with_capture(
    api_key: &str,
    captured: Arc<Mutex<Option<serde_json::Value>>>,
) -> StreamOptions {
    let mut options = make_options(api_key);
    options.on_payload = Some(Arc::new(move |payload, _model| {
        let captured = captured.clone();
        Box::pin(async move {
            *captured.lock() = Some(payload.clone());
            Some(payload)
        })
    }));
    options
}

/// Build a Google SSE response body from JSON data chunks.
/// Google SSE format is `data: {json}\n\n` (no `event:` prefix, no `[DONE]` sentinel).
fn google_sse(chunks: Vec<&str>) -> String {
    chunks
        .iter()
        .map(|c| format!("data: {}\n\n", c))
        .collect::<String>()
}

// ============================================================================
// Provider unit tests
// ============================================================================

#[test]
fn test_provider_type() {
    let provider = GoogleProtocol::new();
    assert_eq!(provider.provider_type(), Provider::Google);
}

// ============================================================================
// Streaming integration tests with wiremock
// ============================================================================

#[tokio::test]
async fn test_stream_simple_text_response() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![&json!({
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
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(header("x-goog-api-key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
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
    let text_deltas: Vec<_> = events
        .iter()
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

    let sse_body = google_sse(vec![&json!({
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
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(header("x-goog-api-key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
    let model = make_model(&server.uri());
    let mut context = make_context("You are helpful.", "What's the weather in Tokyo?");
    context.set_tools(vec![Tool::new(
        "get_weather",
        "Get weather for a city",
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
    assert_eq!(tool_calls[0].arguments["city"], "Tokyo");
}

#[tokio::test]
async fn test_stream_sends_tool_config_and_multimodal_function_response() {
    let server = MockServer::start().await;
    let captured = Arc::new(Mutex::new(None));

    let sse_body = google_sse(vec![&json!({
        "candidates": [{
            "content": {
                "parts": [{"text": "done"}],
                "role": "model"
            },
            "finishReason": "STOP"
        }]
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-3-pro:streamGenerateContent"))
        .and(header("x-goog-api-key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
    let model = make_model_with_id(&server.uri(), "gemini-3-pro");
    let mut context = make_context("You are helpful.", "Render an icon");
    context.set_tools(vec![Tool::new(
        "render_icon",
        "Render an icon",
        json!({"type": "object", "properties": {"shape": {"type": "string"}}}),
    )]);
    context.add_message(Message::ToolResult(ToolResultMessage::new(
        "call_1",
        "render_icon",
        vec![
            ContentBlock::Text(TextContent::new("Rendered successfully")),
            ContentBlock::Image(ImageContent::new("aGVsbG8=", "image/png")),
        ],
        false,
    )));

    let mut options = make_options_with_capture("test-key", captured.clone());
    options.tool_choice = Some(ToolChoice::Mode(ToolChoiceMode::Any));

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let payload = captured.lock().clone().expect("payload captured");
    assert_eq!(
        payload["toolConfig"]["functionCallingConfig"]["mode"],
        json!("ANY")
    );
    assert_eq!(
        payload["contents"][0]["parts"][1]["functionResponse"]["response"]["output"],
        json!("Rendered successfully")
    );
    assert_eq!(
        payload["contents"][0]["parts"][1]["functionResponse"]["parts"][0]["inlineData"]["mimeType"],
        json!("image/png")
    );
}

#[tokio::test]
async fn test_stream_applies_model_aware_tool_call_id_rules() {
    let server = MockServer::start().await;
    let captured_claude = Arc::new(Mutex::new(None));
    let captured_gemini = Arc::new(Mutex::new(None));

    let sse_body = google_sse(vec![&json!({
        "candidates": [{
            "content": {
                "parts": [{"text": "ok"}],
                "role": "model"
            },
            "finishReason": "STOP"
        }]
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/claude-3-7-sonnet:streamGenerateContent"))
        .and(header("x-goog-api-key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body.clone())
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(header("x-goog-api-key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut context = Context::new();
    context.add_message(Message::User(UserMessage::text("Use the tool")));
    context.add_message(Message::Assistant(
        AssistantMessage::builder()
            .api(Api::GoogleGenerativeAi)
            .provider(Provider::Google)
            .model("source-model")
            .content(vec![ContentBlock::ToolCall(ToolCall::new(
                "call/abc",
                "get_weather",
                json!({"city": "Tokyo"}),
            ))])
            .stop_reason(StopReason::ToolUse)
            .build()
            .unwrap(),
    ));

    let provider = GoogleProtocol::new();

    let mut claude_options = make_options_with_capture("test-key", captured_claude.clone());
    claude_options.tool_choice = Some(ToolChoice::Mode(ToolChoiceMode::Auto));
    let claude_stream = provider.stream(
        &make_model_with_id(&server.uri(), "claude-3-7-sonnet"),
        &context,
        claude_options,
    );
    let _ = claude_stream.result().await;

    let gemini_stream = provider.stream(
        &make_model_with_id(&server.uri(), "gemini-2.0-flash"),
        &context,
        make_options_with_capture("test-key", captured_gemini.clone()),
    );
    let _ = gemini_stream.result().await;

    let claude_payload = captured_claude
        .lock()
        .clone()
        .expect("claude payload captured");
    assert_eq!(
        claude_payload["contents"][1]["parts"][0]["functionCall"]["id"],
        json!("call_abc")
    );

    let gemini_payload = captured_gemini
        .lock()
        .clone()
        .expect("gemini payload captured");
    assert!(gemini_payload["contents"][1]["parts"][0]["functionCall"]["id"].is_null());
}

#[tokio::test]
async fn test_stream_http_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(header("x-goog-api-key", "invalid-key"))
        .and(query_param("alt", "sse"))
        .respond_with(ResponseTemplate::new(401).set_body_string(
            r#"{"error": {"message": "API key not valid. Please pass a valid API key."}}"#,
        ))
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
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
        })
        .to_string(),
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
        })
        .to_string(),
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
        })
        .to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(header("x-goog-api-key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
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
    let thinking_deltas: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AssistantMessageEvent::ThinkingDelta { .. }))
        .collect();
    assert!(
        !thinking_deltas.is_empty(),
        "Expected thinking delta events"
    );

    let text_deltas: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AssistantMessageEvent::TextDelta { .. }))
        .collect();
    assert!(!text_deltas.is_empty(), "Expected text delta events");

    // Verify via result
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "The answer is 42.");
    assert!(result
        .thinking_content()
        .contains("Let me think about this..."));
    assert!(result
        .thinking_content()
        .contains("The answer involves computation."));
}

#[tokio::test]
async fn test_stream_usage_tracking() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![&json!({
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
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(header("x-goog-api-key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
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
        })
        .to_string(),
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
        })
        .to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(header("x-goog-api-key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
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
        })
        .to_string(),
        // Second chunk: more text
        &json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": ", "}],
                    "role": "model"
                }
            }]
        })
        .to_string(),
        // Third chunk: more text
        &json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "world"}],
                    "role": "model"
                }
            }]
        })
        .to_string(),
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
        })
        .to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(header("x-goog-api-key", "test-key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
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
    let text_deltas: Vec<String> = events
        .iter()
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
    let text_starts: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AssistantMessageEvent::TextStart { .. }))
        .collect();
    assert_eq!(
        text_starts.len(),
        1,
        "Should have exactly one TextStart event"
    );

    let text_ends: Vec<_> = events
        .iter()
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

// ============================================================================
// Additional coverage: with_api_key, default, stream_simple, safety reasons,
// thinking with signature, function call after text, DONE handling
// ============================================================================

#[test]
fn test_provider_with_api_key() {
    let provider = GoogleProtocol::with_api_key("test-api-key");
    assert_eq!(provider.provider_type(), Provider::Google);
}

#[test]
fn test_provider_default() {
    let provider = GoogleProtocol::default();
    assert_eq!(provider.provider_type(), Provider::Google);
}

#[tokio::test]
async fn test_stream_simple_delegates_correctly() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![&json!({
        "candidates": [{"content":{"parts":[{"text":"simple"}],"role":"model"},"finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1}
    }).to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let stream = provider.stream_simple(
        &model,
        &context,
        SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("test-key".into()),
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
async fn test_stream_safety_finish_reason() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![&json!({
        "candidates": [{
            "content": {"parts": [{"text": "partial"}], "role": "model"},
            "finishReason": "SAFETY"
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1}
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "something");
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Error);
}

#[tokio::test]
async fn test_stream_recitation_finish_reason() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![&json!({
        "candidates": [{
            "content": {"parts": [{"text": "copied"}], "role": "model"},
            "finishReason": "RECITATION"
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1}
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "something");
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Error);
}

#[tokio::test]
async fn test_stream_thinking_with_signature() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![
        &json!({
            "candidates": [{"content": {"parts": [{
                "text": "deep thinking...",
                "thought": true,
                "thoughtSignature": "sig_xyz"
            }], "role": "model"}}]
        }).to_string(),
        &json!({
            "candidates": [{"content": {"parts": [{"text": "final answer"}], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5}
        }).to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "think about it");
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "final answer");
    assert!(result.thinking_content().contains("deep thinking..."));
}

#[tokio::test]
async fn test_stream_function_call_after_text() {
    let server = MockServer::start().await;

    // Text first, then a function call in same response — should close text block
    let sse_body = google_sse(vec![
        &json!({
            "candidates": [{"content": {"parts": [{"text": "Let me search"}], "role": "model"}}]
        })
        .to_string(),
        &json!({
            "candidates": [{"content": {"parts": [{
                "functionCall": {"name": "search", "args": {"q": "test"}}
            }], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10}
        })
        .to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
    let model = make_model(&server.uri());
    let mut context = make_context("test", "find info");
    context.set_tools(vec![Tool::new(
        "search",
        "Search",
        json!({"type": "object", "properties": {"q": {"type": "string"}}}),
    )]);
    let options = make_options("test-key");

    let mut stream = provider.stream(&model, &context, options);
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert_eq!(result.text_content(), "Let me search");
    assert_eq!(result.tool_calls().len(), 1);
    assert_eq!(result.tool_calls()[0].name, "search");

    // Verify TextEnd was emitted before tool call
    let text_ends: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AssistantMessageEvent::TextEnd { .. }))
        .collect();
    assert!(
        !text_ends.is_empty(),
        "TextEnd should be emitted when function call arrives"
    );
}

#[tokio::test]
async fn test_stream_function_call_after_thinking() {
    let server = MockServer::start().await;

    // Thinking then function call — should close thinking block
    let sse_body = google_sse(vec![
        &json!({
            "candidates": [{"content": {"parts": [{
                "text": "reasoning...",
                "thought": true
            }], "role": "model"}}]
        })
        .to_string(),
        &json!({
            "candidates": [{"content": {"parts": [{
                "functionCall": {"name": "calc", "args": {"expr": "1+1"}}
            }], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10}
        })
        .to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
    let model = make_model(&server.uri());
    let mut context = make_context("test", "calculate");
    context.set_tools(vec![Tool::new(
        "calc",
        "Calculate",
        json!({"type": "object", "properties": {"expr": {"type": "string"}}}),
    )]);
    let options = make_options("test-key");

    let mut stream = provider.stream(&model, &context, options);
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert!(result.thinking_content().contains("reasoning..."));
    assert_eq!(result.tool_calls().len(), 1);

    // Verify ThinkingEnd was emitted
    let thinking_ends: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AssistantMessageEvent::ThinkingEnd { .. }))
        .collect();
    assert!(
        !thinking_ends.is_empty(),
        "ThinkingEnd should be emitted when function call arrives"
    );
}

#[tokio::test]
async fn test_stream_done_line_ignored() {
    let server = MockServer::start().await;

    // Append [DONE] line which Google shouldn't normally send but should handle
    let sse_body = google_sse(vec![&json!({
        "candidates": [{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1}
    })
    .to_string()])
        + "data: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "ok");
}

#[tokio::test]
async fn test_stream_blocklist_finish_reason() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![&json!({
        "candidates": [{
            "content": {"parts": [{"text": "blocked"}], "role": "model"},
            "finishReason": "BLOCKLIST"
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1}
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Error);
}

#[tokio::test]
async fn test_stream_max_tokens_without_parts() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![&json!({
        "candidates": [{
            "content": {"role": "model"},
            "finishReason": "MAX_TOKENS"
        }],
        "usageMetadata": {"promptTokenCount": 6, "candidatesTokenCount": 4, "totalTokenCount": 10}
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Length);
    assert_eq!(result.text_content(), "");
    assert_eq!(result.usage.input, 6);
    assert_eq!(result.usage.output, 4);
}

// ============================================================================
// Message conversion coverage: multi-turn with assistant/tool messages
// ============================================================================

#[tokio::test]
async fn test_stream_with_rich_context_multiturn() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![&json!({
        "candidates": [{"content":{"parts":[{"text":"continued"}],"role":"model"},"finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":50,"candidatesTokenCount":5}
    }).to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut ctx = Context::with_system_prompt("system");
    ctx.add_message(Message::User(UserMessage::text("hello")));
    let asst = AssistantMessage::builder()
        .api(Api::GoogleGenerativeAi)
        .provider(Provider::Google)
        .model("gemini-2.0-flash")
        .content(vec![
            ContentBlock::Thinking(ThinkingContent {
                thinking: "Thinking...".to_string(),
                thinking_signature: Some("sig1".to_string()),
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
    ctx.add_message(Message::User(UserMessage::text("now use a tool")));
    let asst2 = AssistantMessage::builder()
        .api(Api::GoogleGenerativeAi)
        .provider(Provider::Google)
        .model("gemini-2.0-flash")
        .content(vec![ContentBlock::ToolCall(ToolCall {
            id: "tc_1".to_string(),
            name: "search".to_string(),
            arguments: json!({"q": "test"}),
            thought_signature: Some("sig2".to_string()),
        })])
        .stop_reason(StopReason::ToolUse)
        .build()
        .unwrap();
    ctx.add_message(Message::Assistant(asst2));
    ctx.add_message(Message::ToolResult(ToolResultMessage::text(
        "tc_1",
        "search",
        "result data",
        false,
    )));
    let asst_err = AssistantMessage::builder()
        .api(Api::GoogleGenerativeAi)
        .provider(Provider::Google)
        .model("gemini-2.0-flash")
        .content(vec![ContentBlock::Text(TextContent {
            text: "err".to_string(),
            text_signature: None,
        })])
        .stop_reason(StopReason::Error)
        .build()
        .unwrap();
    ctx.add_message(Message::Assistant(asst_err));
    ctx.add_message(Message::User(UserMessage::text("continue")));
    ctx.set_tools(vec![Tool::new(
        "search",
        "Search",
        json!({"type":"object","properties":{"q":{"type":"string"}}}),
    )]);

    let model = make_model(&server.uri());
    let provider = GoogleProtocol::new();
    let options = make_options("test-key");
    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "continued");
}

#[tokio::test]
async fn test_stream_google_replays_cross_provider_thinking_as_text() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![&json!({
        "candidates": [{"content":{"parts":[{"text":"continued"}],"role":"model"},"finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":50,"candidatesTokenCount":5}
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut ctx = Context::with_system_prompt("system");
    ctx.add_message(Message::User(UserMessage::text("hello")));
    ctx.add_message(Message::Assistant(
        AssistantMessage::builder()
            .api(Api::AnthropicMessages)
            .provider(Provider::Anthropic)
            .model("claude-3-5-sonnet")
            .content(vec![
                ContentBlock::Thinking(ThinkingContent::new("Deep thought")),
                ContentBlock::Text(TextContent::new("answer")),
            ])
            .stop_reason(StopReason::Stop)
            .build()
            .unwrap(),
    ));
    ctx.add_message(Message::User(UserMessage::text("continue")));

    let provider = GoogleProtocol::new();
    let model = make_model(&server.uri());
    let captured = Arc::new(Mutex::new(None));
    let options = make_options_with_capture("test-key", captured.clone());

    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let payload = captured.lock().clone().expect("payload should be captured");
    let contents = payload["contents"]
        .as_array()
        .expect("contents should be an array");
    let assistant_parts = contents[1]["parts"]
        .as_array()
        .expect("assistant parts should be an array");

    assert_eq!(assistant_parts[0], json!({ "text": "Deep thought" }));
    assert_eq!(assistant_parts[1], json!({ "text": "answer" }));
}

#[tokio::test]
async fn test_stream_with_error_tool_result() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![&json!({
        "candidates": [{"content":{"parts":[{"text":"error handled"}],"role":"model"},"finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":5}
    }).to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut ctx = Context::with_system_prompt("test");
    ctx.add_message(Message::User(UserMessage::text("use tool")));
    let asst = AssistantMessage::builder()
        .api(Api::GoogleGenerativeAi)
        .provider(Provider::Google)
        .model("gemini-2.0-flash")
        .content(vec![ContentBlock::ToolCall(ToolCall {
            id: "tc_e".to_string(),
            name: "fn1".to_string(),
            arguments: json!({}),
            thought_signature: None,
        })])
        .stop_reason(StopReason::ToolUse)
        .build()
        .unwrap();
    ctx.add_message(Message::Assistant(asst));
    ctx.add_message(Message::ToolResult(ToolResultMessage::text(
        "tc_e",
        "fn1",
        "error occurred",
        true,
    )));
    ctx.add_message(Message::User(UserMessage::text("retry")));

    let model = make_model(&server.uri());
    let provider = GoogleProtocol::new();
    let options = make_options("test-key");
    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "error handled");
}

#[tokio::test]
async fn test_stream_vertex_ai_mode() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![&json!({
        "candidates": [{"content":{"parts":[{"text":"vertex response"}],"role":"model"},"finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2}
    }).to_string()]);

    // Vertex AI URL format: /v1/publishers/google/models/{model}:streamGenerateContent
    Mock::given(method("POST"))
        .and(path(
            "/v1/publishers/google/models/gemini-2.0-flash:streamGenerateContent",
        ))
        .and(query_param("alt", "sse"))
        .and(header("authorization", "Bearer vertex-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let model = Model::builder()
        .id("gemini-2.0-flash")
        .name("Gemini 2.0 Flash (Vertex)")
        .api(Api::GoogleVertex)
        .provider(Provider::Google)
        .base_url(&server.uri())
        .context_window(1048576)
        .max_tokens(8192)
        .build()
        .unwrap();

    let context = make_context("test", "hello");
    let options = StreamOptions {
        api_key: Some("vertex-key".to_string()),
        ..Default::default()
    };
    let provider = GoogleProtocol::new();
    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "vertex response");
}

#[tokio::test]
async fn test_stream_strips_google_vendor_prefix_in_generative_ai_mode() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![&json!({
        "candidates": [{"content":{"parts":[{"text":"prefixed response"}],"role":"model"},"finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2}
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-pro:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .and(header("x-goog-api-key", "test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut model = make_model(&server.uri());
    model.id = "google/gemini-2.5-pro".to_string();

    let context = make_context("test", "hello");
    let options = make_options("test-key");
    let provider = GoogleProtocol::new();
    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "prefixed response");
}

#[tokio::test]
async fn test_stream_strips_google_vendor_prefix_in_vertex_mode() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![&json!({
        "candidates": [{"content":{"parts":[{"text":"vertex prefixed response"}],"role":"model"},"finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2}
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path(
            "/v1/publishers/google/models/gemini-2.5-pro:streamGenerateContent",
        ))
        .and(query_param("alt", "sse"))
        .and(header("authorization", "Bearer vertex-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut model = make_model(&server.uri());
    model.id = "google/gemini-2.5-pro".to_string();
    model.api = Some(Api::GoogleVertex);

    let context = make_context("test", "hello");
    let options = StreamOptions {
        api_key: Some("vertex-key".to_string()),
        ..Default::default()
    };
    let provider = GoogleProtocol::new();
    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "vertex prefixed response");
}

#[tokio::test]
async fn test_stream_with_image_user_content() {
    let server = MockServer::start().await;

    let sse_body = google_sse(vec![&json!({
        "candidates": [{"content":{"parts":[{"text":"I see an image"}],"role":"model"},"finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":3}
    }).to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut ctx = Context::with_system_prompt("test");
    ctx.add_message(Message::User(UserMessage {
        role: Role::User,
        content: UserContent::Blocks(vec![
            ContentBlock::Text(TextContent {
                text: "Describe this".to_string(),
                text_signature: None,
            }),
            ContentBlock::Image(ImageContent {
                mime_type: "image/png".to_string(),
                data: "iVBORw0KGgo=".to_string(),
            }),
        ]),
        timestamp: 0,
    }));

    let provider = GoogleProtocol::new();
    let model = Model::builder()
        .id("gemini-2.0-flash")
        .name("Gemini 2.0 Flash")
        .api(Api::GoogleGenerativeAi)
        .provider(Provider::Google)
        .base_url(&server.uri())
        .input(vec![InputType::Text, InputType::Image])
        .context_window(1048576)
        .max_tokens(8192)
        .build()
        .unwrap();

    let options = make_options("test-key");
    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "I see an image");
}

#[tokio::test]
async fn test_stream_http_error_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .respond_with(ResponseTemplate::new(503).set_body_string("Service unavailable"))
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Error);
}

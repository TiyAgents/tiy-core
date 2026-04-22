//! Tests for OpenAI Completions provider using wiremock for HTTP mocking.

use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::json;
use std::sync::Arc;
use tiycore::protocol::openai_completions::OpenAICompletionsProtocol;
use tiycore::protocol::LLMProtocol;
use tiycore::types::*;
use wiremock::matchers::{body_partial_json, header, method, path};
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
    let provider = OpenAICompletionsProtocol::new();
    assert_eq!(provider.provider_type(), Provider::OpenAI);
}

#[test]
fn test_provider_with_api_key() {
    let provider = OpenAICompletionsProtocol::with_api_key("sk-test-key");
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

    let provider = OpenAICompletionsProtocol::new();
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
async fn test_stream_reports_incomplete_stream_for_truncated_response() {
    let server = MockServer::start().await;

    let sse_body = [
        format!(
            "data: {}\n\n",
            json!({
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": null
                }]
            })
        ),
        format!(
            "data: {}\n\n",
            json!({
                "choices": [{
                    "index": 0,
                    "delta": {"content": " world"},
                    "finish_reason": null
                }]
            })
        ),
    ]
    .join("");

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert!(result
        .error_message
        .as_deref()
        .is_some_and(|message| message.contains("[incomplete_stream]openai_completions:")));
    assert!(result
        .error_message
        .as_deref()
        .is_some_and(|message| message.contains("missing finish_reason")));
}

#[tokio::test]
async fn test_stream_clamps_small_max_completion_tokens_to_minimum() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": "ok"},
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
                "completion_tokens": 2
            }
        })
        .to_string(),
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "max_completion_tokens": 16
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let mut options = make_options("test-key");
    options.max_tokens = Some(8);

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "ok");
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

    let provider = OpenAICompletionsProtocol::new();
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

    let provider = OpenAICompletionsProtocol::new();
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

    let provider = OpenAICompletionsProtocol::new();
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.usage.input, 70);
    assert_eq!(result.usage.output, 60);
    assert_eq!(result.usage.cache_read, 30);
    assert_eq!(result.usage.total_tokens, 160);
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

    let provider = OpenAICompletionsProtocol::new();
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    // With [DONE] sentinel present but no finish_reason, the stream is
    // treated as a normal stop (tolerate missing finish_reason).
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "");
    assert!(result.error_message.is_none());
}

#[tokio::test]
async fn test_stream_done_without_finish_reason_but_with_tool_calls_infers_tool_use() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}
                    }]
                },
                "finish_reason": null
            }]
        })
        .to_string(),
        "[DONE]",
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "use tool");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert!(result.has_tool_calls());
    assert_eq!(result.tool_calls()[0].name, "get_weather");
}

#[tokio::test]
async fn test_stream_tolerates_null_tool_calls_in_chunk_delta() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {"tool_calls": null, "content": "hello"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "hello");
    assert!(!result.has_tool_calls());
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Length);
}

// ============================================================================
// Additional coverage: content_filter, stream_simple, choice-level usage,
// reasoning alternatives, multiple tool calls, text→tool transitions
// ============================================================================

#[tokio::test]
async fn test_stream_content_filter_stop_reason() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Error);
}

#[tokio::test]
async fn test_stream_simple_delegates_correctly() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {"content": "simple text"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}
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

    let provider = OpenAICompletionsProtocol::new();
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
            thinking_display: None,
        },
    );
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "simple text");
}

#[tokio::test]
async fn test_stream_choice_level_usage() {
    let server = MockServer::start().await;

    // Usage inside the choice object (some providers do this)
    let sse_body = sse_response(vec![
        &json!({
            "choices": [{
                "index": 0,
                "delta": {"content": "ok"},
                "finish_reason": null
            }]
        })
        .to_string(),
        &json!({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 42, "completion_tokens": 17}
            }]
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.usage.input, 42);
    assert_eq!(result.usage.output, 17);
}

#[tokio::test]
async fn test_stream_reasoning_field_alternative() {
    let server = MockServer::start().await;

    // Use "reasoning" instead of "reasoning_content"
    let sse_body = sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {"reasoning": "Step 1..."}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {"content": "Answer"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "think about this");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Answer");
    assert!(result.thinking_content().contains("Step 1..."));
}

#[tokio::test]
async fn test_stream_reasoning_text_field() {
    let server = MockServer::start().await;

    // Use "reasoning_text" alternative field name
    let sse_body = sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {"reasoning_text": "Hmm..."}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {"content": "Done"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "reason about this");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.text_content(), "Done");
    assert!(result.thinking_content().contains("Hmm..."));
    let thinking = result
        .content
        .iter()
        .find_map(|block| block.as_thinking())
        .expect("thinking block should exist");
    assert_eq!(
        thinking.thinking_signature.as_deref(),
        Some("reasoning_text")
    );
}

#[tokio::test]
async fn test_stream_emits_block_end_events_and_tool_choice_payload() {
    let server = MockServer::start().await;
    let captured = Arc::new(Mutex::new(None));

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {"reasoning": "think"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {"content": "text"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":\"v\"}"}}]}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let mut context = make_context("test", "use tool");
    context.set_tools(vec![Tool::new(
        "lookup",
        "Lookup",
        json!({"type": "object", "properties": {"q": {"type": "string"}}}),
    )]);
    let mut options = make_options("key");
    options.tool_choice = Some(ToolChoice::Mode(ToolChoiceMode::Required));
    options.on_payload = Some(Arc::new({
        let captured = captured.clone();
        move |payload, _model| {
            let captured = captured.clone();
            Box::pin(async move {
                *captured.lock() = Some(payload.clone());
                Some(payload)
            })
        }
    }));

    let mut stream = provider.stream(&model, &context, options);
    let mut saw_text_end = false;
    let mut saw_thinking_end = false;
    let mut saw_tool_end = false;
    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::TextEnd { .. } => saw_text_end = true,
            AssistantMessageEvent::ThinkingEnd { .. } => saw_thinking_end = true,
            AssistantMessageEvent::ToolCallEnd { .. } => saw_tool_end = true,
            _ => {}
        }
    }
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert!(saw_text_end && saw_thinking_end && saw_tool_end);

    let payload = captured.lock().clone().expect("payload captured");
    assert_eq!(payload["tool_choice"], json!("required"));
}

#[tokio::test]
async fn test_convert_non_vision_user_image_to_placeholder() {
    let server = MockServer::start().await;
    let captured = Arc::new(Mutex::new(None));

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_response(vec![
                    &json!({
                        "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": null}]
                    })
                    .to_string(),
                    &json!({
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                    })
                    .to_string(),
                ]))
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAICompletionsProtocol::new();
    let model = Model::builder()
        .id("gpt-4o-mini")
        .name("GPT-4o Mini")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url(server.uri())
        .input(vec![InputType::Text])
        .context_window(128000)
        .max_tokens(16384)
        .build()
        .unwrap();
    let mut context = Context::new();
    context.add_message(Message::User(UserMessage::blocks(vec![
        ContentBlock::Image(ImageContent::new("abc", "image/png")),
    ])));
    let mut options = make_options("key");
    options.on_payload = Some(Arc::new({
        let captured = captured.clone();
        move |payload, _model| {
            let captured = captured.clone();
            Box::pin(async move {
                *captured.lock() = Some(payload.clone());
                Some(payload)
            })
        }
    }));

    let _ = provider.stream(&model, &context, options).result().await;
    let payload = captured.lock().clone().expect("payload captured");
    assert!(payload["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|msg| msg["content"] == json!("(image omitted: model does not support images)")));
}

#[tokio::test]
async fn test_stream_multiple_tool_calls_by_index() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![
        // First tool call start
        &json!({
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "tool_a", "arguments": ""}}]
            }, "finish_reason": null}]
        })
        .to_string(),
        // First tool call args
        &json!({
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": "{\"x\": 1}"}}]
            }, "finish_reason": null}]
        })
        .to_string(),
        // Second tool call start
        &json!({
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 1, "id": "call_2", "type": "function", "function": {"name": "tool_b", "arguments": ""}}]
            }, "finish_reason": null}]
        })
        .to_string(),
        // Second tool call args
        &json!({
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 1, "function": {"arguments": "{\"y\": 2}"}}]
            }, "finish_reason": null}]
        })
        .to_string(),
        // Finish
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let mut context = make_context("test", "use tools");
    context.set_tools(vec![
        Tool::new(
            "tool_a",
            "Tool A",
            json!({"type": "object", "properties": {"x": {"type": "integer"}}}),
        ),
        Tool::new(
            "tool_b",
            "Tool B",
            json!({"type": "object", "properties": {"y": {"type": "integer"}}}),
        ),
    ]);
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::ToolUse);
    let tcs = result.tool_calls();
    assert_eq!(tcs.len(), 2);
    assert_eq!(tcs[0].name, "tool_a");
    assert_eq!(tcs[0].id, "call_1");
    assert_eq!(tcs[1].name, "tool_b");
    assert_eq!(tcs[1].id, "call_2");
}

#[tokio::test]
async fn test_stream_text_then_tool_call_transition() {
    let server = MockServer::start().await;

    // Text followed by a tool call (text block must close before tool call starts)
    let sse_body = sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {"content": "Let me check"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "call_x", "type": "function", "function": {"name": "search", "arguments": "{\"q\": \"test\"}"}}]
            }, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10}
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let mut context = make_context("test", "search for something");
    context.set_tools(vec![Tool::new(
        "search",
        "Search",
        json!({"type": "object", "properties": {"q": {"type": "string"}}}),
    )]);
    let options = make_options("key");

    let mut stream = provider.stream(&model, &context, options);
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert_eq!(result.text_content(), "Let me check");
    assert_eq!(result.tool_calls().len(), 1);
    assert_eq!(result.tool_calls()[0].name, "search");
}

#[tokio::test]
async fn test_stream_default_provider() {
    let provider = OpenAICompletionsProtocol::default();
    assert_eq!(provider.provider_type(), Provider::OpenAI);
}

#[tokio::test]
async fn test_stream_function_call_finish_reason() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "fn1", "arguments": "{}"}}]
            }, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "function_call"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5}
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "use fn");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::ToolUse);
}

// ============================================================================
// Message conversion coverage: multi-turn with assistant/tool messages, images
// ============================================================================

#[tokio::test]
async fn test_stream_multiturn_with_tool_calls_and_results() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5}
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

    let mut ctx = Context::with_system_prompt("system");
    ctx.add_message(Message::User(UserMessage::text("hello")));
    // Previous assistant with text + tool call
    let asst = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("gpt-4o-mini")
        .content(vec![
            ContentBlock::Text(TextContent {
                text: "Let me search".to_string(),
                text_signature: None,
            }),
            ContentBlock::ToolCall(ToolCall {
                id: "tc_1".to_string(),
                name: "search".to_string(),
                arguments: json!({"q": "test"}),
                thought_signature: None,
            }),
        ])
        .stop_reason(StopReason::ToolUse)
        .build()
        .unwrap();
    ctx.add_message(Message::Assistant(asst));
    // Tool result
    ctx.add_message(Message::ToolResult(ToolResultMessage::text(
        "tc_1", "search", "found it", false,
    )));
    // Errored assistant (should be skipped)
    let asst_err = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("gpt-4o-mini")
        .content(vec![ContentBlock::Text(TextContent {
            text: "err".to_string(),
            text_signature: None,
        })])
        .stop_reason(StopReason::Error)
        .build()
        .unwrap();
    ctx.add_message(Message::Assistant(asst_err));
    // Aborted assistant (should be skipped)
    let asst_abort = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("gpt-4o-mini")
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let options = make_options("key");
    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "done");
}

#[tokio::test]
async fn test_stream_with_image_user_content() {
    let server = MockServer::start().await;

    let sse_body =
        sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {"content": "I see an image"}, "finish_reason": null}]
        }).to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 3}
        }).to_string(),
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

    let mut ctx = Context::with_system_prompt("test");
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

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let options = make_options("key");
    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "I see an image");
}

#[tokio::test]
async fn test_stream_with_thinking_in_assistant_context() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {"content": "continued"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 1}
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

    let mut ctx = Context::with_system_prompt("system");
    ctx.add_message(Message::User(UserMessage::text("hello")));
    let asst = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("gpt-4o-mini")
        .content(vec![
            ContentBlock::Thinking(ThinkingContent {
                thinking: "Let me consider...".to_string(),
                thinking_signature: Some("sig_1".to_string()),
                redacted: false,
            }),
            ContentBlock::Text(TextContent {
                text: "answer".to_string(),
                text_signature: None,
            }),
        ])
        .stop_reason(StopReason::Stop)
        .build()
        .unwrap();
    ctx.add_message(Message::Assistant(asst));
    ctx.add_message(Message::User(UserMessage::text("go on")));

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let options = make_options("key");
    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "continued");
}

#[tokio::test]
async fn test_stream_with_developer_role_compat() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
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

    // Model with reasoning=true and supports_developer_role=true
    let model = Model::builder()
        .id("o1-mini")
        .name("o1-mini")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url(&server.uri())
        .context_window(128000)
        .max_tokens(16384)
        .reasoning(true)
        .compat(OpenAICompletionsCompat {
            supports_developer_role: true,
            ..Default::default()
        })
        .build()
        .unwrap();

    let context = make_context("system prompt", "hello");
    let options = make_options("key");
    let provider = OpenAICompletionsProtocol::new();
    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
}

#[tokio::test]
async fn test_stream_openai_completions_replays_cross_provider_thinking_as_text() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
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

    let mut ctx = Context::with_system_prompt("system");
    ctx.add_message(Message::User(UserMessage::text("hello")));
    ctx.add_message(Message::Assistant(
        AssistantMessage::builder()
            .api(Api::AnthropicMessages)
            .provider(Provider::Anthropic)
            .model("claude-3-5-sonnet")
            .content(vec![
                ContentBlock::Thinking(ThinkingContent::new("Let me reason")),
                ContentBlock::Text(TextContent::new("answer")),
            ])
            .stop_reason(StopReason::Stop)
            .build()
            .unwrap(),
    ));
    ctx.add_message(Message::User(UserMessage::text("go on")));

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let captured = Arc::new(Mutex::new(None));
    let options = make_options_with_capture("key", captured.clone());
    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let payload = captured.lock().clone().expect("payload should be captured");
    let messages = payload["messages"]
        .as_array()
        .expect("messages should be an array");
    let assistant = &messages[2];

    assert_eq!(assistant["role"], json!("assistant"));
    assert_eq!(assistant["content"], json!("Let me reasonanswer"));
    assert!(assistant.get("reasoning_content").is_none());
}

#[tokio::test]
async fn test_stream_openai_completions_normalizes_composite_tool_call_ids() {
    let server = MockServer::start().await;

    let sse_body = sse_response(vec![
        &json!({
            "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": null}]
        })
        .to_string(),
        &json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
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

    let mut ctx = Context::with_system_prompt("system");
    ctx.add_message(Message::User(UserMessage::text("hello")));
    ctx.add_message(Message::Assistant(
        AssistantMessage::builder()
            .api(Api::OpenAIResponses)
            .provider(Provider::OpenAI)
            .model("gpt-4o")
            .content(vec![ContentBlock::ToolCall(ToolCall::new(
                "call_abc|fc_abc",
                "search",
                json!({"q": "test"}),
            ))])
            .stop_reason(StopReason::ToolUse)
            .build()
            .unwrap(),
    ));
    ctx.add_message(Message::ToolResult(ToolResultMessage::text(
        "call_abc|fc_abc",
        "search",
        "done",
        false,
    )));
    ctx.add_message(Message::User(UserMessage::text("continue")));

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let captured = Arc::new(Mutex::new(None));
    let options = make_options_with_capture("key", captured.clone());
    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let payload = captured.lock().clone().expect("payload should be captured");
    let messages = payload["messages"]
        .as_array()
        .expect("messages should be an array");
    let assistant = &messages[2];
    let tool_result = &messages[3];

    assert_eq!(assistant["tool_calls"][0]["id"], json!("call_abc"));
    assert_eq!(tool_result["tool_call_id"], json!("call_abc"));
}

#[tokio::test]
async fn test_stream_http_error_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("Rate limit exceeded"))
        .mount(&server)
        .await;

    let provider = OpenAICompletionsProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Error);
}

//! Tests for OpenAI Responses API provider using wiremock for HTTP mocking.

use serde_json::json;
use tiy_core::types::*;
use tiy_core::provider::LLMProvider;
use tiy_core::provider::openai_responses::OpenAIResponsesProvider;
use futures::StreamExt;
use wiremock::{MockServer, Mock, ResponseTemplate};
use wiremock::matchers::{method, path, header};

// ============================================================================
// Helper functions
// ============================================================================

fn make_model(base_url: &str) -> Model {
    Model::builder()
        .id("gpt-4o")
        .name("GPT-4o")
        .api(Api::OpenAIResponses)
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

/// Build an SSE body from a list of (event_type, data_json) pairs.
/// The OpenAI Responses API uses typed `event:` lines unlike the Completions API.
fn responses_sse(events: Vec<(&str, &str)>) -> String {
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
    let provider = OpenAIResponsesProvider::new();
    assert_eq!(provider.provider_type(), Provider::OpenAIResponses);
}

// ============================================================================
// Streaming integration tests with wiremock
// ============================================================================

#[tokio::test]
async fn test_stream_simple_text_response() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello world!"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15
                    },
                    "output": [
                        {"type": "message", "id": "item_01"}
                    ]
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let options = make_options("test-key");

    let mut stream = provider.stream(&model, &context, options);

    // Collect all streamed events
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Should have: Start, TextStart, TextDelta, TextEnd, Done
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
    assert_eq!(result.text_content(), "Hello world!");
}

#[tokio::test]
async fn test_stream_with_tool_call() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "item_02",
                    "call_id": "call_abc123",
                    "name": "get_weather",
                    "arguments": ""
                }
            })
            .to_string(),
        ),
        (
            "response.function_call_arguments.delta",
            &json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": "{\"city\":"
            })
            .to_string(),
        ),
        (
            "response.function_call_arguments.delta",
            &json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": " \"Tokyo\"}"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "item_02",
                    "call_id": "call_abc123",
                    "name": "get_weather"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 15,
                        "total_tokens": 35
                    },
                    "output": [
                        {
                            "type": "function_call",
                            "id": "item_02",
                            "call_id": "call_abc123",
                            "name": "get_weather"
                        }
                    ]
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProvider::new();
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
    // The provider creates composite IDs: "{call_id}|{item_id}"
    assert!(tool_calls[0].id.contains("call_abc123"));
    assert_eq!(tool_calls[0].arguments["city"], "Tokyo");
}

#[tokio::test]
async fn test_stream_http_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_string(r#"{"error": {"message": "Invalid API key"}}"#),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProvider::new();
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

    let sse_body = responses_sse(vec![
        // First: reasoning item added
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "item_03"
                }
            })
            .to_string(),
        ),
        // Reasoning summary text delta
        (
            "response.reasoning_summary_text.delta",
            &json!({
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "summary_index": 0,
                "delta": "Let me think"
            })
            .to_string(),
        ),
        (
            "response.reasoning_summary_text.delta",
            &json!({
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "summary_index": 0,
                "delta": " about this..."
            })
            .to_string(),
        ),
        // Reasoning item done
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "item_03"
                }
            })
            .to_string(),
        ),
        // Then: message item added
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "type": "message",
                    "id": "item_04",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 1,
                "content_index": 0,
                "delta": "The answer is 42."
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "message",
                    "id": "item_04"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "total_tokens": 30
                    },
                    "output": [
                        {"type": "reasoning", "id": "item_03"},
                        {"type": "message", "id": "item_04"}
                    ]
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "What is the meaning of life?");
    let options = make_options("test-key");

    let mut stream = provider.stream(&model, &context, options);

    // Collect all events to verify thinking events are emitted
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Verify thinking events are present
    let thinking_deltas: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AssistantMessageEvent::ThinkingDelta { .. }))
        .collect();
    assert!(!thinking_deltas.is_empty(), "Should have thinking delta events");

    // Verify thinking start/end events
    assert!(
        events.iter().any(|e| matches!(e, AssistantMessageEvent::ThinkingStart { .. })),
        "Should have ThinkingStart event"
    );
    assert!(
        events.iter().any(|e| matches!(e, AssistantMessageEvent::ThinkingEnd { .. })),
        "Should have ThinkingEnd event"
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "The answer is 42.");
    assert!(result.thinking_content().contains("Let me think about this..."));
}

#[tokio::test]
async fn test_stream_usage_tracking() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hi"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150
                    },
                    "output": [
                        {"type": "message", "id": "item_01"}
                    ]
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProvider::new();
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
async fn test_stream_incomplete_stop_reason() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "truncated"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "incomplete",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 100,
                        "total_tokens": 110
                    }
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Length);
    assert_eq!(result.text_content(), "truncated");
    assert_eq!(result.usage.input, 10);
    assert_eq!(result.usage.output, 100);
    assert_eq!(result.usage.total_tokens, 110);
}

#[tokio::test]
async fn test_stream_multiple_text_deltas() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello"
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": " "
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "world"
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "!"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 8,
                        "total_tokens": 18
                    },
                    "output": [
                        {"type": "message", "id": "item_01"}
                    ]
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProvider::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let options = make_options("test-key");

    let mut stream = provider.stream(&model, &context, options);

    // Collect all events
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Verify we got 4 text deltas
    let text_deltas: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AssistantMessageEvent::TextDelta { delta, .. } => Some(delta.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text_deltas.len(), 4);
    assert_eq!(text_deltas[0], "Hello");
    assert_eq!(text_deltas[1], " ");
    assert_eq!(text_deltas[2], "world");
    assert_eq!(text_deltas[3], "!");

    // Verify final concatenated text
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Hello world!");
}

/// Some providers/proxies skip `response.output_item.added` and start directly
/// with `response.output_text.delta`. The parser must auto-register the item
/// and still emit TextStart + TextDelta events.
#[tokio::test]
async fn test_stream_text_without_output_item_added() {
    let server = MockServer::start().await;

    // SSE stream that skips response.output_item.added entirely
    let sse_body = responses_sse(vec![
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello "
            }).to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "world!"
            }).to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": { "type": "message", "id": "item_01" }
            }).to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15
                    },
                    "output": []
                }
            }).to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let model = make_model(&server.uri());
    let context = make_context("You are a test assistant.", "Hi");
    let options = make_options("test-key");
    let provider = OpenAIResponsesProvider::new();

    let stream = provider.stream(&model, &context, options);

    // Collect all events (clone so we can still call result())
    let events: Vec<_> = stream.clone().collect().await;

    // Should have auto-generated TextStart
    let has_text_start = events.iter().any(|e| matches!(e, AssistantMessageEvent::TextStart { .. }));
    assert!(has_text_start, "Expected TextStart event from auto-registration");

    // Should have both TextDelta events
    let text_deltas: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AssistantMessageEvent::TextDelta { delta, .. } => Some(delta.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text_deltas.len(), 2);
    assert_eq!(text_deltas[0], "Hello ");
    assert_eq!(text_deltas[1], "world!");

    // Verify final result
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Hello world!");
}

/// Some proxies strip the SSE `event:` line and only forward `data:` lines.
/// The parser must extract the event type from the JSON `type` field.
#[tokio::test]
async fn test_stream_without_sse_event_lines() {
    let server = MockServer::start().await;

    // Build raw SSE body WITHOUT any "event:" lines — only "data:" lines.
    // Each data JSON has a "type" field that the parser should use.
    let sse_body = [
        format!("data: {}\n\n", json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": { "type": "message", "id": "item_01", "role": "assistant", "content": [] }
        })),
        format!("data: {}\n\n", json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello from "
        })),
        format!("data: {}\n\n", json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "data-only SSE!"
        })),
        format!("data: {}\n\n", json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": { "type": "message", "id": "item_01" }
        })),
        format!("data: {}\n\n", json!({
            "type": "response.completed",
            "response": {
                "id": "resp_01",
                "status": "completed",
                "usage": { "input_tokens": 12, "output_tokens": 8, "total_tokens": 20 },
                "output": []
            }
        })),
    ].join("");

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let model = make_model(&server.uri());
    let context = make_context("You are a test assistant.", "Hi");
    let options = make_options("test-key");
    let provider = OpenAIResponsesProvider::new();

    let stream = provider.stream(&model, &context, options);

    let events: Vec<_> = stream.clone().collect().await;

    // Should have TextStart, TextDelta events even without SSE event: lines
    let has_text_start = events.iter().any(|e| matches!(e, AssistantMessageEvent::TextStart { .. }));
    assert!(has_text_start, "Expected TextStart from data-only SSE");

    let text_deltas: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AssistantMessageEvent::TextDelta { delta, .. } => Some(delta.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text_deltas.len(), 2);
    assert_eq!(text_deltas[0], "Hello from ");
    assert_eq!(text_deltas[1], "data-only SSE!");

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Hello from data-only SSE!");
    assert_eq!(result.usage.input, 12);
    assert_eq!(result.usage.output, 8);
}

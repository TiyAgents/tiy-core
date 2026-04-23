//! Replay-focused tests for OpenAI Responses history conversion.

use parking_lot::Mutex;
use serde_json::{json, Value};
use std::sync::Arc;
use tiycore::protocol::openai_responses::OpenAIResponsesProtocol;
use tiycore::protocol::LLMProtocol;
use tiycore::types::*;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_model(base_url: &str, id: &str) -> Model {
    Model::builder()
        .id(id)
        .name(id)
        .api(Api::OpenAIResponses)
        .provider(Provider::OpenAI)
        .base_url(base_url)
        .input(vec![InputType::Text, InputType::Image])
        .context_window(128000)
        .max_tokens(16384)
        .build()
        .unwrap()
}

fn make_options(api_key: &str, captured: Option<Arc<Mutex<Option<Value>>>>) -> StreamOptions {
    let mut options = StreamOptions {
        api_key: Some(api_key.to_string()),
        ..Default::default()
    };

    if let Some(captured) = captured {
        options.on_payload = Some(Arc::new(move |payload, _model| {
            let captured = captured.clone();
            Box::pin(async move {
                *captured.lock() = Some(payload.clone());
                Some(payload)
            })
        }));
    }

    options
}

fn responses_sse(events: Vec<(&str, Value)>) -> String {
    events
        .into_iter()
        .map(|(event_type, data)| format!("event: {event_type}\ndata: {data}\n\n"))
        .collect::<String>()
}

fn success_sse() -> String {
    responses_sse(vec![
        (
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": []
                }
            }),
        ),
        (
            "response.output_text.delta",
            json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "ok"
            }),
        ),
        (
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1"
                }
            }),
        ),
        (
            "response.completed",
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12
                    },
                    "output": [
                        {"type": "message", "id": "msg_1"}
                    ]
                }
            }),
        ),
    ])
}

#[tokio::test]
async fn test_same_model_strips_server_ids_when_store_is_false() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(success_sse())
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri(), "gpt-4o");
    let captured = Arc::new(Mutex::new(None));

    let mut context = Context::new();
    context.add_message(Message::User(UserMessage::text("Use the tool.")));
    context.add_message(Message::Assistant(
        AssistantMessage::builder()
            .api(Api::OpenAIResponses)
            .provider(Provider::OpenAI)
            .model("gpt-4o")
            .content(vec![
                ContentBlock::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: Some(r#"{"type":"reasoning","id":"rs_123"}"#.to_string()),
                    redacted: false,
                }),
                ContentBlock::Text(TextContent {
                    text: "Calling the tool.".to_string(),
                    text_signature: Some(
                        r#"{"v":1,"id":"msg_sig","phase":"completed"}"#.to_string(),
                    ),
                }),
                ContentBlock::ToolCall(ToolCall::new(
                    "call_123|fc_123",
                    "double_number",
                    json!({"value": 21}),
                )),
            ])
            .stop_reason(StopReason::ToolUse)
            .build()
            .unwrap(),
    ));
    context.add_message(Message::ToolResult(ToolResultMessage::text(
        "call_123|fc_123",
        "double_number",
        "42",
        false,
    )));
    context.add_message(Message::User(UserMessage::text("What was the result?")));

    let result = provider
        .stream(
            &model,
            &context,
            make_options("test-key", Some(captured.clone())),
        )
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);

    let payload = captured.lock().clone().expect("payload should be captured");
    let input = payload["input"]
        .as_array()
        .expect("input should be an array");

    assert!(
        !input.iter().any(|item| {
            item.get("type") == Some(&json!("reasoning"))
                && item.get("id") == Some(&json!("rs_123"))
        }),
        "reasoning item should be stripped when store=false"
    );

    let message_item = input
        .iter()
        .find(|item| {
            item.get("type") == Some(&json!("message"))
                && item.get("role") == Some(&json!("assistant"))
        })
        .expect("assistant message should exist");
    assert!(message_item.get("id").is_none());
    assert!(message_item.get("phase").is_none());

    let function_call = input
        .iter()
        .find(|item| item.get("type") == Some(&json!("function_call")))
        .expect("function_call item should exist");
    assert!(function_call.get("id").is_none());
    assert_eq!(function_call.get("call_id"), Some(&json!("call_123")));

    let function_call_output = input
        .iter()
        .find(|item| item.get("type") == Some(&json!("function_call_output")))
        .expect("function_call_output item should exist");
    assert_eq!(
        function_call_output.get("call_id"),
        function_call.get("call_id")
    );
}

#[tokio::test]
async fn test_different_model_omits_openai_function_item_id() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(success_sse())
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri(), "gpt-5.2-codex");
    let captured = Arc::new(Mutex::new(None));

    let mut context = Context::new();
    context.add_message(Message::User(UserMessage::text("Use the tool.")));
    context.add_message(Message::Assistant(
        AssistantMessage::builder()
            .api(Api::OpenAIResponses)
            .provider(Provider::OpenAI)
            .model("gpt-5-mini")
            .content(vec![
                ContentBlock::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: Some(r#"{"type":"reasoning","id":"rs_abc"}"#.to_string()),
                    redacted: false,
                }),
                ContentBlock::Text(TextContent::new("Calling the tool.")),
                ContentBlock::ToolCall(ToolCall::new(
                    "call_abc|fc_abc",
                    "double_number",
                    json!({"value": 21}),
                )),
            ])
            .stop_reason(StopReason::ToolUse)
            .build()
            .unwrap(),
    ));
    context.add_message(Message::ToolResult(ToolResultMessage::text(
        "call_abc|fc_abc",
        "double_number",
        "42",
        false,
    )));
    context.add_message(Message::User(UserMessage::text("What was the result?")));

    let result = provider
        .stream(
            &model,
            &context,
            make_options("test-key", Some(captured.clone())),
        )
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);

    let payload = captured.lock().clone().expect("payload should be captured");
    let input = payload["input"]
        .as_array()
        .expect("input should be an array");

    assert_eq!(
        input
            .iter()
            .filter(|item| item.get("type") == Some(&json!("reasoning")))
            .count(),
        0
    );

    let function_call = input
        .iter()
        .find(|item| item.get("type") == Some(&json!("function_call")))
        .expect("function_call item should exist");
    assert!(function_call.get("id").is_none());
    assert_eq!(function_call.get("call_id"), Some(&json!("call_abc")));

    let assistant_message = input
        .iter()
        .find(|item| {
            item.get("type") == Some(&json!("message"))
                && item.get("role") == Some(&json!("assistant"))
        })
        .expect("assistant message should exist");
    assert!(assistant_message.get("id").is_none());
}

#[tokio::test]
async fn test_store_false_replay_hashes_long_responses_call_id_consistently() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(success_sse())
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri(), "gpt-4o");
    let captured = Arc::new(Mutex::new(None));

    let long_call_id = "helper_agentid_super_long_tool_call_id_1234567890_extra";
    let composite_id = format!("{}|fc_helper_generated_item_identifier", long_call_id);

    let mut context = Context::new();
    context.add_message(Message::User(UserMessage::text("Use the tool.")));
    context.add_message(Message::Assistant(
        AssistantMessage::builder()
            .api(Api::OpenAIResponses)
            .provider(Provider::OpenAI)
            .model("gpt-4o")
            .content(vec![ContentBlock::ToolCall(ToolCall::new(
                &composite_id,
                "double_number",
                json!({"value": 21}),
            ))])
            .stop_reason(StopReason::ToolUse)
            .build()
            .unwrap(),
    ));
    context.add_message(Message::ToolResult(ToolResultMessage::text(
        &composite_id,
        "double_number",
        "42",
        false,
    )));
    context.add_message(Message::User(UserMessage::text("What was the result?")));

    let result = provider
        .stream(
            &model,
            &context,
            make_options("test-key", Some(captured.clone())),
        )
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);

    let payload = captured.lock().clone().expect("payload should be captured");
    let input = payload["input"]
        .as_array()
        .expect("input should be an array");

    let function_call = input
        .iter()
        .find(|item| item.get("type") == Some(&json!("function_call")))
        .expect("function_call item should exist");
    let function_call_output = input
        .iter()
        .find(|item| item.get("type") == Some(&json!("function_call_output")))
        .expect("function_call_output item should exist");

    let replayed_call_id = function_call["call_id"]
        .as_str()
        .expect("function_call.call_id should be string");
    assert!(replayed_call_id.starts_with("call_"));
    assert!(replayed_call_id.len() <= 40);
    assert_eq!(
        function_call_output["call_id"].as_str(),
        Some(replayed_call_id)
    );
    assert!(function_call.get("id").is_none());
}

#[tokio::test]
async fn test_stream_captures_reasoning_signature_from_done_item() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs_123"
                }
            }),
        ),
        (
            "response.reasoning_summary_text.delta",
            json!({
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "delta": "Step by step"
            }),
        ),
        (
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs_123",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "Step by step"
                        }
                    ]
                }
            }),
        ),
        (
            "response.completed",
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_reasoning",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15
                    },
                    "output": [
                        {"type": "reasoning", "id": "rs_123"}
                    ]
                }
            }),
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

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri(), "gpt-4o");
    let context = {
        let mut context = Context::new();
        context.add_message(Message::User(UserMessage::text("Think about it.")));
        context
    };

    let result = provider
        .stream(&model, &context, make_options("test-key", None))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);

    let thinking = result
        .content
        .iter()
        .find_map(|block| block.as_thinking())
        .expect("thinking block should exist");
    assert_eq!(thinking.thinking, "Step by step");

    let signature = thinking
        .thinking_signature
        .as_ref()
        .expect("thinking signature should be captured");
    let signature_json: Value =
        serde_json::from_str(signature).expect("thinking signature should be valid json");
    assert_eq!(
        signature_json,
        json!({
            "type": "reasoning",
            "id": "rs_123",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "Step by step"
                }
            ]
        })
    );
}

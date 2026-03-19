//! Tests for transform module: messages and tool_calls.

use serde_json::json;
use tiy_core::transform::{normalize_tool_call_id, transform_messages, ToolCallIdMapper};
use tiy_core::types::*;

// ============================================================================
// Helper functions
// ============================================================================

fn make_model(provider: Provider, api: Api, id: &str) -> Model {
    Model::builder()
        .id(id)
        .name(id)
        .api(api)
        .provider(provider)
        .base_url("http://test")
        .context_window(128000)
        .max_tokens(16384)
        .build()
        .unwrap()
}

fn make_assistant_msg(
    provider: Provider,
    api: Api,
    model_id: &str,
    content: Vec<ContentBlock>,
    stop_reason: StopReason,
) -> AssistantMessage {
    AssistantMessage {
        role: Role::Assistant,
        content,
        api,
        provider,
        model: model_id.to_string(),
        usage: Usage::default(),
        stop_reason,
        error_message: None,
        response_id: None,
        timestamp: chrono::Utc::now().timestamp_millis(),
    }
}

// ============================================================================
// transform_messages tests
// ============================================================================

#[test]
fn test_transform_messages_skips_error_messages() {
    let target = make_model(Provider::OpenAI, Api::OpenAICompletions, "gpt-4o");

    let messages = vec![
        Message::User(UserMessage::text("Hello")),
        Message::Assistant(make_assistant_msg(
            Provider::OpenAI,
            Api::OpenAICompletions,
            "gpt-4o",
            vec![ContentBlock::Text(TextContent::new("Error occurred"))],
            StopReason::Error,
        )),
        Message::User(UserMessage::text("Try again")),
    ];

    let result = transform_messages(&messages, &target, None);
    // Error assistant message should be skipped
    assert_eq!(result.len(), 2);
    assert!(result[0].is_user());
    assert!(result[1].is_user());
}

#[test]
fn test_transform_messages_skips_aborted_messages() {
    let target = make_model(Provider::OpenAI, Api::OpenAICompletions, "gpt-4o");

    let messages = vec![
        Message::User(UserMessage::text("Hello")),
        Message::Assistant(make_assistant_msg(
            Provider::OpenAI,
            Api::OpenAICompletions,
            "gpt-4o",
            vec![ContentBlock::Text(TextContent::new("Partial..."))],
            StopReason::Aborted,
        )),
    ];

    let result = transform_messages(&messages, &target, None);
    assert_eq!(result.len(), 1);
    assert!(result[0].is_user());
}

#[test]
fn test_transform_messages_keeps_valid_messages() {
    let target = make_model(Provider::OpenAI, Api::OpenAICompletions, "gpt-4o");

    let messages = vec![
        Message::User(UserMessage::text("Hello")),
        Message::Assistant(make_assistant_msg(
            Provider::OpenAI,
            Api::OpenAICompletions,
            "gpt-4o",
            vec![ContentBlock::Text(TextContent::new("Hi there!"))],
            StopReason::Stop,
        )),
    ];

    let result = transform_messages(&messages, &target, None);
    assert_eq!(result.len(), 2);
    assert!(result[0].is_user());
    assert!(result[1].is_assistant());
}

#[test]
fn test_transform_thinking_same_model_preserved() {
    let target = make_model(
        Provider::Anthropic,
        Api::AnthropicMessages,
        "claude-sonnet-4",
    );

    let messages = vec![
        Message::User(UserMessage::text("Hello")),
        Message::Assistant(make_assistant_msg(
            Provider::Anthropic,
            Api::AnthropicMessages,
            "claude-sonnet-4",
            vec![
                ContentBlock::Thinking(ThinkingContent {
                    thinking: "Let me think...".to_string(),
                    thinking_signature: Some("sig123".to_string()),
                    redacted: false,
                }),
                ContentBlock::Text(TextContent::new("Answer")),
            ],
            StopReason::Stop,
        )),
    ];

    let result = transform_messages(&messages, &target, None);
    assert_eq!(result.len(), 2);
    if let Message::Assistant(ref a) = result[1] {
        // Thinking block should be preserved for same model
        assert!(a.content[0].is_thinking());
        assert_eq!(
            a.content[0].as_thinking().unwrap().thinking,
            "Let me think..."
        );
    } else {
        panic!("Expected assistant message");
    }
}

#[test]
fn test_transform_thinking_cross_model_converts_to_text() {
    // Source is Anthropic, target is OpenAI — thinking should become text
    let target = make_model(Provider::OpenAI, Api::OpenAICompletions, "gpt-4o");

    let messages = vec![
        Message::User(UserMessage::text("Hello")),
        Message::Assistant(make_assistant_msg(
            Provider::Anthropic,
            Api::AnthropicMessages,
            "claude-sonnet-4",
            vec![
                ContentBlock::Thinking(ThinkingContent::new("Deep thought here")),
                ContentBlock::Text(TextContent::new("Answer")),
            ],
            StopReason::Stop,
        )),
    ];

    let result = transform_messages(&messages, &target, None);
    assert_eq!(result.len(), 2);
    if let Message::Assistant(ref a) = result[1] {
        // Thinking should be converted to plain text without tags
        assert!(a.content[0].is_text());
        let text = a.content[0].as_text().unwrap().text.as_str();
        assert_eq!(text, "Deep thought here");
    } else {
        panic!("Expected assistant message");
    }
}

#[test]
fn test_transform_empty_thinking_is_dropped() {
    let target = make_model(Provider::OpenAI, Api::OpenAICompletions, "gpt-4o");

    let messages = vec![
        Message::User(UserMessage::text("Hello")),
        Message::Assistant(make_assistant_msg(
            Provider::Anthropic,
            Api::AnthropicMessages,
            "claude-sonnet-4",
            vec![
                ContentBlock::Thinking(ThinkingContent::new("   ")),
                ContentBlock::Text(TextContent::new("Answer")),
            ],
            StopReason::Stop,
        )),
    ];

    let result = transform_messages(&messages, &target, None);
    if let Message::Assistant(ref a) = result[1] {
        assert_eq!(a.content.len(), 1);
        assert!(a.content[0].is_text());
        assert_eq!(a.content[0].as_text().unwrap().text, "Answer");
    }
}

#[test]
fn test_transform_redacted_thinking_cross_model_is_dropped() {
    let target = make_model(Provider::OpenAI, Api::OpenAICompletions, "gpt-4o");

    let messages = vec![
        Message::User(UserMessage::text("Hello")),
        Message::Assistant(make_assistant_msg(
            Provider::Anthropic,
            Api::AnthropicMessages,
            "claude-sonnet-4",
            vec![
                ContentBlock::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: Some("opaque".to_string()),
                    redacted: true,
                }),
                ContentBlock::Text(TextContent::new("Answer")),
            ],
            StopReason::Stop,
        )),
    ];

    let result = transform_messages(&messages, &target, None);
    if let Message::Assistant(ref a) = result[1] {
        assert_eq!(a.content.len(), 1);
        assert!(a.content[0].is_text());
        assert_eq!(a.content[0].as_text().unwrap().text, "Answer");
    } else {
        panic!("Expected assistant message");
    }
}

#[test]
fn test_transform_tool_call_drops_thought_signature_cross_model() {
    let target = make_model(Provider::OpenAI, Api::OpenAICompletions, "gpt-4o");

    let messages = vec![
        Message::User(UserMessage::text("Hello")),
        Message::Assistant(make_assistant_msg(
            Provider::Google,
            Api::GoogleGenerativeAi,
            "gemini-2.0-flash",
            vec![ContentBlock::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "search".to_string(),
                arguments: json!({"q": "test"}),
                thought_signature: Some("sig_1".to_string()),
            })],
            StopReason::ToolUse,
        )),
    ];

    let result = transform_messages(&messages, &target, None);
    if let Message::Assistant(ref a) = result[1] {
        let tool_call = a.content[0].as_tool_call().expect("expected tool call");
        assert!(tool_call.thought_signature.is_none());
    } else {
        panic!("Expected assistant message");
    }
}

#[test]
fn test_transform_orphan_tool_calls_get_synthetic_results() {
    let target = make_model(Provider::OpenAI, Api::OpenAICompletions, "gpt-4o");

    // Assistant made a tool call but no result was provided
    let messages = vec![
        Message::User(UserMessage::text("Get weather")),
        Message::Assistant(make_assistant_msg(
            Provider::OpenAI,
            Api::OpenAICompletions,
            "gpt-4o",
            vec![ContentBlock::ToolCall(ToolCall::new(
                "call_1",
                "get_weather",
                json!({"city": "Tokyo"}),
            ))],
            StopReason::ToolUse,
        )),
        // No ToolResult for call_1!
        Message::User(UserMessage::text("Never mind")),
    ];

    let result = transform_messages(&messages, &target, None);
    // Should have: user, assistant, synthetic error tool result, user
    assert_eq!(result.len(), 4);
    assert!(result[0].is_user());
    assert!(result[1].is_assistant());
    assert!(result[2].is_tool_result()); // synthetic
    assert!(result[3].is_user());

    if let Message::ToolResult(ref tr) = result[2] {
        assert!(tr.is_error);
        assert_eq!(tr.tool_call_id, "call_1");
    }
}

#[test]
fn test_transform_matched_tool_calls_no_synthetic() {
    let target = make_model(Provider::OpenAI, Api::OpenAICompletions, "gpt-4o");

    let messages = vec![
        Message::User(UserMessage::text("Get weather")),
        Message::Assistant(make_assistant_msg(
            Provider::OpenAI,
            Api::OpenAICompletions,
            "gpt-4o",
            vec![ContentBlock::ToolCall(ToolCall::new(
                "call_1",
                "get_weather",
                json!({"city": "Tokyo"}),
            ))],
            StopReason::ToolUse,
        )),
        Message::ToolResult(ToolResultMessage::text(
            "call_1",
            "get_weather",
            "Sunny 25C",
            false,
        )),
    ];

    let result = transform_messages(&messages, &target, None);
    assert_eq!(result.len(), 3); // No synthetic results needed
}

#[test]
fn test_transform_with_tool_call_id_normalization() {
    let target = make_model(
        Provider::Anthropic,
        Api::AnthropicMessages,
        "claude-sonnet-4",
    );

    fn normalize(id: &str) -> String {
        id.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .take(64)
            .collect()
    }

    let messages = vec![
        Message::User(UserMessage::text("Hi")),
        Message::Assistant(make_assistant_msg(
            Provider::OpenAI,
            Api::OpenAICompletions,
            "gpt-4o",
            vec![ContentBlock::ToolCall(ToolCall::new(
                "call+special/chars",
                "tool",
                json!({}),
            ))],
            StopReason::ToolUse,
        )),
        Message::ToolResult(ToolResultMessage::text(
            "call+special/chars",
            "tool",
            "result",
            false,
        )),
    ];

    let result = transform_messages(&messages, &target, Some(&normalize));

    // Tool call ID should be normalized
    if let Message::Assistant(ref a) = result[1] {
        let tc = a.content[0].as_tool_call().unwrap();
        assert!(!tc.id.contains('+'));
        assert!(!tc.id.contains('/'));
    }

    // Tool result ID should also be normalized
    if let Message::ToolResult(ref tr) = result[2] {
        assert!(!tr.tool_call_id.contains('+'));
        assert!(!tr.tool_call_id.contains('/'));
    }
}

// ============================================================================
// normalize_tool_call_id tests
// ============================================================================

#[test]
fn test_normalize_for_anthropic_basic() {
    let id = "call_abc123";
    let normalized = normalize_tool_call_id(id, &Provider::Anthropic);
    assert_eq!(normalized, "call_abc123");
}

#[test]
fn test_normalize_for_anthropic_special_chars() {
    let id = "call_abc+def/ghi=jkl";
    let normalized = normalize_tool_call_id(id, &Provider::Anthropic);
    assert!(normalized
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'));
    assert_eq!(normalized, "call_abc_def_ghi_jkl");
}

#[test]
fn test_normalize_for_anthropic_max_length() {
    let id = "a".repeat(100);
    let normalized = normalize_tool_call_id(&id, &Provider::Anthropic);
    assert!(normalized.len() <= 64);
}

#[test]
fn test_normalize_for_anthropic_pipe_separated() {
    let id = "call_123|very_long_reasoning_item_suffix";
    let normalized = normalize_tool_call_id(id, &Provider::Anthropic);
    assert!(!normalized.contains('|'));
    assert_eq!(normalized, "call_123");
}

#[test]
fn test_normalize_for_openai_truncation() {
    let id = "a".repeat(50);
    let normalized = normalize_tool_call_id(&id, &Provider::OpenAI);
    assert_eq!(normalized.len(), 40);
}

#[test]
fn test_normalize_for_openai_short_id() {
    let id = "call_abc";
    let normalized = normalize_tool_call_id(id, &Provider::OpenAI);
    assert_eq!(normalized, "call_abc");
}

#[test]
fn test_normalize_for_groq_same_as_openai() {
    let id = "a".repeat(50);
    let normalized = normalize_tool_call_id(&id, &Provider::Groq);
    assert_eq!(normalized.len(), 40);
}

#[test]
fn test_normalize_for_other_providers_passthrough() {
    let id = "call+special/chars|and|pipes";
    let normalized = normalize_tool_call_id(id, &Provider::Google);
    assert_eq!(normalized, id); // No transformation
}

// ============================================================================
// ToolCallIdMapper tests
// ============================================================================

#[test]
fn test_mapper_basic() {
    let mut mapper = ToolCallIdMapper::new(Provider::Anthropic);
    let normalized = mapper.normalize("call_abc123");
    assert_eq!(normalized, "call_abc123");
}

#[test]
fn test_mapper_caching() {
    let mut mapper = ToolCallIdMapper::new(Provider::Anthropic);
    let first = mapper.normalize("call_abc123");
    let second = mapper.normalize("call_abc123");
    assert_eq!(first, second);
}

#[test]
fn test_mapper_collision_handling() {
    let mut mapper = ToolCallIdMapper::new(Provider::Anthropic);
    // Two different IDs that normalize to the same value
    let id1 = "call+abc";
    let id2 = "call/abc";

    let n1 = mapper.normalize(id1);
    let n2 = mapper.normalize(id2);

    // They should NOT be the same due to collision handling
    assert_ne!(n1, n2);
    assert_eq!(n1, "call_abc");
    assert!(n2.starts_with("call_abc_"));
}

#[test]
fn test_mapper_denormalize() {
    let mut mapper = ToolCallIdMapper::new(Provider::Anthropic);
    mapper.normalize("call+abc");
    let original = mapper.denormalize("call_abc");
    assert_eq!(original, Some(&"call+abc".to_string()));
}

#[test]
fn test_mapper_denormalize_missing() {
    let mapper = ToolCallIdMapper::new(Provider::Anthropic);
    assert!(mapper.denormalize("nonexistent").is_none());
}

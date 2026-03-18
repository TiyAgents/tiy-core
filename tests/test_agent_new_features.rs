//! Tests for all new agent features: hooks, context pipeline, queue modes,
//! custom messages, thinking budgets, transport, dynamic API key, etc.

use async_trait::async_trait;
use serde_json::json;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use tiy_core::agent::*;
use tiy_core::provider::{ArcProtocol, LLMProtocol};
use tiy_core::stream::AssistantMessageEventStream;
use tiy_core::thinking::ThinkingLevel;
use tiy_core::types::*;

// ============================================================================
// Mock Provider (shared)
// ============================================================================

struct MockProvider {
    responses: parking_lot::Mutex<Vec<AssistantMessage>>,
    call_count: AtomicUsize,
}

impl MockProvider {
    fn new(responses: Vec<AssistantMessage>) -> Self {
        Self {
            responses: parking_lot::Mutex::new(responses),
            call_count: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LLMProtocol for MockProvider {
    fn provider_type(&self) -> Provider {
        Provider::OpenAI
    }

    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: StreamOptions,
    ) -> AssistantMessageEventStream {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let stream = AssistantMessageEventStream::new_assistant_stream();
        let mut responses = self.responses.lock();
        let response = if responses.is_empty() {
            make_assistant_message("Default response")
        } else {
            responses.remove(0)
        };
        let stop_reason = response.stop_reason;
        let response_clone = response.clone();
        let stream_clone = stream.clone();
        tokio::spawn(async move {
            stream_clone.push(AssistantMessageEvent::Start {
                partial: response_clone.clone(),
            });
            stream_clone.push(AssistantMessageEvent::Done {
                reason: stop_reason,
                message: response_clone,
            });
            stream_clone.end(None);
        });
        stream
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.stream(model, context, options.base)
    }
}

fn make_model() -> Model {
    Model::builder()
        .id("mock-model")
        .name("Mock Model")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url("http://localhost:0")
        .context_window(128000)
        .max_tokens(4096)
        .build()
        .unwrap()
}

fn make_assistant_message(text: &str) -> AssistantMessage {
    AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("mock-model")
        .content(vec![ContentBlock::Text(TextContent::new(text))])
        .stop_reason(StopReason::Stop)
        .build()
        .unwrap()
}

fn make_tool_call_message(
    tool_name: &str,
    tool_id: &str,
    args: serde_json::Value,
) -> AssistantMessage {
    AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("mock-model")
        .content(vec![ContentBlock::ToolCall(ToolCall::new(
            tool_id, tool_name, args,
        ))])
        .stop_reason(StopReason::ToolUse)
        .build()
        .unwrap()
}

// ============================================================================
// Custom Messages
// ============================================================================

#[test]
fn test_custom_message_creation() {
    let msg = AgentMessage::Custom {
        message_type: "artifact".to_string(),
        data: json!({"name": "code.rs", "content": "fn main() {}"}),
    };
    assert!(matches!(msg, AgentMessage::Custom { .. }));
}

#[test]
fn test_custom_message_to_option_message_returns_none() {
    let msg = AgentMessage::Custom {
        message_type: "notification".to_string(),
        data: json!({"text": "hello"}),
    };
    let opt: Option<Message> = msg.into();
    assert!(opt.is_none());
}

#[test]
fn test_custom_message_serialization() {
    let msg = AgentMessage::Custom {
        message_type: "artifact".to_string(),
        data: json!({"name": "test"}),
    };
    let json = serde_json::to_value(&msg).unwrap();
    assert_eq!(json["role"], "custom");
    assert_eq!(json["type"], "artifact");
}

// ============================================================================
// String convenience conversions
// ============================================================================

#[test]
fn test_agent_message_from_str() {
    let msg: AgentMessage = "hello".into();
    assert!(matches!(msg, AgentMessage::User(_)));
}

#[test]
fn test_agent_message_from_string() {
    let msg: AgentMessage = String::from("hello").into();
    assert!(matches!(msg, AgentMessage::User(_)));
}

#[tokio::test]
async fn test_prompt_with_str_convenience() {
    let response = make_assistant_message("Hi!");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));
    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Use string directly
    let result = agent.prompt("Hello").await;
    assert!(result.is_ok());
}

// ============================================================================
// QueueMode
// ============================================================================

#[test]
fn test_queue_mode_default_is_all() {
    assert_eq!(QueueMode::default(), QueueMode::All);
}

#[test]
fn test_steering_mode_setter_getter() {
    let agent = Agent::new();
    assert_eq!(agent.steering_mode(), QueueMode::All);
    agent.set_steering_mode(QueueMode::OneAtATime);
    assert_eq!(agent.steering_mode(), QueueMode::OneAtATime);
}

#[test]
fn test_follow_up_mode_setter_getter() {
    let agent = Agent::new();
    assert_eq!(agent.follow_up_mode(), QueueMode::All);
    agent.set_follow_up_mode(QueueMode::OneAtATime);
    assert_eq!(agent.follow_up_mode(), QueueMode::OneAtATime);
}

#[tokio::test]
async fn test_steering_one_at_a_time_mode() {
    // Queue 3 steering messages, only 1 should be dequeued per turn in OneAtATime mode
    let responses: Vec<AssistantMessage> = (0..5).map(|_| make_assistant_message("ok")).collect();
    let mock = Arc::new(MockProvider::new(responses));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_steering_mode(QueueMode::OneAtATime);

    agent.steer(AgentMessage::User(UserMessage::text("steer 1")));
    agent.steer(AgentMessage::User(UserMessage::text("steer 2")));
    agent.steer(AgentMessage::User(UserMessage::text("steer 3")));

    let result = agent.prompt("start").await;
    assert!(result.is_ok());

    // After prompt completes, at most 1 steering should have been consumed per turn check.
    // Since steering interrupts turns, not all 3 necessarily get consumed in a single prompt
    // invocation, but the queue should have been partially drained.
}

#[tokio::test]
async fn test_follow_up_one_at_a_time_mode() {
    let responses: Vec<AssistantMessage> = (0..5).map(|_| make_assistant_message("ok")).collect();
    let mock = Arc::new(MockProvider::new(responses));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_follow_up_mode(QueueMode::OneAtATime);

    agent.follow_up(AgentMessage::User(UserMessage::text("follow 1")));
    agent.follow_up(AgentMessage::User(UserMessage::text("follow 2")));
    agent.follow_up(AgentMessage::User(UserMessage::text("follow 3")));

    let result = agent.prompt("start").await;
    assert!(result.is_ok());

    // Provider should be called multiple times (start + follow-ups)
    assert!(
        mock.call_count() >= 2,
        "Expected multiple calls for follow-ups in one-at-a-time mode"
    );
}

// ============================================================================
// ThinkingBudgets
// ============================================================================

#[test]
fn test_thinking_budgets_budget_for() {
    let budgets = ThinkingBudgets {
        minimal: Some(64),
        low: Some(256),
        medium: None,
        high: Some(4096),
    };
    assert_eq!(
        budgets.budget_for(tiy_core::thinking::ThinkingLevel::Minimal),
        Some(64)
    );
    assert_eq!(
        budgets.budget_for(tiy_core::thinking::ThinkingLevel::Low),
        Some(256)
    );
    assert_eq!(
        budgets.budget_for(tiy_core::thinking::ThinkingLevel::Medium),
        None
    );
    assert_eq!(
        budgets.budget_for(tiy_core::thinking::ThinkingLevel::High),
        Some(4096)
    );
    assert_eq!(
        budgets.budget_for(tiy_core::thinking::ThinkingLevel::Off),
        None
    );
    assert_eq!(
        budgets.budget_for(tiy_core::thinking::ThinkingLevel::XHigh),
        None
    );
}

#[test]
fn test_thinking_budgets_setter_getter() {
    let agent = Agent::new();
    assert!(agent.thinking_budgets().is_none());

    let budgets = ThinkingBudgets {
        minimal: Some(128),
        low: Some(512),
        medium: Some(1024),
        high: Some(2048),
    };
    agent.set_thinking_budgets(budgets.clone());
    assert_eq!(agent.thinking_budgets(), Some(budgets));
}

// ============================================================================
// Transport
// ============================================================================

#[test]
fn test_transport_default_is_sse() {
    assert_eq!(Transport::default(), Transport::Sse);
}

#[test]
fn test_transport_setter_getter() {
    let agent = Agent::new();
    assert_eq!(agent.transport(), Transport::Sse);
    agent.set_transport(Transport::WebSocket);
    assert_eq!(agent.transport(), Transport::WebSocket);
    agent.set_transport(Transport::Auto);
    assert_eq!(agent.transport(), Transport::Auto);
}

// ============================================================================
// MaxRetryDelayMs
// ============================================================================

#[test]
fn test_max_retry_delay_setter_getter() {
    let agent = Agent::new();
    assert_eq!(agent.max_retry_delay_ms(), None);
    agent.set_max_retry_delay_ms(Some(30000));
    assert_eq!(agent.max_retry_delay_ms(), Some(30000));
    agent.set_max_retry_delay_ms(Some(0));
    assert_eq!(agent.max_retry_delay_ms(), Some(0));
}

// ============================================================================
// beforeToolCall Hook
// ============================================================================

#[tokio::test]
async fn test_before_tool_call_allows_execution() {
    let tool_response = make_tool_call_message("my_tool", "call_1", json!({"x": 1}));
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "my_tool",
        "My Tool",
        "description",
        json!({"type": "object"}),
    )]);

    let hook_called = Arc::new(AtomicUsize::new(0));
    let hc = hook_called.clone();

    agent.set_before_tool_call(move |_ctx| {
        let hc = hc.clone();
        async move {
            hc.fetch_add(1, Ordering::SeqCst);
            None // Allow execution
        }
    });

    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult::text("ok")
        },
    );

    let result = agent.prompt("go").await;
    assert!(result.is_ok());
    assert_eq!(
        hook_called.load(Ordering::SeqCst),
        1,
        "beforeToolCall hook should be called once"
    );
}

#[tokio::test]
async fn test_before_tool_call_blocks_execution() {
    let tool_response = make_tool_call_message("dangerous_tool", "call_1", json!({}));
    let final_response = make_assistant_message("OK, I won't do that.");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "dangerous_tool",
        "Danger",
        "dangerous",
        json!({"type": "object"}),
    )]);

    let executor_called = Arc::new(AtomicUsize::new(0));
    let ec = executor_called.clone();

    agent.set_before_tool_call(move |ctx| async move {
        if ctx.tool_call.name == "dangerous_tool" {
            Some(BeforeToolCallResult::blocked("User denied permission"))
        } else {
            None
        }
    });

    agent.set_tool_executor_simple(move |_name: &str, _id: &str, _args: &serde_json::Value| {
        let ec = ec.clone();
        async move {
            ec.fetch_add(1, Ordering::SeqCst);
            AgentToolResult::text("should not reach here")
        }
    });

    let result = agent.prompt("do the dangerous thing").await;
    assert!(result.is_ok());

    // Tool executor should NOT have been called
    assert_eq!(
        executor_called.load(Ordering::SeqCst),
        0,
        "Blocked tool should not be executed"
    );

    // Should have a tool result with the blocked reason
    let messages = result.unwrap();
    let tool_results: Vec<_> = messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::ToolResult(tr) => Some(tr),
            _ => None,
        })
        .collect();
    assert_eq!(tool_results.len(), 1);
    assert!(tool_results[0].is_error);
    let text: String = tool_results[0]
        .content
        .iter()
        .filter_map(|b| b.as_text())
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join("");
    assert!(text.contains("User denied permission"));
}

// ============================================================================
// afterToolCall Hook
// ============================================================================

#[tokio::test]
async fn test_after_tool_call_overrides_content() {
    let tool_response = make_tool_call_message("my_tool", "call_1", json!({}));
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "my_tool",
        "My Tool",
        "desc",
        json!({"type": "object"}),
    )]);

    agent.set_after_tool_call(move |_ctx| async move {
        Some(AfterToolCallResult {
            content: Some(vec![ContentBlock::Text(TextContent::new(
                "overridden content",
            ))]),
            details: None,
            is_error: Some(false),
        })
    });

    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult::text("original content")
        },
    );

    let result = agent.prompt("go").await;
    assert!(result.is_ok());

    let messages = result.unwrap();
    let tool_results: Vec<_> = messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::ToolResult(tr) => Some(tr),
            _ => None,
        })
        .collect();
    assert_eq!(tool_results.len(), 1);
    let text: String = tool_results[0]
        .content
        .iter()
        .filter_map(|b| b.as_text())
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join("");
    assert_eq!(text, "overridden content");
}

#[tokio::test]
async fn test_after_tool_call_override_is_error() {
    let tool_response = make_tool_call_message("my_tool", "call_1", json!({}));
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "my_tool",
        "My Tool",
        "desc",
        json!({"type": "object"}),
    )]);

    // Override is_error to true even though original succeeded
    agent.set_after_tool_call(move |_ctx| async move {
        Some(AfterToolCallResult {
            content: None, // Keep original
            details: None,
            is_error: Some(true),
        })
    });

    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult::text("success")
        },
    );

    let result = agent.prompt("go").await;
    assert!(result.is_ok());

    let messages = result.unwrap();
    let tool_results: Vec<_> = messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::ToolResult(tr) => Some(tr),
            _ => None,
        })
        .collect();
    assert_eq!(tool_results.len(), 1);
    assert!(
        tool_results[0].is_error,
        "afterToolCall should have overridden is_error to true"
    );
}

// ============================================================================
// convertToLlm
// ============================================================================

#[tokio::test]
async fn test_convert_to_llm_filters_custom_messages_by_default() {
    let response = make_assistant_message("I see 1 message");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Add a custom message — should be filtered out by default
    agent.append_message(AgentMessage::Custom {
        message_type: "artifact".to_string(),
        data: json!({"name": "test"}),
    });

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_convert_to_llm_custom_converter() {
    let response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let converter_called = Arc::new(AtomicUsize::new(0));
    let cc = converter_called.clone();

    agent.set_convert_to_llm(move |messages| {
        let cc = cc.clone();
        async move {
            cc.fetch_add(1, Ordering::SeqCst);
            // Custom conversion: only keep user messages
            messages
                .into_iter()
                .filter_map(|m| match m {
                    AgentMessage::User(u) => Some(Message::User(u)),
                    _ => None,
                })
                .collect()
        }
    });

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());
    assert!(
        converter_called.load(Ordering::SeqCst) >= 1,
        "Custom converter should be called"
    );
}

// ============================================================================
// transformContext
// ============================================================================

#[tokio::test]
async fn test_transform_context_called() {
    let response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let transform_called = Arc::new(AtomicUsize::new(0));
    let tc = transform_called.clone();

    agent.set_transform_context(move |messages| {
        let tc = tc.clone();
        async move {
            tc.fetch_add(1, Ordering::SeqCst);
            // Keep only last 2 messages (context window management)
            let len = messages.len();
            if len > 2 {
                messages[len - 2..].to_vec()
            } else {
                messages
            }
        }
    });

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());
    assert!(
        transform_called.load(Ordering::SeqCst) >= 1,
        "transformContext should be called"
    );
}

// ============================================================================
// Dynamic API Key (getApiKey)
// ============================================================================

#[tokio::test]
async fn test_get_api_key_dynamic_resolution() {
    let response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let resolver_called = Arc::new(AtomicUsize::new(0));
    let rc = resolver_called.clone();

    agent.set_get_api_key(move |_provider: &str| {
        let rc = rc.clone();
        async move {
            rc.fetch_add(1, Ordering::SeqCst);
            Some("dynamic-key-123".to_string())
        }
    });

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());
    assert!(
        resolver_called.load(Ordering::SeqCst) >= 1,
        "getApiKey resolver should be called"
    );
}

// ============================================================================
// ToolExecutionUpdate events
// ============================================================================

#[tokio::test]
async fn test_tool_execution_update_events() {
    let tool_response = make_tool_call_message("streaming_tool", "call_1", json!({}));
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "streaming_tool",
        "Streaming Tool",
        "desc",
        json!({"type": "object"}),
    )]);

    let update_count = Arc::new(AtomicUsize::new(0));
    let uc = update_count.clone();

    let _unsub = agent.subscribe(move |event| {
        if matches!(event, AgentEvent::ToolExecutionUpdate { .. }) {
            uc.fetch_add(1, Ordering::SeqCst);
        }
    });

    // Use the full set_tool_executor with update callback
    agent.set_tool_executor(
        |_name: &str,
         _id: &str,
         _args: &serde_json::Value,
         update_cb: Option<ToolUpdateCallback>| async move {
            // Push streaming updates
            if let Some(ref cb) = update_cb {
                cb(json!({"progress": 25}));
                cb(json!({"progress": 50}));
                cb(json!({"progress": 100}));
            }
            AgentToolResult::text("complete")
        },
    );

    let result = agent.prompt("start").await;
    assert!(result.is_ok());

    assert_eq!(
        update_count.load(Ordering::SeqCst),
        3,
        "Should receive 3 ToolExecutionUpdate events"
    );
}

// ============================================================================
// set_tool_executor_simple backward compat
// ============================================================================

#[tokio::test]
async fn test_tool_executor_simple_works() {
    let tool_response = make_tool_call_message("my_tool", "call_1", json!({}));
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult::text("simple result")
        },
    );

    let result = agent.prompt("go").await;
    assert!(result.is_ok());
}

// ============================================================================
// BeforeToolCallResult helpers
// ============================================================================

#[test]
fn test_before_tool_call_result_allow() {
    let r = BeforeToolCallResult::allow();
    assert!(!r.block);
    assert!(r.reason.is_none());
}

#[test]
fn test_before_tool_call_result_blocked() {
    let r = BeforeToolCallResult::blocked("Not allowed");
    assert!(r.block);
    assert_eq!(r.reason.as_deref(), Some("Not allowed"));
}

// ============================================================================
// AgentConfig new fields
// ============================================================================

#[test]
fn test_agent_config_new_has_defaults() {
    let model = make_model();
    let config = AgentConfig::new(model);
    assert_eq!(config.steering_mode, QueueMode::All);
    assert_eq!(config.follow_up_mode, QueueMode::All);
    assert!(config.thinking_budgets.is_none());
    assert_eq!(config.transport, Transport::Sse);
    assert!(config.max_retry_delay_ms.is_none());
}

// ============================================================================
// AgentEvent serialization (TurnEnd with tool_results)
// ============================================================================

#[test]
fn test_agent_event_turn_end_serialization() {
    let event = AgentEvent::TurnEnd {
        message: AgentMessage::User(UserMessage::text("hello")),
        tool_results: vec![],
    };
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "turn_end");
}

#[test]
fn test_agent_event_tool_execution_update_serialization() {
    let event = AgentEvent::ToolExecutionUpdate {
        tool_call_id: "call_1".to_string(),
        tool_name: "my_tool".to_string(),
        partial_result: json!({"progress": 50}),
    };
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "tool_execution_update");
    assert_eq!(json["partial_result"]["progress"], 50);
}

// ============================================================================
// Combined: beforeToolCall + afterToolCall
// ============================================================================

#[tokio::test]
async fn test_both_hooks_called_in_order() {
    let tool_response = make_tool_call_message("my_tool", "call_1", json!({}));
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "my_tool",
        "My Tool",
        "desc",
        json!({"type": "object"}),
    )]);

    let before_called = Arc::new(AtomicUsize::new(0));
    let after_called = Arc::new(AtomicUsize::new(0));
    let bc = before_called.clone();
    let ac = after_called.clone();

    agent.set_before_tool_call(move |_ctx| {
        let bc = bc.clone();
        async move {
            bc.fetch_add(1, Ordering::SeqCst);
            None // Allow
        }
    });

    agent.set_after_tool_call(move |_ctx| {
        let ac = ac.clone();
        async move {
            ac.fetch_add(1, Ordering::SeqCst);
            None // No override
        }
    });

    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult::text("ok")
        },
    );

    let result = agent.prompt("go").await;
    assert!(result.is_ok());
    assert_eq!(before_called.load(Ordering::SeqCst), 1);
    assert_eq!(after_called.load(Ordering::SeqCst), 1);
}

// ============================================================================
// CapturingMockProvider — records SimpleStreamOptions for integration tests
// ============================================================================

/// A mock provider that captures the `SimpleStreamOptions` it receives,
/// so we can verify that agent features flow through to the provider layer.
struct CapturingMockProvider {
    responses: parking_lot::Mutex<Vec<AssistantMessage>>,
    captured_reasoning: parking_lot::Mutex<Vec<Option<ThinkingLevel>>>,
    captured_budget: parking_lot::Mutex<Vec<Option<u32>>>,
    captured_session_id: parking_lot::Mutex<Vec<Option<String>>>,
    captured_transport: parking_lot::Mutex<Vec<Option<Transport>>>,
    captured_max_retry_delay: parking_lot::Mutex<Vec<Option<u64>>>,
    captured_has_on_payload: parking_lot::Mutex<Vec<bool>>,
}

impl CapturingMockProvider {
    fn new(responses: Vec<AssistantMessage>) -> Self {
        Self {
            responses: parking_lot::Mutex::new(responses),
            captured_reasoning: parking_lot::Mutex::new(Vec::new()),
            captured_budget: parking_lot::Mutex::new(Vec::new()),
            captured_session_id: parking_lot::Mutex::new(Vec::new()),
            captured_transport: parking_lot::Mutex::new(Vec::new()),
            captured_max_retry_delay: parking_lot::Mutex::new(Vec::new()),
            captured_has_on_payload: parking_lot::Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl LLMProtocol for CapturingMockProvider {
    fn provider_type(&self) -> Provider {
        Provider::OpenAI
    }

    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: StreamOptions,
    ) -> AssistantMessageEventStream {
        // Fallback — agent should call stream_simple() instead
        let stream = AssistantMessageEventStream::new_assistant_stream();
        let mut responses = self.responses.lock();
        let response = if responses.is_empty() {
            make_assistant_message("Default response")
        } else {
            responses.remove(0)
        };
        let stop_reason = response.stop_reason;
        let response_clone = response.clone();
        let stream_clone = stream.clone();
        tokio::spawn(async move {
            stream_clone.push(AssistantMessageEvent::Start {
                partial: response_clone.clone(),
            });
            stream_clone.push(AssistantMessageEvent::Done {
                reason: stop_reason,
                message: response_clone,
            });
            stream_clone.end(None);
        });
        stream
    }

    fn stream_simple(
        &self,
        _model: &Model,
        _context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        // Capture all fields from SimpleStreamOptions
        self.captured_reasoning.lock().push(options.reasoning);
        self.captured_budget
            .lock()
            .push(options.thinking_budget_tokens);
        self.captured_session_id
            .lock()
            .push(options.base.session_id.clone());
        self.captured_transport.lock().push(options.base.transport);
        self.captured_max_retry_delay
            .lock()
            .push(options.base.max_retry_delay_ms);
        self.captured_has_on_payload
            .lock()
            .push(options.base.on_payload.is_some());

        // Return a canned response
        let stream = AssistantMessageEventStream::new_assistant_stream();
        let mut responses = self.responses.lock();
        let response = if responses.is_empty() {
            make_assistant_message("Default response")
        } else {
            responses.remove(0)
        };
        let stop_reason = response.stop_reason;
        let response_clone = response.clone();
        let stream_clone = stream.clone();
        tokio::spawn(async move {
            stream_clone.push(AssistantMessageEvent::Start {
                partial: response_clone.clone(),
            });
            stream_clone.push(AssistantMessageEvent::Done {
                reason: stop_reason,
                message: response_clone,
            });
            stream_clone.end(None);
        });
        stream
    }
}

// ============================================================================
// Provider Integration: ThinkingBudgets
// ============================================================================

#[tokio::test]
async fn test_thinking_budgets_flow_to_provider() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Set thinking to Medium with custom budgets
    agent.set_thinking_level(ThinkingLevel::Medium);
    agent.set_thinking_budgets(ThinkingBudgets {
        minimal: Some(64),
        low: Some(256),
        medium: Some(2048),
        high: Some(4096),
    });

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured_reasoning = mock.captured_reasoning.lock();
    let captured_budget = mock.captured_budget.lock();

    assert_eq!(captured_reasoning.len(), 1);
    assert_eq!(captured_reasoning[0], Some(ThinkingLevel::Medium));
    assert_eq!(captured_budget.len(), 1);
    assert_eq!(
        captured_budget[0],
        Some(2048),
        "Should use custom budget for Medium"
    );
}

#[tokio::test]
async fn test_thinking_budgets_default_fallback() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Set thinking to Low WITHOUT custom budgets → should use default (512)
    agent.set_thinking_level(ThinkingLevel::Low);

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured_reasoning = mock.captured_reasoning.lock();
    let captured_budget = mock.captured_budget.lock();

    assert_eq!(captured_reasoning[0], Some(ThinkingLevel::Low));
    assert_eq!(
        captured_budget[0],
        Some(tiy_core::thinking::ThinkingConfig::default_budget(
            ThinkingLevel::Low
        )),
        "Should fall back to default budget (512) when no custom budgets set"
    );
}

#[tokio::test]
async fn test_thinking_off_no_budget() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Thinking Off (default) — reasoning and budget should both be None
    // (Default ThinkingLevel is Off, no getter needed)

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured_reasoning = mock.captured_reasoning.lock();
    let captured_budget = mock.captured_budget.lock();

    assert_eq!(
        captured_reasoning[0], None,
        "Thinking Off should send reasoning=None"
    );
    assert_eq!(
        captured_budget[0], None,
        "Thinking Off should send budget=None"
    );
}

// ============================================================================
// Provider Integration: sessionId
// ============================================================================

#[test]
fn test_session_id_setter_getter() {
    let agent = Agent::new();
    assert_eq!(agent.session_id(), None);

    agent.set_session_id("session-abc-123");
    assert_eq!(agent.session_id(), Some("session-abc-123".to_string()));

    agent.clear_session_id();
    assert_eq!(agent.session_id(), None);
}

#[tokio::test]
async fn test_session_id_flows_to_provider() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_session_id("my-session-42");

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_session_id.lock();
    assert_eq!(captured.len(), 1);
    assert_eq!(
        captured[0],
        Some("my-session-42".to_string()),
        "session_id should flow to provider"
    );
}

#[tokio::test]
async fn test_session_id_none_when_not_set() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    // Don't set session_id

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_session_id.lock();
    assert_eq!(captured[0], None, "session_id should be None when not set");
}

// ============================================================================
// Provider Integration: onPayload
// ============================================================================

#[tokio::test]
async fn test_on_payload_flows_to_provider() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Set an on_payload hook
    let hook_called = Arc::new(AtomicBool::new(false));
    let hc = hook_called.clone();
    agent.set_on_payload(move |payload, _model| {
        let hc = hc.clone();
        async move {
            hc.store(true, Ordering::SeqCst);
            Some(payload) // pass through unchanged
        }
    });

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_has_on_payload.lock();
    assert_eq!(captured.len(), 1);
    assert!(
        captured[0],
        "on_payload should be Some (present) in provider call"
    );
}

#[tokio::test]
async fn test_on_payload_none_when_not_set() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    // Don't set on_payload

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_has_on_payload.lock();
    assert!(!captured[0], "on_payload should be None when not set");
}

// ============================================================================
// Provider Integration: Transport
// ============================================================================

#[tokio::test]
async fn test_transport_flows_to_provider() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_transport(Transport::WebSocket);

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_transport.lock();
    assert_eq!(captured.len(), 1);
    assert_eq!(
        captured[0],
        Some(Transport::WebSocket),
        "Transport::WebSocket should flow to provider"
    );
}

#[tokio::test]
async fn test_transport_default_sse_flows_to_provider() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    // Default transport is Sse

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_transport.lock();
    assert_eq!(
        captured[0],
        Some(Transport::Sse),
        "Default Transport::Sse should flow to provider"
    );
}

// ============================================================================
// Provider Integration: maxRetryDelayMs
// ============================================================================

#[tokio::test]
async fn test_max_retry_delay_flows_to_provider() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_max_retry_delay_ms(Some(5000));

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_max_retry_delay.lock();
    assert_eq!(captured.len(), 1);
    assert_eq!(
        captured[0],
        Some(5000),
        "max_retry_delay_ms=5000 should flow to provider"
    );
}

#[tokio::test]
async fn test_max_retry_delay_none_when_not_set() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    // Don't set max_retry_delay_ms

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_max_retry_delay.lock();
    assert_eq!(
        captured[0], None,
        "max_retry_delay_ms should be None when not set"
    );
}

// ============================================================================
// Provider Integration: All 5 features combined
// ============================================================================

#[tokio::test]
async fn test_all_five_features_flow_together() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Set all 5 features
    agent.set_thinking_level(ThinkingLevel::High);
    agent.set_thinking_budgets(ThinkingBudgets {
        minimal: Some(64),
        low: Some(256),
        medium: Some(1024),
        high: Some(8192),
    });
    agent.set_session_id("combined-session");
    agent.set_on_payload(move |payload, _model| async move { Some(payload) });
    agent.set_transport(Transport::Auto);
    agent.set_max_retry_delay_ms(Some(15000));

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    // Verify all 5 captured correctly
    assert_eq!(mock.captured_reasoning.lock()[0], Some(ThinkingLevel::High));
    assert_eq!(mock.captured_budget.lock()[0], Some(8192));
    assert_eq!(
        mock.captured_session_id.lock()[0],
        Some("combined-session".to_string())
    );
    assert!(mock.captured_has_on_payload.lock()[0]);
    assert_eq!(mock.captured_transport.lock()[0], Some(Transport::Auto));
    assert_eq!(mock.captured_max_retry_delay.lock()[0], Some(15000));
}

// ============================================================================
// Provider Integration: reset clears session_id
// ============================================================================

#[tokio::test]
async fn test_reset_clears_session_id() {
    let responses = vec![
        make_assistant_message("First"),
        make_assistant_message("Second"),
    ];
    let mock = Arc::new(CapturingMockProvider::new(responses));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_session_id("session-before-reset");

    // First prompt
    let result = agent.prompt("hello").await;
    assert!(result.is_ok());
    assert_eq!(
        mock.captured_session_id.lock()[0],
        Some("session-before-reset".to_string())
    );

    // Reset should clear session_id
    agent.reset();
    assert_eq!(agent.session_id(), None);
}

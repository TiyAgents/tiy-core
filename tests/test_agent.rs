//! Tests for agent module: types, state, agent.

use serde_json::json;
use tiycore::agent::*;
use tiycore::thinking::ThinkingLevel;
use tiycore::types::*;

// ============================================================================
// AgentMessage tests
// ============================================================================

#[test]
fn test_agent_message_from_user() {
    let user = UserMessage::text("Hello");
    let agent_msg: AgentMessage = user.into();
    assert!(matches!(agent_msg, AgentMessage::User(_)));
}

#[test]
fn test_agent_message_from_assistant() {
    let assistant = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("gpt-4o")
        .build()
        .unwrap();
    let agent_msg: AgentMessage = assistant.into();
    assert!(matches!(agent_msg, AgentMessage::Assistant(_)));
}

#[test]
fn test_agent_message_from_tool_result() {
    let tr = ToolResultMessage::text("call_1", "tool", "result", false);
    let agent_msg: AgentMessage = tr.into();
    assert!(matches!(agent_msg, AgentMessage::ToolResult(_)));
}

#[test]
fn test_agent_message_from_message() {
    let msg = Message::User(UserMessage::text("Hello"));
    let agent_msg: AgentMessage = msg.into();
    assert!(matches!(agent_msg, AgentMessage::User(_)));
}

#[test]
fn test_agent_message_to_message() {
    let agent_msg = AgentMessage::User(UserMessage::text("Hello"));
    let msg: Option<Message> = agent_msg.into();
    assert!(msg.is_some());
    assert!(msg.unwrap().is_user());
}

// ============================================================================
// AgentTool tests
// ============================================================================

#[test]
fn test_agent_tool_new() {
    let tool = AgentTool::new(
        "get_weather",
        "Get Weather",
        "Get weather for a location",
        json!({"type": "object", "properties": {"city": {"type": "string"}}}),
    );
    assert_eq!(tool.name, "get_weather");
    assert_eq!(tool.label, "Get Weather");
    assert_eq!(tool.description, "Get weather for a location");
}

#[test]
fn test_agent_tool_as_tool() {
    let agent_tool = AgentTool::new(
        "calc",
        "Calculator",
        "Perform calculations",
        json!({"type": "object"}),
    );
    let tool = agent_tool.as_tool();
    assert_eq!(tool.name, "calc");
    assert_eq!(tool.description, "Perform calculations");
}

#[test]
fn test_agent_tool_from_tool() {
    let tool = Tool::new("my_tool", "My description", json!({"type": "object"}));
    let agent_tool: AgentTool = tool.into();
    assert_eq!(agent_tool.name, "my_tool");
    assert_eq!(agent_tool.label, "my_tool"); // label defaults to name
    assert_eq!(agent_tool.description, "My description");
}

// ============================================================================
// AgentConfig tests
// ============================================================================

#[test]
fn test_agent_config_new() {
    let model = Model::builder()
        .id("gpt-4o")
        .name("GPT-4o")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url("http://test")
        .context_window(128000)
        .max_tokens(16384)
        .build()
        .unwrap();

    let config = AgentConfig::new(model.clone());
    assert_eq!(config.model.id, "gpt-4o");
    assert_eq!(config.thinking_level, ThinkingLevel::Off);
    assert_eq!(config.tool_execution, ToolExecutionMode::Parallel);
}

// ============================================================================
// ToolExecutionMode tests
// ============================================================================

#[test]
fn test_tool_execution_mode_default_is_parallel() {
    assert_eq!(ToolExecutionMode::default(), ToolExecutionMode::Parallel);
}

// ============================================================================
// AgentToolResult tests
// ============================================================================

#[test]
fn test_agent_tool_result_text() {
    let result = AgentToolResult::text("Hello result");
    assert_eq!(result.content.len(), 1);
    assert!(result.content[0].is_text());
    assert!(result.details.is_none());
}

#[test]
fn test_agent_tool_result_error() {
    let result = AgentToolResult::error("Something failed");
    assert_eq!(result.content.len(), 1);
    assert!(result.content[0].is_text());
}

// ============================================================================
// AgentContext tests
// ============================================================================

#[test]
fn test_agent_context_default() {
    let ctx = AgentContext::default();
    assert_eq!(ctx.system_prompt, "");
    assert!(ctx.messages.is_empty());
    assert!(ctx.tools.is_none());
}

// ============================================================================
// AgentEvent tests
// ============================================================================

#[test]
fn test_agent_event_variants() {
    let _start = AgentEvent::AgentStart;
    let _end = AgentEvent::AgentEnd { messages: vec![] };
    let _turn_start = AgentEvent::TurnStart { turn_index: 0 };
    let _tool_start = AgentEvent::ToolExecutionStart {
        turn_index: 0,
        tool_call_id: "id".to_string(),
        tool_name: "tool".to_string(),
        args: json!({}),
    };
    let _tool_end = AgentEvent::ToolExecutionEnd {
        turn_index: 0,
        tool_call_id: "id".to_string(),
        tool_name: "tool".to_string(),
        result: json!({}),
        is_error: false,
    };
    // Just verifying these compile and can be created
}

// ============================================================================
// AgentState tests
// ============================================================================

#[test]
fn test_agent_state_new() {
    let state = AgentState::new();
    assert_eq!(*state.system_prompt.read(), "");
    assert!(state.messages.read().is_empty());
    assert!(!state.is_streaming());
    assert_eq!(state.message_count(), 0);
}

#[test]
fn test_agent_state_set_system_prompt() {
    let state = AgentState::new();
    state.set_system_prompt("You are helpful.");
    assert_eq!(*state.system_prompt.read(), "You are helpful.");
}

// test_agent_state_set_model removed: model now lives in AgentConfig only.

#[test]
fn test_agent_state_set_thinking_level() {
    // thinking_level now lives in AgentConfig; AgentState no longer stores it.
    // This test verifies it's accessible via Agent::snapshot().
    let agent = Agent::new();
    agent.set_thinking_level(ThinkingLevel::High);
    let snapshot = agent.snapshot();
    assert_eq!(snapshot.thinking_level, ThinkingLevel::High);
}

#[test]
fn test_agent_state_set_tools() {
    let state = AgentState::new();
    state.set_tools(vec![
        AgentTool::new("tool1", "Tool 1", "desc1", json!({})),
        AgentTool::new("tool2", "Tool 2", "desc2", json!({})),
    ]);
    assert_eq!(state.tools.read().len(), 2);
}

#[test]
fn test_agent_state_messages() {
    let state = AgentState::new();
    state.add_message(AgentMessage::User(UserMessage::text("Hello")));
    state.add_message(AgentMessage::User(UserMessage::text("World")));
    assert_eq!(state.message_count(), 2);

    state.replace_messages(vec![AgentMessage::User(UserMessage::text("New"))]);
    assert_eq!(state.message_count(), 1);

    state.clear_messages();
    assert_eq!(state.message_count(), 0);
}

#[test]
fn test_agent_state_streaming() {
    let state = AgentState::new();
    assert!(!state.is_streaming());
    state.set_streaming(true);
    assert!(state.is_streaming());
    state.set_streaming(false);
    assert!(!state.is_streaming());
}

#[test]
fn test_agent_state_reset() {
    let state = AgentState::new();
    state.set_system_prompt("test");
    state.add_message(AgentMessage::User(UserMessage::text("hello")));
    state.set_streaming(true);
    *state.error.write() = Some("err".to_string());

    state.reset();

    assert_eq!(*state.system_prompt.read(), "");
    assert!(state.messages.read().is_empty());
    assert!(!state.is_streaming());
    assert!(state.error.read().is_none());
}

#[test]
fn test_agent_state_clone() {
    let state = AgentState::new();
    state.set_system_prompt("test");
    state.add_message(AgentMessage::User(UserMessage::text("hello")));

    let cloned = state.clone();
    assert_eq!(*cloned.system_prompt.read(), "test");
    assert_eq!(cloned.message_count(), 1);

    // Modifying original doesn't affect clone
    state.set_system_prompt("modified");
    assert_eq!(*cloned.system_prompt.read(), "test");
}

// test_agent_state_with_model removed: model now lives in AgentConfig only.
// Use Agent::with_model() instead.

// ============================================================================
// Agent tests
// ============================================================================

#[test]
fn test_agent_new_defaults() {
    let agent = Agent::new();
    let state = agent.state();
    assert_eq!(*state.system_prompt.read(), "");
    // thinking_level now accessed via snapshot
    let snapshot = agent.snapshot();
    assert_eq!(snapshot.thinking_level, ThinkingLevel::Off);
    assert!(!state.is_streaming());
}

#[test]
fn test_agent_with_model() {
    let model = Model::builder()
        .id("claude-sonnet-4")
        .name("Claude Sonnet 4")
        .api(Api::AnthropicMessages)
        .provider(Provider::Anthropic)
        .base_url("https://api.anthropic.com/v1")
        .context_window(200000)
        .max_tokens(16000)
        .build()
        .unwrap();

    let agent = Agent::with_model(model);
    let snapshot = agent.snapshot();
    assert_eq!(snapshot.model.id, "claude-sonnet-4");
}

#[test]
fn test_agent_set_system_prompt() {
    let agent = Agent::new();
    agent.set_system_prompt("You are an AI.");
    assert_eq!(*agent.state().system_prompt.read(), "You are an AI.");
}

#[test]
fn test_agent_set_model() {
    let agent = Agent::new();
    let model = Model::builder()
        .id("new-model")
        .name("New")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url("http://test")
        .context_window(4096)
        .max_tokens(1024)
        .build()
        .unwrap();

    agent.set_model(model);
    let snapshot = agent.snapshot();
    assert_eq!(snapshot.model.id, "new-model");
}

#[test]
fn test_agent_set_thinking_level() {
    let agent = Agent::new();
    agent.set_thinking_level(ThinkingLevel::High);
    let snapshot = agent.snapshot();
    assert_eq!(snapshot.thinking_level, ThinkingLevel::High);
}

#[test]
fn test_agent_set_tools() {
    let agent = Agent::new();
    agent.set_tools(vec![AgentTool::new("tool1", "Tool 1", "desc1", json!({}))]);
    assert_eq!(agent.state().tools.read().len(), 1);
}

#[test]
fn test_agent_messages_operations() {
    let agent = Agent::new();

    agent.append_message(AgentMessage::User(UserMessage::text("Hello")));
    assert_eq!(agent.state().message_count(), 1);

    agent.append_message(AgentMessage::User(UserMessage::text("World")));
    assert_eq!(agent.state().message_count(), 2);

    agent.replace_messages(vec![AgentMessage::User(UserMessage::text("New"))]);
    assert_eq!(agent.state().message_count(), 1);

    agent.clear_messages();
    assert_eq!(agent.state().message_count(), 0);
}

#[test]
fn test_agent_steering_queue() {
    let agent = Agent::new();

    assert!(!agent.has_queued_messages());

    agent.steer(AgentMessage::User(UserMessage::text("Interrupt")));
    assert!(agent.has_queued_messages());

    agent.clear_steering_queue();
    assert!(!agent.has_queued_messages());
}

#[test]
fn test_agent_follow_up_queue() {
    let agent = Agent::new();

    agent.follow_up(AgentMessage::User(UserMessage::text("Later")));
    assert!(agent.has_queued_messages());

    agent.clear_follow_up_queue();
    assert!(!agent.has_queued_messages());
}

#[test]
fn test_agent_clear_all_queues() {
    let agent = Agent::new();

    agent.steer(AgentMessage::User(UserMessage::text("Interrupt")));
    agent.follow_up(AgentMessage::User(UserMessage::text("Later")));
    assert!(agent.has_queued_messages());

    agent.clear_all_queues();
    assert!(!agent.has_queued_messages());
}

#[test]
fn test_agent_reset() {
    let agent = Agent::new();
    agent.set_system_prompt("test");
    agent.append_message(AgentMessage::User(UserMessage::text("hi")));
    agent.steer(AgentMessage::User(UserMessage::text("interrupt")));
    agent.follow_up(AgentMessage::User(UserMessage::text("later")));

    agent.reset();

    assert_eq!(*agent.state().system_prompt.read(), "");
    assert_eq!(agent.state().message_count(), 0);
    assert!(!agent.has_queued_messages());
}

#[test]
fn test_agent_abort() {
    let agent = Agent::new();
    agent.state().set_streaming(true);
    agent.steer(AgentMessage::User(UserMessage::text("x")));

    agent.abort();

    assert!(!agent.state().is_streaming());
    assert!(agent.has_queued_messages());
}

#[tokio::test]
async fn test_agent_prompt_basic() {
    // Without a provider registered, prompt should return a ProviderError
    let agent = Agent::new();
    let result = agent.prompt(UserMessage::text("Hello")).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        AgentError::ProviderError(_) => {}
        other => panic!("Expected ProviderError, got {:?}", other),
    }
    // Even on error, streaming should be cleaned up
    assert!(!agent.state().is_streaming());
}

#[tokio::test]
async fn test_agent_prompt_already_streaming() {
    let agent = Agent::new();
    agent.state().set_streaming(true);

    let result = agent.prompt(UserMessage::text("Hello")).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        AgentError::AlreadyStreaming => {}
        other => panic!("Expected AlreadyStreaming, got {:?}", other),
    }

    agent.state().set_streaming(false);
}

#[tokio::test]
async fn test_agent_continue_no_messages() {
    let agent = Agent::new();
    let result = agent.continue_().await;
    assert!(matches!(result, Err(AgentError::NoMessages)));
}

#[tokio::test]
async fn test_agent_continue_from_assistant() {
    let agent = Agent::new();
    let assistant = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("gpt-4o")
        .build()
        .unwrap();
    agent.append_message(AgentMessage::Assistant(assistant));

    let result = agent.continue_().await;
    assert!(matches!(
        result,
        Err(AgentError::CannotContinueFromAssistant)
    ));
}

#[tokio::test]
async fn test_agent_continue_from_tool_result() {
    // Without a provider registered, continue_ should return a ProviderError
    let agent = Agent::new();
    agent.append_message(AgentMessage::ToolResult(ToolResultMessage::text(
        "call_1", "tool", "result", false,
    )));

    let result = agent.continue_().await;
    assert!(result.is_err());
    match result.unwrap_err() {
        AgentError::ProviderError(_) => {}
        other => panic!("Expected ProviderError, got {:?}", other),
    }
}

#[test]
fn test_agent_subscribe_and_emit() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    let agent = Agent::new();
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let _unsub = agent.subscribe(move |_event| {
        count_clone.fetch_add(1, Ordering::SeqCst);
    });

    // Trigger events via prompt (which emits AgentStart then ProviderError → AgentEnd)
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let _ = agent.prompt(UserMessage::text("hello")).await;
    });

    // Should have received AgentStart + AgentEnd even on error
    assert!(count.load(Ordering::SeqCst) >= 2);
}

// ============================================================================
// AgentError tests
// ============================================================================

#[test]
fn test_agent_error_display() {
    assert_eq!(
        format!("{}", AgentError::AlreadyStreaming),
        "Agent is already streaming"
    );
    assert_eq!(
        format!("{}", AgentError::NoMessages),
        "No messages in context"
    );
    assert_eq!(
        format!("{}", AgentError::CannotContinueFromAssistant),
        "Cannot continue from assistant message"
    );
    assert_eq!(
        format!("{}", AgentError::ToolNotFound("foo".into())),
        "Tool not found: foo"
    );
    assert_eq!(
        format!("{}", AgentError::ProviderError("bad".into())),
        "Provider error: bad"
    );
    assert_eq!(
        format!("{}", AgentError::MaxTurnsReached(25)),
        "Agent reached the maximum turn limit (25) before producing a final response"
    );
    assert_eq!(format!("{}", AgentError::Other("misc".into())), "misc");
}

// ============================================================================
// Promote follow-up → steering tests
// ============================================================================

#[test]
fn test_promote_follow_up_to_steering_success() {
    let agent = Agent::new();
    let handle = agent.follow_up(AgentMessage::from("hello"));
    assert_eq!(handle.kind, QueueKind::FollowUp);

    let new_handle = agent
        .promote_follow_up_to_steering(handle.id)
        .expect("promote should succeed");

    // New handle points to the steering queue
    assert_eq!(new_handle.kind, QueueKind::Steering);

    // Old follow-up handle is invalidated
    assert!(agent.cancel_follow_up_message(handle.id).is_none());

    // New steering handle is valid
    let msg = agent
        .cancel_steering_message(new_handle.id)
        .expect("should be able to cancel the promoted message");
    assert!(matches!(msg, AgentMessage::User(_)));
}

#[test]
fn test_promote_follow_up_not_found() {
    let agent = Agent::new();
    // Enqueue and immediately drain to get an id that no longer exists
    let handle = agent.follow_up(AgentMessage::from("temp"));
    agent.clear_follow_up_queue();
    assert!(agent.promote_follow_up_to_steering(handle.id).is_none());
}

#[test]
fn test_promote_follow_up_event_sequence() {
    use std::sync::{Arc, Mutex};

    let agent = Agent::new();
    let events: Arc<Mutex<Vec<QueueEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    agent.set_on_queue_event(move |e| {
        events_clone.lock().unwrap().push(e);
    });

    let handle = agent.follow_up(AgentMessage::from("test"));
    // Clear events from the follow_up call
    events.lock().unwrap().clear();

    let _new_handle = agent.promote_follow_up_to_steering(handle.id).unwrap();

    let captured = events.lock().unwrap();
    // Expected: Removed(FollowUp), Transferred(FollowUp→Steering), Enqueued(Steering)
    assert_eq!(captured.len(), 3);

    assert!(matches!(
        &captured[0],
        QueueEvent::Removed { kind: QueueKind::FollowUp, count: 1, .. }
    ));
    assert!(matches!(
        &captured[1],
        QueueEvent::Transferred {
            from: QueueKind::FollowUp,
            to: QueueKind::Steering,
            count: 1,
            ..
        }
    ));
    assert!(matches!(
        &captured[2],
        QueueEvent::Enqueued { kind: QueueKind::Steering, count: 1, .. }
    ));
}

#[test]
fn test_try_promote_follow_up_queue_full() {
    let agent = Agent::new();
    // Set steering queue to max_depth=1, Reject
    agent.set_steering_backpressure(BackpressureConfig {
        max_depth: 1,
        overflow: OverflowBehavior::Reject,
    });
    // Fill the steering queue
    agent.steer(AgentMessage::from("blocking"));

    let handle = agent.follow_up(AgentMessage::from("promote-me"));

    let result = agent.try_promote_follow_up_to_steering(handle.id);
    assert!(result.is_err());
    match result.unwrap_err() {
        PromoteError::QueueFull(err) => {
            assert_eq!(err.max_depth, 1);
        }
        other => panic!("Expected QueueFull, got {:?}", other),
    }

    // Message should still be in the follow-up queue (can_push pre-check prevented removal)
    let msg = agent
        .cancel_follow_up_message(handle.id)
        .expect("message should still be in follow-up queue");
    assert!(matches!(msg, AgentMessage::User(_)));
}

#[test]
fn test_try_promote_follow_up_not_found() {
    let agent = Agent::new();
    // Enqueue and immediately drain to get an id that no longer exists
    let handle = agent.follow_up(AgentMessage::from("temp"));
    agent.clear_follow_up_queue();
    let result = agent.try_promote_follow_up_to_steering(handle.id);
    assert!(matches!(result, Err(PromoteError::NotFound)));
}

#[test]
fn test_promote_error_display() {
    let not_found = PromoteError::NotFound;
    assert_eq!(format!("{}", not_found), "message not found in follow-up queue");

    // Construct a QueueFullError via Agent-level try_steer (which calls try_push)
    let agent = Agent::new();
    agent.set_steering_backpressure(BackpressureConfig {
        max_depth: 1,
        overflow: OverflowBehavior::Reject,
    });
    agent.steer(AgentMessage::from("fill"));
    let err = agent.try_steer(AgentMessage::from("overflow")).unwrap_err();
    let queue_full = PromoteError::QueueFull(Box::new(err));
    assert!(format!("{}", queue_full).contains("steering queue full"));
}

#[test]
fn test_promote_multiple_messages_preserves_order() {
    let agent = Agent::new();

    let h1 = agent.follow_up(AgentMessage::from("first"));
    let h2 = agent.follow_up(AgentMessage::from("second"));

    // Promote the second one first
    let new_h2 = agent.promote_follow_up_to_steering(h2.id).unwrap();
    assert_eq!(new_h2.kind, QueueKind::Steering);

    // First should still be in follow-up
    let msg1 = agent.cancel_follow_up_message(h1.id).unwrap();
    assert!(matches!(msg1, AgentMessage::User(_)));
}

#[test]
fn test_queue_full_error_into_message() {
    // Construct a QueueFullError with a message via Agent try_steer
    let agent = Agent::new();
    agent.set_steering_backpressure(BackpressureConfig {
        max_depth: 1,
        overflow: OverflowBehavior::Reject,
    });
    agent.steer(AgentMessage::from("fill"));
    let err = agent.try_steer(AgentMessage::from("overflow")).unwrap_err();
    let msg = err.into_message().expect("should have a rejected message");
    assert!(matches!(msg, AgentMessage::User(_)));

    // can_push errors don't carry a message (not constructable from outside,
    // but we verify into_message returns None when message field is None
    // by checking a QueueFullError created without one)
    let err2 = QueueFullError::new(1, 1);
    assert!(err2.into_message().is_none());
}


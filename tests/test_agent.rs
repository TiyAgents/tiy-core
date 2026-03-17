//! Tests for agent module: types, state, agent.

use serde_json::json;
use tiy_core::types::*;
use tiy_core::agent::*;
use tiy_core::thinking::ThinkingLevel;

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
    let _turn_start = AgentEvent::TurnStart;
    let _tool_start = AgentEvent::ToolExecutionStart {
        tool_call_id: "id".to_string(),
        tool_name: "tool".to_string(),
        args: json!({}),
    };
    let _tool_end = AgentEvent::ToolExecutionEnd {
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
    assert_eq!(*state.thinking_level.read(), ThinkingLevel::Off);
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

#[test]
fn test_agent_state_set_model() {
    let state = AgentState::new();
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
    state.set_model(model);
    assert_eq!(state.model.read().id, "claude-sonnet-4");
}

#[test]
fn test_agent_state_set_thinking_level() {
    let state = AgentState::new();
    state.set_thinking_level(ThinkingLevel::High);
    assert_eq!(*state.thinking_level.read(), ThinkingLevel::High);
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
    state.set_thinking_level(ThinkingLevel::High);
    state.add_message(AgentMessage::User(UserMessage::text("hello")));
    state.set_streaming(true);
    *state.error.write() = Some("err".to_string());

    state.reset();

    assert_eq!(*state.system_prompt.read(), "");
    assert_eq!(*state.thinking_level.read(), ThinkingLevel::Off);
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

#[test]
fn test_agent_state_with_model() {
    let model = Model::builder()
        .id("custom-model")
        .name("Custom")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url("http://test")
        .context_window(4096)
        .max_tokens(1024)
        .build()
        .unwrap();

    let state = AgentState::with_model(model);
    assert_eq!(state.model.read().id, "custom-model");
}

// ============================================================================
// Agent tests
// ============================================================================

#[test]
fn test_agent_new_defaults() {
    let agent = Agent::new();
    let state = agent.state();
    assert_eq!(*state.system_prompt.read(), "");
    assert_eq!(*state.thinking_level.read(), ThinkingLevel::Off);
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
    assert_eq!(agent.state().model.read().id, "claude-sonnet-4");
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
    assert_eq!(agent.state().model.read().id, "new-model");
}

#[test]
fn test_agent_set_thinking_level() {
    let agent = Agent::new();
    agent.set_thinking_level(ThinkingLevel::High);
    assert_eq!(*agent.state().thinking_level.read(), ThinkingLevel::High);
}

#[test]
fn test_agent_set_tools() {
    let agent = Agent::new();
    agent.set_tools(vec![
        AgentTool::new("tool1", "Tool 1", "desc1", json!({})),
    ]);
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
    assert!(!agent.has_queued_messages());
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
    assert!(matches!(result, Err(AgentError::CannotContinueFromAssistant)));
}

#[tokio::test]
async fn test_agent_continue_from_tool_result() {
    // Without a provider registered, continue_ should return a ProviderError
    let agent = Agent::new();
    agent.append_message(AgentMessage::ToolResult(
        ToolResultMessage::text("call_1", "tool", "result", false),
    ));

    let result = agent.continue_().await;
    assert!(result.is_err());
    match result.unwrap_err() {
        AgentError::ProviderError(_) => {}
        other => panic!("Expected ProviderError, got {:?}", other),
    }
}

#[test]
fn test_agent_subscribe_and_emit() {
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

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
    assert_eq!(format!("{}", AgentError::AlreadyStreaming), "Agent is already streaming");
    assert_eq!(format!("{}", AgentError::NoMessages), "No messages in context");
    assert_eq!(format!("{}", AgentError::CannotContinueFromAssistant), "Cannot continue from assistant message");
    assert_eq!(format!("{}", AgentError::ToolNotFound("foo".into())), "Tool not found: foo");
    assert_eq!(format!("{}", AgentError::ProviderError("bad".into())), "Provider error: bad");
    assert_eq!(format!("{}", AgentError::Other("misc".into())), "misc");
}

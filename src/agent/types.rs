//! Agent types and configurations.

use crate::thinking::ThinkingLevel;
use crate::types::*;
use serde::{Deserialize, Serialize};

/// Agent message - can include custom message types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum AgentMessage {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
    // Custom message types can be added here
}

impl From<UserMessage> for AgentMessage {
    fn from(msg: UserMessage) -> Self {
        AgentMessage::User(msg)
    }
}

impl From<AssistantMessage> for AgentMessage {
    fn from(msg: AssistantMessage) -> Self {
        AgentMessage::Assistant(msg)
    }
}

impl From<ToolResultMessage> for AgentMessage {
    fn from(msg: ToolResultMessage) -> Self {
        AgentMessage::ToolResult(msg)
    }
}

impl From<Message> for AgentMessage {
    fn from(msg: Message) -> Self {
        match msg {
            Message::User(m) => AgentMessage::User(m),
            Message::Assistant(m) => AgentMessage::Assistant(m),
            Message::ToolResult(m) => AgentMessage::ToolResult(m),
        }
    }
}

impl From<AgentMessage> for Option<Message> {
    fn from(msg: AgentMessage) -> Self {
        match msg {
            AgentMessage::User(m) => Some(Message::User(m)),
            AgentMessage::Assistant(m) => Some(Message::Assistant(m)),
            AgentMessage::ToolResult(m) => Some(Message::ToolResult(m)),
        }
    }
}

/// Agent tool with execution capability.
#[derive(Debug, Clone)]
pub struct AgentTool {
    /// Tool name.
    pub name: String,
    /// Human-readable label for UI.
    pub label: String,
    /// Tool description.
    pub description: String,
    /// Parameters schema.
    pub parameters: serde_json::Value,
}

impl AgentTool {
    /// Create a new agent tool.
    pub fn new(
        name: impl Into<String>,
        label: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            description: description.into(),
            parameters,
        }
    }

    /// Convert to a basic Tool.
    pub fn as_tool(&self) -> Tool {
        Tool::new(&self.name, &self.description, self.parameters.clone())
    }
}

impl From<Tool> for AgentTool {
    fn from(tool: Tool) -> Self {
        let name = tool.name.clone();
        Self {
            name,
            label: tool.name,
            description: tool.description,
            parameters: tool.parameters,
        }
    }
}

/// Agent context.
#[derive(Debug, Clone)]
pub struct AgentContext {
    /// System prompt.
    pub system_prompt: String,
    /// Messages.
    pub messages: Vec<AgentMessage>,
    /// Tools.
    pub tools: Option<Vec<AgentTool>>,
}

impl Default for AgentContext {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            messages: Vec::new(),
            tools: None,
        }
    }
}

/// Agent event types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Agent started.
    AgentStart,
    /// Agent finished.
    AgentEnd { messages: Vec<AgentMessage> },
    /// Turn started.
    TurnStart,
    /// Turn finished.
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<ToolResultMessage>,
    },
    /// Message started.
    MessageStart { message: AgentMessage },
    /// Message updated (streaming).
    MessageUpdate {
        message: AgentMessage,
        assistant_event: AssistantMessageEvent,
    },
    /// Message finished.
    MessageEnd { message: AgentMessage },
    /// Tool execution started.
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    /// Tool execution progress.
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        partial_result: serde_json::Value,
    },
    /// Tool execution finished.
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: serde_json::Value,
        is_error: bool,
    },
}

/// Tool execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ToolExecutionMode {
    /// Execute tools sequentially.
    Sequential,
    /// Execute tools in parallel.
    #[default]
    Parallel,
}

/// Agent configuration.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Model to use.
    pub model: Model,
    /// Thinking level.
    pub thinking_level: ThinkingLevel,
    /// Tool execution mode.
    pub tool_execution: ToolExecutionMode,
    /// Security and resource limits.
    pub security: crate::types::SecurityConfig,
}

impl AgentConfig {
    /// Create a new agent config with a model.
    pub fn new(model: Model) -> Self {
        Self {
            model,
            thinking_level: ThinkingLevel::default(),
            tool_execution: ToolExecutionMode::default(),
            security: crate::types::SecurityConfig::default(),
        }
    }
}

/// Tool result from execution.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentToolResult<T = serde_json::Value> {
    /// Content blocks.
    pub content: Vec<ContentBlock>,
    /// Additional details.
    pub details: Option<T>,
}

impl AgentToolResult {
    /// Create a text result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text(TextContent::new(text))],
            details: None,
        }
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text(TextContent::new(message))],
            details: None,
        }
    }
}

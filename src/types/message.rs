//! Message types for LLM conversations.

use crate::types::{ContentBlock, UserContent};
use serde::{Deserialize, Serialize};

/// Role of a message participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// User message.
    User,
    /// Assistant/AI message.
    Assistant,
    /// Tool result message.
    #[serde(rename = "toolResult")]
    ToolResult,
}

impl Default for Role {
    fn default() -> Self {
        Role::User
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::ToolResult => write!(f, "toolResult"),
        }
    }
}

/// Stop reason for assistant messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StopReason {
    /// Normal completion.
    Stop,
    /// Maximum token limit reached.
    Length,
    /// Tool call requested.
    #[serde(rename = "toolUse")]
    ToolUse,
    /// Error occurred.
    Error,
    /// Request was aborted.
    Aborted,
}

impl Default for StopReason {
    fn default() -> Self {
        StopReason::Stop
    }
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::Stop => write!(f, "stop"),
            StopReason::Length => write!(f, "length"),
            StopReason::ToolUse => write!(f, "toolUse"),
            StopReason::Error => write!(f, "error"),
            StopReason::Aborted => write!(f, "aborted"),
        }
    }
}

/// User message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserMessage {
    /// Role identifier.
    pub role: Role,
    /// Message content.
    pub content: UserContent,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

impl UserMessage {
    /// Create a new user message with text content.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: UserContent::Text(content.into()),
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Create a new user message with content blocks.
    pub fn blocks(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content: UserContent::Blocks(content),
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Create a new user message with the current timestamp.
    pub fn new(content: impl Into<UserContent>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }
}

/// Assistant message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantMessage {
    /// Role identifier.
    pub role: Role,
    /// Message content blocks.
    pub content: Vec<ContentBlock>,
    /// API type used.
    pub api: crate::types::Api,
    /// Provider name.
    pub provider: crate::types::Provider,
    /// Model identifier.
    pub model: String,
    /// Token usage information.
    pub usage: crate::types::Usage,
    /// Stop reason.
    pub stop_reason: StopReason,
    /// Error message if stop_reason is Error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Provider-specific upstream response or message identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

impl AssistantMessage {
    /// Create a new assistant message builder.
    pub fn builder() -> AssistantMessageBuilder {
        AssistantMessageBuilder::default()
    }

    /// Check if this message contains tool calls.
    pub fn has_tool_calls(&self) -> bool {
        self.content.iter().any(|b| b.is_tool_call())
    }

    /// Get all tool calls from this message.
    pub fn tool_calls(&self) -> Vec<&crate::types::ToolCall> {
        self.content
            .iter()
            .filter_map(|b| b.as_tool_call())
            .collect()
    }

    /// Get text content from this message.
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| b.as_text())
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get thinking content from this message.
    pub fn thinking_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| b.as_thinking())
            .map(|t| t.thinking.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Builder for AssistantMessage.
#[derive(Debug, Default)]
pub struct AssistantMessageBuilder {
    content: Vec<ContentBlock>,
    api: Option<crate::types::Api>,
    provider: Option<crate::types::Provider>,
    model: Option<String>,
    usage: crate::types::Usage,
    stop_reason: StopReason,
    error_message: Option<String>,
    response_id: Option<String>,
}

impl AssistantMessageBuilder {
    pub fn content(mut self, content: Vec<ContentBlock>) -> Self {
        self.content = content;
        self
    }

    pub fn add_content(mut self, block: ContentBlock) -> Self {
        self.content.push(block);
        self
    }

    pub fn api(mut self, api: crate::types::Api) -> Self {
        self.api = Some(api);
        self
    }

    pub fn provider(mut self, provider: crate::types::Provider) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn usage(mut self, usage: crate::types::Usage) -> Self {
        self.usage = usage;
        self
    }

    pub fn stop_reason(mut self, reason: StopReason) -> Self {
        self.stop_reason = reason;
        self
    }

    pub fn error_message(mut self, message: impl Into<String>) -> Self {
        self.error_message = Some(message.into());
        self
    }

    pub fn response_id(mut self, response_id: impl Into<String>) -> Self {
        self.response_id = Some(response_id.into());
        self
    }

    pub fn build(self) -> Result<AssistantMessage, String> {
        let api = self.api.ok_or("api is required")?;
        let provider = self.provider.ok_or("provider is required")?;
        let model = self.model.ok_or("model is required")?;

        Ok(AssistantMessage {
            role: Role::Assistant,
            content: self.content,
            api,
            provider,
            model,
            usage: self.usage,
            stop_reason: self.stop_reason,
            error_message: self.error_message,
            response_id: self.response_id,
            timestamp: chrono::Utc::now().timestamp_millis(),
        })
    }
}

/// Tool result message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultMessage<T = serde_json::Value> {
    /// Role identifier.
    pub role: Role,
    /// Tool call ID this result corresponds to.
    pub tool_call_id: String,
    /// Tool name.
    pub tool_name: String,
    /// Result content blocks.
    pub content: Vec<ContentBlock>,
    /// Additional details (for UI display or logging).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<T>,
    /// Whether this result represents an error.
    pub is_error: bool,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

impl ToolResultMessage {
    /// Create a new tool result message.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: Vec<ContentBlock>,
        is_error: bool,
    ) -> Self {
        Self {
            role: Role::ToolResult,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content,
            details: None,
            is_error,
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Create a tool result from text.
    pub fn text(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        text: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::new(
            tool_call_id,
            tool_name,
            vec![ContentBlock::Text(crate::types::TextContent::new(text))],
            is_error,
        )
    }

    /// Create an error tool result.
    pub fn error(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        error_message: impl Into<String>,
    ) -> Self {
        Self::text(tool_call_id, tool_name, error_message, true)
    }

    /// Get text content from this message.
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| b.as_text())
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Add details to this result.
    pub fn with_details<T>(self, details: T) -> ToolResultMessage<T> {
        ToolResultMessage {
            role: self.role,
            tool_call_id: self.tool_call_id,
            tool_name: self.tool_name,
            content: self.content,
            details: Some(details),
            is_error: self.is_error,
            timestamp: self.timestamp,
        }
    }
}

/// Message union type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// User message.
    #[serde(rename = "user")]
    User(UserMessage),
    /// Assistant message.
    #[serde(rename = "assistant")]
    Assistant(AssistantMessage),
    /// Tool result message.
    #[serde(rename = "toolResult")]
    ToolResult(ToolResultMessage),
}

impl Message {
    /// Get the role of this message.
    pub fn role(&self) -> Role {
        match self {
            Message::User(m) => m.role,
            Message::Assistant(m) => m.role,
            Message::ToolResult(m) => m.role,
        }
    }

    /// Get the timestamp of this message.
    pub fn timestamp(&self) -> i64 {
        match self {
            Message::User(m) => m.timestamp,
            Message::Assistant(m) => m.timestamp,
            Message::ToolResult(m) => m.timestamp,
        }
    }

    /// Check if this is a user message.
    pub fn is_user(&self) -> bool {
        matches!(self, Message::User(_))
    }

    /// Check if this is an assistant message.
    pub fn is_assistant(&self) -> bool {
        matches!(self, Message::Assistant(_))
    }

    /// Check if this is a tool result message.
    pub fn is_tool_result(&self) -> bool {
        matches!(self, Message::ToolResult(_))
    }
}

impl From<UserMessage> for Message {
    fn from(msg: UserMessage) -> Self {
        Message::User(msg)
    }
}

impl From<AssistantMessage> for Message {
    fn from(msg: AssistantMessage) -> Self {
        Message::Assistant(msg)
    }
}

impl From<ToolResultMessage> for Message {
    fn from(msg: ToolResultMessage) -> Self {
        Message::ToolResult(msg)
    }
}

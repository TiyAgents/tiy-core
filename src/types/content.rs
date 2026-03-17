//! Content block types for messages.

use serde::{Deserialize, Serialize};

/// Text content block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextContent {
    /// The actual text.
    pub text: String,
    /// Optional signature for OpenAI responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

impl Default for TextContent {
    fn default() -> Self {
        Self {
            text: String::new(),
            text_signature: None,
        }
    }
}

impl TextContent {
    /// Create a new text content.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            text_signature: None,
        }
    }
}

/// Thinking content block (Extended Thinking).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinkingContent {
    /// The thinking/reasoning content.
    pub thinking: String,
    /// Optional signature for thinking blocks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    /// Whether the thinking was redacted by safety filters.
    #[serde(default)]
    pub redacted: bool,
}

impl Default for ThinkingContent {
    fn default() -> Self {
        Self {
            thinking: String::new(),
            thinking_signature: None,
            redacted: false,
        }
    }
}

impl ThinkingContent {
    /// Create a new thinking content.
    pub fn new(thinking: impl Into<String>) -> Self {
        Self {
            thinking: thinking.into(),
            thinking_signature: None,
            redacted: false,
        }
    }
}

/// Image content block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageContent {
    /// Base64 encoded image data.
    pub data: String,
    /// MIME type of the image (e.g., "image/jpeg", "image/png").
    pub mime_type: String,
}

impl ImageContent {
    /// Create a new image content.
    pub fn new(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            mime_type: mime_type.into(),
        }
    }
}

/// Tool call content block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// Name of the tool to call.
    pub name: String,
    /// Arguments for the tool call as JSON.
    pub arguments: serde_json::Value,
    /// Optional thought signature (Google specific).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

impl ToolCall {
    /// Create a new tool call.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
            thought_signature: None,
        }
    }
}

/// Content block enum that can appear in messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ContentBlock {
    /// Text content.
    #[serde(rename = "text")]
    Text(TextContent),
    /// Thinking content.
    #[serde(rename = "thinking")]
    Thinking(ThinkingContent),
    /// Tool call.
    #[serde(rename = "toolCall")]
    ToolCall(ToolCall),
    /// Image content.
    #[serde(rename = "image")]
    Image(ImageContent),
}

impl ContentBlock {
    /// Check if this is a text block.
    pub fn is_text(&self) -> bool {
        matches!(self, ContentBlock::Text(_))
    }

    /// Check if this is a thinking block.
    pub fn is_thinking(&self) -> bool {
        matches!(self, ContentBlock::Thinking(_))
    }

    /// Check if this is a tool call block.
    pub fn is_tool_call(&self) -> bool {
        matches!(self, ContentBlock::ToolCall(_))
    }

    /// Check if this is an image block.
    pub fn is_image(&self) -> bool {
        matches!(self, ContentBlock::Image(_))
    }

    /// Get text content if this is a text block.
    pub fn as_text(&self) -> Option<&TextContent> {
        match self {
            ContentBlock::Text(text) => Some(text),
            _ => None,
        }
    }

    /// Get thinking content if this is a thinking block.
    pub fn as_thinking(&self) -> Option<&ThinkingContent> {
        match self {
            ContentBlock::Thinking(thinking) => Some(thinking),
            _ => None,
        }
    }

    /// Get tool call if this is a tool call block.
    pub fn as_tool_call(&self) -> Option<&ToolCall> {
        match self {
            ContentBlock::ToolCall(tool_call) => Some(tool_call),
            _ => None,
        }
    }

    /// Get image content if this is an image block.
    pub fn as_image(&self) -> Option<&ImageContent> {
        match self {
            ContentBlock::Image(image) => Some(image),
            _ => None,
        }
    }
}

impl From<TextContent> for ContentBlock {
    fn from(text: TextContent) -> Self {
        ContentBlock::Text(text)
    }
}

impl From<ThinkingContent> for ContentBlock {
    fn from(thinking: ThinkingContent) -> Self {
        ContentBlock::Thinking(thinking)
    }
}

impl From<ToolCall> for ContentBlock {
    fn from(tool_call: ToolCall) -> Self {
        ContentBlock::ToolCall(tool_call)
    }
}

impl From<ImageContent> for ContentBlock {
    fn from(image: ImageContent) -> Self {
        ContentBlock::Image(image)
    }
}

/// User content can be a simple string or an array of content blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    /// Simple text content.
    Text(String),
    /// Array of content blocks.
    Blocks(Vec<ContentBlock>),
}

impl UserContent {
    /// Create simple text content.
    pub fn text(text: impl Into<String>) -> Self {
        UserContent::Text(text.into())
    }

    /// Check if this is simple text.
    pub fn is_text(&self) -> bool {
        matches!(self, UserContent::Text(_))
    }

    /// Get text if this is simple text.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            UserContent::Text(text) => Some(text),
            _ => None,
        }
    }
}

impl From<String> for UserContent {
    fn from(text: String) -> Self {
        UserContent::Text(text)
    }
}

impl From<&str> for UserContent {
    fn from(text: &str) -> Self {
        UserContent::Text(text.to_string())
    }
}

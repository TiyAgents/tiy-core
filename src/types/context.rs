//! Context and Tool definitions.

use crate::types::Message;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

/// Conversation context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Context {
    /// System prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Available tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            system_prompt: None,
            messages: Vec::new(),
            tools: None,
        }
    }
}

impl Context {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a context with a system prompt.
    pub fn with_system_prompt(prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: Some(prompt.into()),
            messages: Vec::new(),
            tools: None,
        }
    }

    /// Add a message to the context.
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Add a user message to the context.
    pub fn user(&mut self, content: impl Into<String>) {
        self.messages
            .push(Message::User(crate::types::UserMessage::text(content)));
    }

    /// Set the system prompt.
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = Some(prompt.into());
    }

    /// Set the tools.
    pub fn set_tools(&mut self, tools: Vec<Tool>) {
        self.tools = Some(tools);
    }

    /// Get the last message.
    pub fn last_message(&self) -> Option<&Message> {
        self.messages.last()
    }

    /// Check if the context has any messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Get the number of messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Clear all messages.
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

/// Tool definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Parameters schema (JSON Schema).
    pub parameters: serde_json::Value,
}

impl Tool {
    /// Create a new tool.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    /// Create a tool builder.
    pub fn builder() -> ToolBuilder {
        ToolBuilder::default()
    }
}

/// Builder for Tool.
#[derive(Debug, Default)]
pub struct ToolBuilder {
    name: Option<String>,
    description: Option<String>,
    parameters: Option<serde_json::Value>,
}

impl ToolBuilder {
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn parameters(mut self, parameters: serde_json::Value) -> Self {
        self.parameters = Some(parameters);
        self
    }

    /// Set parameters from a type that implements JsonSchema.
    pub fn parameters_from_schema<T: JsonSchema>(mut self) -> Self {
        let schema = schemars::schema_for!(T);
        self.parameters = Some(serde_json::to_value(&schema).unwrap_or_default());
        self
    }

    pub fn build(self) -> Result<Tool, String> {
        let name = self.name.ok_or("name is required")?;
        let description = self.description.unwrap_or_default();
        let parameters = self
            .parameters
            .unwrap_or(serde_json::json!({"type": "object", "properties": {}}));

        Ok(Tool {
            name,
            description,
            parameters,
        })
    }
}

/// Stream options.
///
/// Preferred transport for LLM communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    /// Server-Sent Events (default).
    #[default]
    Sse,
    /// WebSocket connection.
    WebSocket,
    /// Provider chooses automatically.
    Auto,
}

/// Prompt cache retention preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
    /// Disable provider-side prompt caching.
    None,
    /// Use the provider's default short-lived cache retention.
    #[default]
    Short,
    /// Request the longest retention the provider supports.
    Long,
}

/// Generic tool choice mode supported by multiple providers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Provider-defined automatic tool selection.
    Mode(ToolChoiceMode),
    /// Force a specific named tool.
    Named(ToolChoiceNamed),
}

/// Tool choice modes shared by multiple providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    Auto,
    Any,
    None,
    Required,
}

/// Named tool selection payloads for provider-specific APIs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoiceNamed {
    Tool { name: String },
    Function { function: ToolChoiceFunction },
}

/// Named OpenAI-style function selector.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

/// OpenAI Responses service tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIServiceTier {
    Auto,
    Default,
    Flex,
    Priority,
}

/// Payload inspection / replacement hook.
///
/// Called with the serialized request body before it is sent to the provider.
/// Return `Some(modified)` to replace the payload, or `None` to keep the original.
pub type OnPayloadFn = Arc<
    dyn Fn(
            serde_json::Value,
            crate::types::Model,
        ) -> Pin<Box<dyn std::future::Future<Output = Option<serde_json::Value>> + Send>>
        + Send
        + Sync,
>;

/// Stream options.
#[derive(Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    /// Temperature for sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// API key override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Base URL override. When set, takes priority over model.base_url.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Custom headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<std::collections::HashMap<String, String>>,
    /// Session ID for caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Prompt cache retention preference. Providers map this to their native settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<CacheRetention>,
    /// Security and resource limits. When None, uses SecurityConfig::default().
    #[serde(skip_serializing_if = "Option::is_none")]
    pub security: Option<crate::types::SecurityConfig>,
    /// Payload inspection / replacement hook. Skipped during serialization.
    #[serde(skip)]
    pub on_payload: Option<OnPayloadFn>,
    /// Preferred transport for LLM communication.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport: Option<Transport>,
    /// Optional metadata to include in provider requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Optional tool choice preference for providers with explicit tool controls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Optional OpenAI Responses service tier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<OpenAIServiceTier>,
    /// Maximum retry delay in milliseconds. `None` = use provider default.
    /// Set to `Some(0)` to disable the cap entirely.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retry_delay_ms: Option<u64>,
}

/// Custom Debug implementation that redacts sensitive fields (api_key, headers).
impl std::fmt::Debug for StreamOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamOptions")
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("base_url", &self.base_url)
            .field(
                "headers",
                &self
                    .headers
                    .as_ref()
                    .map(|h| h.keys().cloned().collect::<Vec<_>>()),
            )
            .field("session_id", &self.session_id)
            .field("cache_retention", &self.cache_retention)
            .field(
                "security",
                &self.security.as_ref().map(|_| "SecurityConfig{...}"),
            )
            .field("on_payload", &self.on_payload.as_ref().map(|_| "<fn>"))
            .field("transport", &self.transport)
            .field("metadata", &self.metadata)
            .field("tool_choice", &self.tool_choice)
            .field("service_tier", &self.service_tier)
            .field("max_retry_delay_ms", &self.max_retry_delay_ms)
            .finish()
    }
}

impl PartialEq for StreamOptions {
    fn eq(&self, other: &Self) -> bool {
        // Compare all fields except on_payload (can't compare fn pointers)
        self.temperature == other.temperature
            && self.max_tokens == other.max_tokens
            && self.api_key == other.api_key
            && self.base_url == other.base_url
            && self.headers == other.headers
            && self.session_id == other.session_id
            && self.cache_retention == other.cache_retention
            && self.security == other.security
            && self.transport == other.transport
            && self.metadata == other.metadata
            && self.tool_choice == other.tool_choice
            && self.service_tier == other.service_tier
            && self.max_retry_delay_ms == other.max_retry_delay_ms
    }
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            temperature: None,
            max_tokens: None,
            api_key: None,
            base_url: None,
            headers: None,
            session_id: None,
            cache_retention: None,
            security: None,
            on_payload: None,
            transport: None,
            metadata: None,
            tool_choice: None,
            service_tier: None,
            max_retry_delay_ms: None,
        }
    }
}

impl StreamOptions {
    /// Get the effective security config (provided or default).
    pub fn security_config(&self) -> std::borrow::Cow<'_, crate::types::SecurityConfig> {
        match &self.security {
            Some(config) => std::borrow::Cow::Borrowed(config),
            None => std::borrow::Cow::Owned(crate::types::SecurityConfig::default()),
        }
    }
}

/// Simple stream options with thinking support.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct SimpleStreamOptions {
    /// Base stream options.
    #[serde(flatten)]
    pub base: StreamOptions,
    /// Thinking/reasoning level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<crate::thinking::ThinkingLevel>,
    /// Custom thinking budget tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<u32>,
}

/// Custom Debug implementation that delegates to StreamOptions (which redacts secrets).
impl std::fmt::Debug for SimpleStreamOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleStreamOptions")
            .field("base", &self.base)
            .field("reasoning", &self.reasoning)
            .field("thinking_budget_tokens", &self.thinking_budget_tokens)
            .finish()
    }
}

impl Default for SimpleStreamOptions {
    fn default() -> Self {
        Self {
            base: StreamOptions::default(),
            reasoning: None,
            thinking_budget_tokens: None,
        }
    }
}

impl From<StreamOptions> for SimpleStreamOptions {
    fn from(base: StreamOptions) -> Self {
        Self {
            base,
            reasoning: None,
            thinking_budget_tokens: None,
        }
    }
}

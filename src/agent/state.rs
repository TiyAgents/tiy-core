//! Agent state management.

use crate::thinking::ThinkingLevel;
use crate::types::Model;
use crate::agent::{AgentMessage, AgentTool};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};

/// Agent state.
#[derive(Debug)]
pub struct AgentState {
    /// System prompt.
    pub system_prompt: RwLock<String>,
    /// Current model.
    pub model: RwLock<Model>,
    /// Thinking level.
    pub thinking_level: RwLock<ThinkingLevel>,
    /// Available tools.
    pub tools: RwLock<Vec<AgentTool>>,
    /// Conversation messages.
    pub messages: RwLock<Vec<AgentMessage>>,
    /// Whether currently streaming.
    pub is_streaming: AtomicBool,
    /// Current streaming message.
    pub stream_message: RwLock<Option<AgentMessage>>,
    /// Pending tool call IDs.
    pub pending_tool_calls: RwLock<HashSet<String>>,
    /// Last error.
    pub error: RwLock<Option<String>>,
}

impl AgentState {
    /// Create a new agent state with default values.
    pub fn new() -> Self {
        Self {
            system_prompt: RwLock::new(String::new()),
            model: RwLock::new(Model::builder()
                .id("gpt-4o-mini")
                .name("GPT-4o Mini")
                .provider(crate::types::Provider::OpenAI)
                .base_url("https://api.openai.com/v1")
                .context_window(128000)
                .max_tokens(16384)
                .build()
                .unwrap()),
            thinking_level: RwLock::new(ThinkingLevel::Off),
            tools: RwLock::new(Vec::new()),
            messages: RwLock::new(Vec::new()),
            is_streaming: AtomicBool::new(false),
            stream_message: RwLock::new(None),
            pending_tool_calls: RwLock::new(HashSet::new()),
            error: RwLock::new(None),
        }
    }

    /// Create state with a model.
    pub fn with_model(model: Model) -> Self {
        let state = Self::new();
        *state.model.write() = model;
        state
    }

    /// Set the system prompt.
    pub fn set_system_prompt(&self, prompt: impl Into<String>) {
        *self.system_prompt.write() = prompt.into();
    }

    /// Set the model.
    pub fn set_model(&self, model: Model) {
        *self.model.write() = model;
    }

    /// Set the thinking level.
    pub fn set_thinking_level(&self, level: ThinkingLevel) {
        *self.thinking_level.write() = level;
    }

    /// Set the tools.
    pub fn set_tools(&self, tools: Vec<AgentTool>) {
        *self.tools.write() = tools;
    }

    /// Add a message.
    pub fn add_message(&self, message: AgentMessage) {
        self.messages.write().push(message);
    }

    /// Replace all messages.
    pub fn replace_messages(&self, messages: Vec<AgentMessage>) {
        *self.messages.write() = messages;
    }

    /// Clear all messages.
    pub fn clear_messages(&self) {
        self.messages.write().clear();
    }

    /// Reset the state.
    pub fn reset(&self) {
        *self.system_prompt.write() = String::new();
        *self.thinking_level.write() = ThinkingLevel::Off;
        *self.tools.write() = Vec::new();
        self.messages.write().clear();
        self.is_streaming.store(false, Ordering::SeqCst);
        *self.stream_message.write() = None;
        self.pending_tool_calls.write().clear();
        *self.error.write() = None;
    }

    /// Check if currently streaming.
    pub fn is_streaming(&self) -> bool {
        self.is_streaming.load(Ordering::SeqCst)
    }

    /// Set streaming state.
    pub fn set_streaming(&self, value: bool) {
        self.is_streaming.store(value, Ordering::SeqCst);
    }

    /// Get message count.
    pub fn message_count(&self) -> usize {
        self.messages.read().len()
    }
}

impl Default for AgentState {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for AgentState {
    fn clone(&self) -> Self {
        Self {
            system_prompt: RwLock::new(self.system_prompt.read().clone()),
            model: RwLock::new(self.model.read().clone()),
            thinking_level: RwLock::new(*self.thinking_level.read()),
            tools: RwLock::new(self.tools.read().clone()),
            messages: RwLock::new(self.messages.read().clone()),
            is_streaming: AtomicBool::new(self.is_streaming.load(Ordering::SeqCst)),
            stream_message: RwLock::new(self.stream_message.read().clone()),
            pending_tool_calls: RwLock::new(self.pending_tool_calls.read().clone()),
            error: RwLock::new(self.error.read().clone()),
        }
    }
}

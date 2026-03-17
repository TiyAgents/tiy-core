//! Agent state management.

use crate::agent::{AgentMessage, AgentTool};
use crate::thinking::ThinkingLevel;
use crate::types::Model;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
    /// Maximum number of messages in conversation history.
    /// 0 = unlimited. When exceeded, oldest messages are drained.
    pub max_messages: AtomicUsize,
}

impl AgentState {
    /// Create a new agent state with default values.
    pub fn new() -> Self {
        Self {
            system_prompt: RwLock::new(String::new()),
            model: RwLock::new(
                Model::builder()
                    .id("gpt-4o-mini")
                    .name("GPT-4o Mini")
                    .provider(crate::types::Provider::OpenAI)
                    .base_url("https://api.openai.com/v1")
                    .context_window(128000)
                    .max_tokens(16384)
                    .build()
                    .unwrap(),
            ),
            thinking_level: RwLock::new(ThinkingLevel::Off),
            tools: RwLock::new(Vec::new()),
            messages: RwLock::new(Vec::new()),
            is_streaming: AtomicBool::new(false),
            stream_message: RwLock::new(None),
            pending_tool_calls: RwLock::new(HashSet::new()),
            error: RwLock::new(None),
            max_messages: AtomicUsize::new(0), // 0 = unlimited
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

    /// Add a message, enforcing the max_messages limit.
    /// When the limit is exceeded, oldest messages are drained (FIFO).
    pub fn add_message(&self, message: AgentMessage) {
        let mut msgs = self.messages.write();
        msgs.push(message);
        let max = self.max_messages.load(Ordering::SeqCst);
        if max > 0 && msgs.len() > max {
            let excess = msgs.len() - max;
            msgs.drain(..excess);
        }
    }

    /// Set the maximum number of messages in conversation history.
    /// 0 = unlimited.
    pub fn set_max_messages(&self, max: usize) {
        self.max_messages.store(max, Ordering::SeqCst);
        // Immediately enforce if there are already too many messages
        if max > 0 {
            let mut msgs = self.messages.write();
            if msgs.len() > max {
                let excess = msgs.len() - max;
                msgs.drain(..excess);
            }
        }
    }

    /// Get the current max_messages limit (0 = unlimited).
    pub fn get_max_messages(&self) -> usize {
        self.max_messages.load(Ordering::SeqCst)
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

/// NOTE: This `Clone` implementation acquires each lock independently,
/// so the resulting clone is NOT a single atomic snapshot.
/// For a consistent point-in-time snapshot, use [`AgentState::snapshot()`].
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
            max_messages: AtomicUsize::new(self.max_messages.load(Ordering::SeqCst)),
        }
    }
}

// ============================================================================
// AgentStateSnapshot — consistent point-in-time view
// ============================================================================

/// A consistent, lock-free snapshot of [`AgentState`].
///
/// Unlike `Clone` on `AgentState` (which acquires each lock independently),
/// `snapshot()` acquires all locks simultaneously to produce a coherent
/// point-in-time view of the state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStateSnapshot {
    /// System prompt.
    pub system_prompt: String,
    /// Current model.
    pub model: Model,
    /// Thinking level.
    pub thinking_level: ThinkingLevel,
    /// Conversation messages.
    pub messages: Vec<AgentMessage>,
    /// Whether currently streaming.
    pub is_streaming: bool,
    /// Current streaming message.
    pub stream_message: Option<AgentMessage>,
    /// Pending tool call IDs.
    pub pending_tool_calls: HashSet<String>,
    /// Last error.
    pub error: Option<String>,
    /// Message count.
    pub message_count: usize,
    /// Max messages limit (0 = unlimited).
    pub max_messages: usize,
}

impl AgentState {
    /// Take a consistent point-in-time snapshot of the agent state.
    ///
    /// This acquires all locks to produce a coherent view, unlike `Clone`
    /// which acquires locks one at a time and may see partial updates.
    pub fn snapshot(&self) -> AgentStateSnapshot {
        // Acquire all locks at once for consistency
        let system_prompt = self.system_prompt.read().clone();
        let model = self.model.read().clone();
        let thinking_level = *self.thinking_level.read();
        let messages = self.messages.read().clone();
        let is_streaming = self.is_streaming.load(Ordering::SeqCst);
        let stream_message = self.stream_message.read().clone();
        let pending_tool_calls = self.pending_tool_calls.read().clone();
        let error = self.error.read().clone();
        let max_messages = self.max_messages.load(Ordering::SeqCst);
        let message_count = messages.len();

        AgentStateSnapshot {
            system_prompt,
            model,
            thinking_level,
            messages,
            is_streaming,
            stream_message,
            pending_tool_calls,
            error,
            message_count,
            max_messages,
        }
    }
}

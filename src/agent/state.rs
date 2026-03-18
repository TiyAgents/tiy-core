//! Agent state management.

use crate::agent::{AgentMessage, AgentTool};
use crate::thinking::ThinkingLevel;
use crate::types::Model;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Agent state — pure runtime state without configuration.
///
/// Configuration values (model, thinking_level) live in [`AgentConfig`]
/// (the single source of truth). This struct only holds conversational
/// and streaming runtime state.
#[derive(Debug)]
pub struct AgentState {
    /// System prompt.
    pub system_prompt: RwLock<String>,
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
            tools: RwLock::new(Vec::new()),
            messages: RwLock::new(Vec::new()),
            is_streaming: AtomicBool::new(false),
            stream_message: RwLock::new(None),
            pending_tool_calls: RwLock::new(HashSet::new()),
            error: RwLock::new(None),
            max_messages: AtomicUsize::new(0), // 0 = unlimited
        }
    }

    /// Set the system prompt.
    pub fn set_system_prompt(&self, prompt: impl Into<String>) {
        *self.system_prompt.write() = prompt.into();
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

    /// Reset the runtime state (messages, streaming, errors).
    ///
    /// Does NOT reset model or thinking_level (those live in `AgentConfig`).
    pub fn reset(&self) {
        *self.system_prompt.write() = String::new();
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
/// For a consistent point-in-time snapshot, use [`Agent::snapshot()`].
impl Clone for AgentState {
    fn clone(&self) -> Self {
        Self {
            system_prompt: RwLock::new(self.system_prompt.read().clone()),
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

/// A consistent, lock-free snapshot of the agent's full state.
///
/// Includes both runtime state (from [`AgentState`]) and configuration
/// (model, thinking_level from [`AgentConfig`]).
///
/// Obtain via [`Agent::snapshot()`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStateSnapshot {
    /// System prompt.
    pub system_prompt: String,
    /// Current model (from AgentConfig).
    pub model: Model,
    /// Thinking level (from AgentConfig).
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

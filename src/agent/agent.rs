//! Agent implementation.

use crate::agent::{AgentConfig, AgentContext, AgentEvent, AgentMessage, AgentState, AgentTool, ToolExecutionMode};
use crate::thinking::ThinkingLevel;
use crate::types::*;
use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::sync::Arc;

/// Agent for managing stateful conversations.
pub struct Agent {
    /// Agent state.
    state: Arc<AgentState>,
    /// Configuration.
    config: RwLock<AgentConfig>,
    /// Steering message queue.
    steering_queue: Mutex<VecDeque<AgentMessage>>,
    /// Follow-up message queue.
    follow_up_queue: Mutex<VecDeque<AgentMessage>>,
    /// Event subscribers.
    subscribers: RwLock<Vec<Box<dyn Fn(&AgentEvent) + Send + Sync>>>,
}

impl Agent {
    /// Create a new agent with default configuration.
    pub fn new() -> Self {
        Self {
            state: Arc::new(AgentState::new()),
            config: RwLock::new(AgentConfig::new(Model::builder()
                .id("gpt-4o-mini")
                .name("GPT-4o Mini")
                .api(Api::OpenAICompletions)
                .provider(Provider::OpenAI)
                .base_url("https://api.openai.com/v1")
                .context_window(128000)
                .max_tokens(16384)
                .build()
                .unwrap())),
            steering_queue: Mutex::new(VecDeque::new()),
            follow_up_queue: Mutex::new(VecDeque::new()),
            subscribers: RwLock::new(Vec::new()),
        }
    }

    /// Create an agent with a model.
    pub fn with_model(model: Model) -> Self {
        let agent = Self::new();
        agent.set_model(model.clone());
        *agent.config.write() = AgentConfig::new(model);
        agent
    }

    /// Subscribe to agent events.
    pub fn subscribe<F>(&self, callback: F) -> impl Fn()
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        let mut subscribers = self.subscribers.write();
        let index = subscribers.len();
        subscribers.push(Box::new(callback));
        drop(subscribers);

        let subs_ptr = Arc::new(&self.subscribers as *const RwLock<Vec<Box<dyn Fn(&AgentEvent) + Send + Sync>>>);

        move || {
            unsafe {
                let subs = &*(*subs_ptr);
                if let Some(mut guard) = subs.try_write() {
                    if index < guard.len() {
                        guard.remove(index);
                    }
                }
            }
        }
    }

    /// Emit an event to all subscribers.
    fn emit(&self, event: AgentEvent) {
        let subscribers = self.subscribers.read();
        for callback in subscribers.iter() {
            callback(&event);
        }
    }

    // State management

    /// Set the system prompt.
    pub fn set_system_prompt(&self, prompt: impl Into<String>) {
        self.state.set_system_prompt(prompt);
    }

    /// Set the model.
    pub fn set_model(&self, model: Model) {
        self.state.set_model(model.clone());
        self.config.write().model = model;
    }

    /// Set the thinking level.
    pub fn set_thinking_level(&self, level: ThinkingLevel) {
        self.state.set_thinking_level(level);
        self.config.write().thinking_level = level;
    }

    /// Set the tools.
    pub fn set_tools(&self, tools: Vec<AgentTool>) {
        self.state.set_tools(tools);
    }

    /// Set tool execution mode.
    pub fn set_tool_execution(&self, mode: ToolExecutionMode) {
        self.config.write().tool_execution = mode;
    }

    /// Replace all messages.
    pub fn replace_messages(&self, messages: Vec<AgentMessage>) {
        self.state.replace_messages(messages);
    }

    /// Append a message.
    pub fn append_message(&self, message: AgentMessage) {
        self.state.add_message(message);
    }

    /// Clear all messages.
    pub fn clear_messages(&self) {
        self.state.clear_messages();
    }

    /// Reset the agent.
    pub fn reset(&self) {
        self.state.reset();
        self.steering_queue.lock().clear();
        self.follow_up_queue.lock().clear();
    }

    // Steering and Follow-up

    /// Add a steering message (interrupts current work).
    pub fn steer(&self, message: AgentMessage) {
        self.steering_queue.lock().push_back(message);
    }

    /// Add a follow-up message (processed after current work completes).
    pub fn follow_up(&self, message: AgentMessage) {
        self.follow_up_queue.lock().push_back(message);
    }

    /// Clear steering queue.
    pub fn clear_steering_queue(&self) {
        self.steering_queue.lock().clear();
    }

    /// Clear follow-up queue.
    pub fn clear_follow_up_queue(&self) {
        self.follow_up_queue.lock().clear();
    }

    /// Clear all queues.
    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    /// Check if there are queued messages.
    pub fn has_queued_messages(&self) -> bool {
        !self.steering_queue.lock().is_empty() || !self.follow_up_queue.lock().is_empty()
    }

    /// Get steering messages.
    fn get_steering_messages(&self) -> Vec<AgentMessage> {
        self.steering_queue.lock().drain(..).collect()
    }

    /// Get follow-up messages.
    fn get_follow_up_messages(&self) -> Vec<AgentMessage> {
        self.follow_up_queue.lock().drain(..).collect()
    }

    // Prompt methods

    /// Send a prompt to the agent.
    pub async fn prompt(&self, message: impl Into<AgentMessage>) -> Result<Vec<AgentMessage>, AgentError> {
        if self.state.is_streaming() {
            return Err(AgentError::AlreadyStreaming);
        }

        let message = message.into();
        self.state.set_streaming(true);

        // Add message to state
        self.state.add_message(message.clone());

        // Emit events
        self.emit(AgentEvent::AgentStart);
        self.emit(AgentEvent::TurnStart);
        self.emit(AgentEvent::MessageStart { message: message.clone() });
        self.emit(AgentEvent::MessageEnd { message });

        // TODO: Actually run the agent loop
        // For now, just return the messages
        let messages = self.state.messages.read().clone();
        self.emit(AgentEvent::AgentEnd { messages: messages.clone() });
        self.state.set_streaming(false);

        Ok(messages)
    }

    /// Continue from current state.
    pub async fn continue_(&self) -> Result<Vec<AgentMessage>, AgentError> {
        if self.state.is_streaming() {
            return Err(AgentError::AlreadyStreaming);
        }

        // Check last message
        let messages = self.state.messages.read();
        if messages.is_empty() {
            return Err(AgentError::NoMessages);
        }

        if let Some(AgentMessage::Assistant(_)) = messages.last() {
            return Err(AgentError::CannotContinueFromAssistant);
        }
        drop(messages);

        self.state.set_streaming(true);

        // TODO: Actually run the agent loop
        let messages = self.state.messages.read().clone();
        self.state.set_streaming(false);

        Ok(messages)
    }

    /// Abort current operation.
    pub fn abort(&self) {
        self.state.set_streaming(false);
        self.clear_all_queues();
    }

    /// Wait for the agent to become idle.
    pub async fn wait_for_idle(&self) {
        while self.state.is_streaming() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    /// Get the current state.
    pub fn state(&self) -> &Arc<AgentState> {
        &self.state
    }
}

impl Default for Agent {
    fn default() -> Self {
        Self::new()
    }
}

/// Agent error type.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Agent is already streaming")]
    AlreadyStreaming,

    #[error("No messages in context")]
    NoMessages,

    #[error("Cannot continue from assistant message")]
    CannotContinueFromAssistant,

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Provider error: {0}")]
    ProviderError(String),

    #[error("{0}")]
    Other(String),
}

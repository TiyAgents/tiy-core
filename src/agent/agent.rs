//! Agent implementation with full conversation loop.

use crate::agent::{
    AgentConfig, AgentEvent, AgentMessage, AgentState, AgentTool, AgentToolResult,
    ToolExecutionMode,
};
use crate::provider::{get_provider, ArcProvider};
use crate::stream::AssistantMessageEventStream;
use crate::thinking::ThinkingLevel;
use crate::types::*;
use futures::StreamExt;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Default maximum number of turns (LLM calls) per prompt.
const DEFAULT_MAX_TURNS: usize = 25;

/// Tool executor function type.
pub type ToolExecutor = Arc<
    dyn Fn(
            &str,
            &str,
            &serde_json::Value,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AgentToolResult> + Send>>
        + Send
        + Sync,
>;

/// Subscriber ID for unsubscription.
pub type SubscriberId = u64;

/// Thread-safe subscriber storage using HashMap to avoid tombstone leaks.
struct Subscribers {
    callbacks: RwLock<HashMap<u64, Arc<dyn Fn(&AgentEvent) + Send + Sync>>>,
    next_id: AtomicU64,
}

impl Subscribers {
    fn new() -> Self {
        Self {
            callbacks: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        }
    }

    fn subscribe(&self, callback: Arc<dyn Fn(&AgentEvent) + Send + Sync>) -> SubscriberId {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.callbacks.write().insert(id, callback);
        id
    }

    fn unsubscribe(&self, id: SubscriberId) {
        self.callbacks.write().remove(&id);
    }

    /// Emit an event to all subscribers.
    /// Clones Arcs under read lock, then calls callbacks outside the lock
    /// to prevent blocking subscribe/unsubscribe operations.
    fn emit(&self, event: &AgentEvent) {
        let snapshot: Vec<Arc<dyn Fn(&AgentEvent) + Send + Sync>> =
            { self.callbacks.read().values().cloned().collect() };
        // Lock released — call callbacks without holding any lock.
        for cb in &snapshot {
            cb(event);
        }
    }
}

/// Agent for managing stateful conversations with LLM providers.
pub struct Agent {
    /// Agent state.
    state: Arc<AgentState>,
    /// Configuration.
    config: RwLock<AgentConfig>,
    /// Provider (optional, resolved from registry if not set).
    provider: RwLock<Option<ArcProvider>>,
    /// Tool executor callback.
    tool_executor: RwLock<Option<ToolExecutor>>,
    /// Maximum turns per prompt.
    max_turns: RwLock<usize>,
    /// Steering message queue.
    steering_queue: Mutex<VecDeque<AgentMessage>>,
    /// Follow-up message queue.
    follow_up_queue: Mutex<VecDeque<AgentMessage>>,
    /// Event subscribers (M7: HashMap-based, no tombstone leak).
    subscribers: Arc<Subscribers>,
    /// Abort flag.
    abort_flag: Arc<AtomicBool>,
    /// API key for the provider.
    api_key: RwLock<Option<String>>,
}

impl Agent {
    /// Create a new agent with default configuration.
    pub fn new() -> Self {
        Self {
            state: Arc::new(AgentState::new()),
            config: RwLock::new(AgentConfig::new(
                Model::builder()
                    .id("gpt-4o-mini")
                    .name("GPT-4o Mini")
                    .provider(Provider::OpenAI)
                    .base_url("https://api.openai.com/v1")
                    .context_window(128000)
                    .max_tokens(16384)
                    .build()
                    .unwrap(),
            )),
            provider: RwLock::new(None),
            tool_executor: RwLock::new(None),
            max_turns: RwLock::new(DEFAULT_MAX_TURNS),
            steering_queue: Mutex::new(VecDeque::new()),
            follow_up_queue: Mutex::new(VecDeque::new()),
            subscribers: Arc::new(Subscribers::new()),
            abort_flag: Arc::new(AtomicBool::new(false)),
            api_key: RwLock::new(None),
        }
    }

    /// Create an agent with a model.
    pub fn with_model(model: Model) -> Self {
        let agent = Self::new();
        agent.set_model(model.clone());
        *agent.config.write() = AgentConfig::new(model);
        agent
    }

    /// Set the LLM provider explicitly.
    pub fn set_provider(&self, provider: ArcProvider) {
        *self.provider.write() = Some(provider);
    }

    /// Set the API key.
    pub fn set_api_key(&self, key: impl Into<String>) {
        *self.api_key.write() = Some(key.into());
    }

    /// Set the tool executor callback.
    ///
    /// The executor receives (tool_name, tool_call_id, arguments) and returns an AgentToolResult.
    pub fn set_tool_executor<F, Fut>(&self, executor: F)
    where
        F: Fn(&str, &str, &serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = AgentToolResult> + Send + 'static,
    {
        let executor = Arc::new(move |name: &str, id: &str, args: &serde_json::Value| {
            let fut = executor(name, id, args);
            Box::pin(fut)
                as std::pin::Pin<Box<dyn std::future::Future<Output = AgentToolResult> + Send>>
        });
        *self.tool_executor.write() = Some(executor);
    }

    /// Set maximum turns per prompt.
    pub fn set_max_turns(&self, max: usize) {
        *self.max_turns.write() = max;
    }

    /// Set the security configuration.
    pub fn set_security_config(&self, config: crate::types::SecurityConfig) {
        self.config.write().security = config;
    }

    /// Get the current security configuration.
    pub fn security_config(&self) -> crate::types::SecurityConfig {
        self.config.read().security.clone()
    }

    /// Subscribe to agent events. Returns an unsubscribe closure.
    pub fn subscribe<F>(&self, callback: F) -> impl Fn()
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        let id = self.subscribers.subscribe(Arc::new(callback));
        let subs = Arc::clone(&self.subscribers);
        move || {
            subs.unsubscribe(id);
        }
    }

    /// Emit an event to all subscribers (M2: lock-free callback invocation).
    fn emit(&self, event: AgentEvent) {
        self.subscribers.emit(&event);
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

    /// Get steering messages (drains the queue).
    pub(crate) fn get_steering_messages(&self) -> Vec<AgentMessage> {
        self.steering_queue.lock().drain(..).collect()
    }

    /// Get follow-up messages (drains the queue).
    pub(crate) fn get_follow_up_messages(&self) -> Vec<AgentMessage> {
        self.follow_up_queue.lock().drain(..).collect()
    }

    // ============================================================================
    // Core Agent Loop
    // ============================================================================

    /// Build the context from current agent state.
    fn build_context(&self) -> Context {
        let system_prompt = self.state.system_prompt.read().clone();
        let messages = self.state.messages.read().clone();
        let tools = self.state.tools.read().clone();

        let mut context = if system_prompt.is_empty() {
            Context::new()
        } else {
            Context::with_system_prompt(&system_prompt)
        };

        // Convert AgentMessages to Messages
        for msg in &messages {
            match msg {
                AgentMessage::User(m) => context.add_message(Message::User(m.clone())),
                AgentMessage::Assistant(m) => context.add_message(Message::Assistant(m.clone())),
                AgentMessage::ToolResult(m) => context.add_message(Message::ToolResult(m.clone())),
            }
        }

        // Add tools
        if !tools.is_empty() {
            let tool_defs: Vec<Tool> = tools.iter().map(|t| t.as_tool()).collect();
            context.set_tools(tool_defs);
        }

        context
    }

    /// Resolve the provider to use.
    fn resolve_provider(&self) -> Result<ArcProvider, AgentError> {
        // First check explicit provider
        if let Some(ref provider) = *self.provider.read() {
            return Ok(provider.clone());
        }

        // Then try registry by Provider type
        let model = self.config.read().model.clone();
        if let Some(provider) = get_provider(&model.provider) {
            return Ok(provider);
        }

        Err(AgentError::ProviderError(format!(
            "No provider registered for provider type: {}",
            model.provider.as_str()
        )))
    }

    /// Build stream options.
    fn build_stream_options(&self) -> StreamOptions {
        let security = self.config.read().security.clone();
        StreamOptions {
            api_key: self.api_key.read().clone(),
            security: Some(security),
            ..Default::default()
        }
    }

    /// Run a single LLM turn: call provider, consume stream, return AssistantMessage.
    async fn run_turn(&self, provider: &ArcProvider) -> Result<AssistantMessage, AgentError> {
        let context = self.build_context();
        let model = self.config.read().model.clone();
        let options = self.build_stream_options();
        let stream_timeout = self.config.read().security.stream.result_timeout();

        // Call provider to create a stream
        let mut stream: AssistantMessageEventStream = provider.stream(&model, &context, options);

        // Process stream events
        while let Some(event) = stream.next().await {
            // Check abort
            if self.abort_flag.load(Ordering::SeqCst) {
                return Err(AgentError::Other("Aborted".to_string()));
            }

            // Check for steering messages
            let steering = self.get_steering_messages();
            if !steering.is_empty() {
                // Apply steering: add steering messages to state
                for steer_msg in steering {
                    self.state.add_message(steer_msg);
                }
                // Abort current turn and restart
                return Err(AgentError::Other("Steered".to_string()));
            }

            // Forward stream event to subscribers
            match &event {
                AssistantMessageEvent::Start { partial } => {
                    *self.state.stream_message.write() =
                        Some(AgentMessage::Assistant(partial.clone()));
                    self.emit(AgentEvent::MessageUpdate {
                        message: AgentMessage::Assistant(partial.clone()),
                        assistant_event: event.clone(),
                    });
                }
                AssistantMessageEvent::TextDelta { .. }
                | AssistantMessageEvent::ThinkingDelta { .. }
                | AssistantMessageEvent::ToolCallDelta { .. } => {
                    if let Some(partial) = event.partial_message() {
                        *self.state.stream_message.write() =
                            Some(AgentMessage::Assistant(partial.clone()));
                        self.emit(AgentEvent::MessageUpdate {
                            message: AgentMessage::Assistant(partial.clone()),
                            assistant_event: event.clone(),
                        });
                    }
                }
                _ => {
                    // Forward all other events too
                    if let Some(partial) = event.partial_message() {
                        self.emit(AgentEvent::MessageUpdate {
                            message: AgentMessage::Assistant(partial.clone()),
                            assistant_event: event.clone(),
                        });
                    }
                }
            }
        }

        // Get the final result (H4: with timeout to prevent infinite blocking)
        let result = match stream.try_result(stream_timeout).await {
            Some(r) => r,
            None => {
                return Err(AgentError::Other(format!(
                    "Stream result timed out after {:?}",
                    stream_timeout
                )));
            }
        };

        // Clear streaming message
        *self.state.stream_message.write() = None;

        if result.stop_reason == StopReason::Error {
            let error_msg = result
                .error_message
                .clone()
                .unwrap_or_else(|| "Unknown error".to_string());
            return Err(AgentError::ProviderError(error_msg));
        }

        Ok(result)
    }

    /// Execute tool calls from an assistant message.
    ///
    /// Includes C5 (tool validation), H3 (bounded parallel exec + timeout),
    /// and M3 (abort-aware execution).
    async fn execute_tool_calls(&self, assistant_msg: &AssistantMessage) -> Vec<ToolResultMessage> {
        let tool_calls = assistant_msg.tool_calls();
        if tool_calls.is_empty() {
            return Vec::new();
        }

        let executor = self.tool_executor.read().clone();
        let execution_mode = self.config.read().tool_execution;
        let security = self.config.read().security.clone();
        let tool_timeout = security.agent.tool_execution_timeout();

        // C5: Build Tool list for validation
        let agent_tools = self.state.tools.read().clone();
        let tool_defs: Vec<Tool> = agent_tools.iter().map(|t| t.as_tool()).collect();

        let mut results = Vec::new();

        match execution_mode {
            ToolExecutionMode::Parallel => {
                let max_parallel = security.agent.max_parallel_tool_calls;
                let abort_flag = Arc::clone(&self.abort_flag);

                // Collect validated tool calls as futures
                let mut tool_futures = Vec::new();

                for tc in &tool_calls {
                    let tc_id = tc.id.clone();
                    let tc_name = tc.name.clone();
                    let tc_args = tc.arguments.clone();

                    // Emit tool execution start
                    self.emit(AgentEvent::ToolExecutionStart {
                        tool_call_id: tc_id.clone(),
                        tool_name: tc_name.clone(),
                        args: tc_args.clone(),
                    });

                    // Track pending
                    self.state.pending_tool_calls.write().insert(tc_id.clone());

                    // C5: Validate tool call before execution
                    if security.agent.validate_tool_calls && !tool_defs.is_empty() {
                        let tc_ref = ToolCall::new(&tc_id, &tc_name, tc_args.clone());
                        if let Err(validation_err) =
                            crate::validation::validate_tool_call(&tool_defs, &tc_ref)
                        {
                            let error_msg = format!("Tool validation failed: {}", validation_err);
                            tracing::warn!(tool = %tc_name, error = %error_msg, "Tool call validation failed");
                            self.emit(AgentEvent::ToolExecutionEnd {
                                tool_call_id: tc_id.clone(),
                                tool_name: tc_name.clone(),
                                result: serde_json::json!({"error": error_msg}),
                                is_error: true,
                            });
                            self.state.pending_tool_calls.write().remove(&tc_id);
                            results.push(ToolResultMessage::new(
                                tc_id,
                                tc_name,
                                vec![ContentBlock::Text(TextContent::new(&error_msg))],
                                true,
                            ));
                            continue;
                        }
                    }

                    let executor = executor.clone();
                    let abort = abort_flag.clone();
                    tool_futures.push(async move {
                        let result = if let Some(ref exec) = executor {
                            // M3 + H3: timeout and abort-aware execution
                            tokio::select! {
                                res = tokio::time::timeout(tool_timeout, exec(&tc_name, &tc_id, &tc_args)) => {
                                    match res {
                                        Ok(r) => r,
                                        Err(_) => AgentToolResult::error(
                                            format!("Tool '{}' timed out after {:?}", tc_name, tool_timeout)
                                        ),
                                    }
                                }
                                _ = wait_for_abort(abort) => {
                                    AgentToolResult::error(format!("Tool '{}' aborted", tc_name))
                                }
                            }
                        } else {
                            AgentToolResult::error(format!("No tool executor registered for '{}'", tc_name))
                        };
                        (tc_id, tc_name, result)
                    });
                }

                // H3: Use buffer_unordered for bounded parallel execution
                let mut buffered =
                    futures::stream::iter(tool_futures).buffer_unordered(max_parallel);

                while let Some((tc_id, tc_name, result)) = buffered.next().await {
                    let is_error = result.content.iter().any(|b| {
                        b.as_text()
                            .map(|t| {
                                t.text.starts_with("No tool executor")
                                    || t.text.contains("timed out")
                                    || t.text.contains("aborted")
                                    || t.text.starts_with("Tool validation failed")
                            })
                            .unwrap_or(false)
                    });

                    let result_json =
                        serde_json::to_value(&result.content).unwrap_or(serde_json::Value::Null);
                    self.emit(AgentEvent::ToolExecutionEnd {
                        tool_call_id: tc_id.clone(),
                        tool_name: tc_name.clone(),
                        result: result_json,
                        is_error,
                    });

                    self.state.pending_tool_calls.write().remove(&tc_id);

                    results.push(ToolResultMessage::new(
                        tc_id,
                        tc_name,
                        result.content,
                        is_error,
                    ));
                }
            }
            ToolExecutionMode::Sequential => {
                for tc in &tool_calls {
                    if self.abort_flag.load(Ordering::SeqCst) {
                        break;
                    }

                    let tc_id = tc.id.clone();
                    let tc_name = tc.name.clone();
                    let tc_args = tc.arguments.clone();

                    self.emit(AgentEvent::ToolExecutionStart {
                        tool_call_id: tc_id.clone(),
                        tool_name: tc_name.clone(),
                        args: tc_args.clone(),
                    });

                    self.state.pending_tool_calls.write().insert(tc_id.clone());

                    // C5: Validate tool call before execution
                    if security.agent.validate_tool_calls && !tool_defs.is_empty() {
                        let tc_ref = ToolCall::new(&tc_id, &tc_name, tc_args.clone());
                        if let Err(validation_err) =
                            crate::validation::validate_tool_call(&tool_defs, &tc_ref)
                        {
                            let error_msg = format!("Tool validation failed: {}", validation_err);
                            tracing::warn!(tool = %tc_name, error = %error_msg, "Tool call validation failed");
                            self.emit(AgentEvent::ToolExecutionEnd {
                                tool_call_id: tc_id.clone(),
                                tool_name: tc_name.clone(),
                                result: serde_json::json!({"error": error_msg}),
                                is_error: true,
                            });
                            self.state.pending_tool_calls.write().remove(&tc_id);
                            results.push(ToolResultMessage::new(
                                tc_id,
                                tc_name,
                                vec![ContentBlock::Text(TextContent::new(&error_msg))],
                                true,
                            ));
                            continue;
                        }
                    }

                    // H3 + M3: timeout and abort-aware execution
                    let abort_flag = Arc::clone(&self.abort_flag);
                    let result = if let Some(ref exec) = executor {
                        tokio::select! {
                            res = tokio::time::timeout(tool_timeout, exec(&tc_name, &tc_id, &tc_args)) => {
                                match res {
                                    Ok(r) => r,
                                    Err(_) => AgentToolResult::error(
                                        format!("Tool '{}' timed out after {:?}", tc_name, tool_timeout)
                                    ),
                                }
                            }
                            _ = wait_for_abort(abort_flag) => {
                                AgentToolResult::error(format!("Tool '{}' aborted", tc_name))
                            }
                        }
                    } else {
                        AgentToolResult::error(format!(
                            "No tool executor registered for '{}'",
                            tc_name
                        ))
                    };

                    let is_error = result.content.iter().any(|b| {
                        b.as_text()
                            .map(|t| {
                                t.text.starts_with("No tool executor")
                                    || t.text.contains("timed out")
                                    || t.text.contains("aborted")
                                    || t.text.starts_with("Tool validation failed")
                            })
                            .unwrap_or(false)
                    });

                    let result_json =
                        serde_json::to_value(&result.content).unwrap_or(serde_json::Value::Null);
                    self.emit(AgentEvent::ToolExecutionEnd {
                        tool_call_id: tc_id.clone(),
                        tool_name: tc_name.clone(),
                        result: result_json,
                        is_error,
                    });

                    self.state.pending_tool_calls.write().remove(&tc_id);

                    results.push(ToolResultMessage::new(
                        tc_id,
                        tc_name,
                        result.content,
                        is_error,
                    ));
                }
            }
        }

        results
    }

    /// Run the agent loop: stream LLM → check tool calls → execute → loop.
    async fn run_loop(&self) -> Result<Vec<AgentMessage>, AgentError> {
        let provider = self.resolve_provider()?;
        let max_turns = *self.max_turns.read();
        let mut new_messages = Vec::new();
        let mut turn_count = 0;

        // M1: Sync message limit from security config
        let max_messages = self.config.read().security.agent.max_messages;
        self.state.set_max_messages(max_messages);

        loop {
            // Check abort
            if self.abort_flag.load(Ordering::SeqCst) {
                self.emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                });
                return Err(AgentError::Other("Aborted".to_string()));
            }

            // Check max turns
            if turn_count >= max_turns {
                break;
            }

            self.emit(AgentEvent::TurnStart);

            // Run one LLM turn
            let assistant_result = self.run_turn(&provider).await;

            match assistant_result {
                Ok(assistant_msg) => {
                    // Add assistant message to state and new_messages
                    let agent_msg = AgentMessage::Assistant(assistant_msg.clone());
                    self.state.add_message(agent_msg.clone());
                    new_messages.push(agent_msg.clone());

                    self.emit(AgentEvent::MessageStart {
                        message: agent_msg.clone(),
                    });
                    self.emit(AgentEvent::MessageEnd {
                        message: agent_msg.clone(),
                    });

                    // Check if there are tool calls
                    if assistant_msg.has_tool_calls()
                        && assistant_msg.stop_reason == StopReason::ToolUse
                    {
                        // Execute tools
                        let tool_results = self.execute_tool_calls(&assistant_msg).await;

                        // Add tool results to state and new_messages
                        for result in &tool_results {
                            let result_msg = AgentMessage::ToolResult(result.clone());
                            self.state.add_message(result_msg.clone());
                            new_messages.push(result_msg);
                        }

                        self.emit(AgentEvent::TurnEnd {
                            message: agent_msg,
                            tool_results,
                        });

                        // Check for follow-up messages
                        let follow_ups = self.get_follow_up_messages();
                        for msg in follow_ups {
                            self.state.add_message(msg.clone());
                            new_messages.push(msg);
                        }

                        turn_count += 1;
                        continue;
                    } else {
                        // No tool calls — conversation turn is complete
                        self.emit(AgentEvent::TurnEnd {
                            message: agent_msg,
                            tool_results: Vec::new(),
                        });

                        // Check for follow-up messages
                        let follow_ups = self.get_follow_up_messages();
                        if !follow_ups.is_empty() {
                            for msg in follow_ups {
                                self.state.add_message(msg.clone());
                                new_messages.push(msg);
                            }
                            turn_count += 1;
                            continue;
                        }

                        break;
                    }
                }
                Err(AgentError::Other(ref msg)) if msg == "Steered" => {
                    turn_count += 1;
                    continue;
                }
                Err(e) => {
                    *self.state.error.write() = Some(e.to_string());
                    return Err(e);
                }
            }
        }

        Ok(new_messages)
    }

    // ============================================================================
    // Prompt methods
    // ============================================================================

    /// Send a prompt to the agent.
    ///
    /// C4: Uses atomic compare_exchange to prevent TOCTOU race condition.
    pub async fn prompt(
        &self,
        message: impl Into<AgentMessage>,
    ) -> Result<Vec<AgentMessage>, AgentError> {
        // Atomic CAS: only one caller wins the race
        if self
            .state
            .is_streaming
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(AgentError::AlreadyStreaming);
        }

        let message = message.into();
        self.abort_flag.store(false, Ordering::SeqCst);

        // Add user message to state
        self.state.add_message(message.clone());

        // Emit start event
        self.emit(AgentEvent::AgentStart);

        // Run the agent loop
        let result = self.run_loop().await;

        self.state.set_streaming(false);

        match result {
            Ok(messages) => {
                self.emit(AgentEvent::AgentEnd {
                    messages: messages.clone(),
                });
                Ok(messages)
            }
            Err(e) => {
                self.emit(AgentEvent::AgentEnd {
                    messages: Vec::new(),
                });
                Err(e)
            }
        }
    }

    /// Continue from current state (e.g., after adding tool results externally).
    ///
    /// C4: Uses atomic compare_exchange to prevent TOCTOU race condition.
    pub async fn continue_(&self) -> Result<Vec<AgentMessage>, AgentError> {
        // Atomic CAS first: only one caller wins the race (C4 TOCTOU fix)
        if self
            .state
            .is_streaming
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(AgentError::AlreadyStreaming);
        }

        // Check messages (release streaming flag on early return)
        {
            let messages = self.state.messages.read();
            if messages.is_empty() {
                self.state.set_streaming(false);
                return Err(AgentError::NoMessages);
            }
            if let Some(AgentMessage::Assistant(_)) = messages.last() {
                self.state.set_streaming(false);
                return Err(AgentError::CannotContinueFromAssistant);
            }
        }

        self.abort_flag.store(false, Ordering::SeqCst);

        self.emit(AgentEvent::AgentStart);

        let result = self.run_loop().await;

        self.state.set_streaming(false);

        match result {
            Ok(messages) => {
                self.emit(AgentEvent::AgentEnd {
                    messages: messages.clone(),
                });
                Ok(messages)
            }
            Err(e) => {
                self.emit(AgentEvent::AgentEnd {
                    messages: Vec::new(),
                });
                Err(e)
            }
        }
    }

    /// Abort current operation.
    pub fn abort(&self) {
        self.abort_flag.store(true, Ordering::SeqCst);
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

/// Helper: wait until the abort flag is set.
async fn wait_for_abort(flag: Arc<AtomicBool>) {
    loop {
        if flag.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
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

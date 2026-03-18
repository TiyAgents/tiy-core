//! Agent implementation with full conversation loop.

use crate::agent::{
    AfterToolCallContext, AfterToolCallFn, AgentConfig, AgentEvent, AgentHooks, AgentMessage,
    AgentState, AgentStateSnapshot, AgentTool, AgentToolResult, BeforeToolCallContext,
    BeforeToolCallFn, BeforeToolCallResult, QueueMode, ThinkingBudgets, ToolExecutionMode,
    ToolExecutor, ToolUpdateCallback, Transport,
};
use crate::provider::{get_provider, ArcProtocol};
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

/// Subscriber ID for unsubscription.
pub type SubscriberId = u64;

/// Callback type for event subscribers.
type SubscriberCallback = Arc<dyn Fn(&AgentEvent) + Send + Sync>;

/// Thread-safe subscriber storage using HashMap to avoid tombstone leaks.
struct Subscribers {
    callbacks: RwLock<HashMap<u64, SubscriberCallback>>,
    next_id: AtomicU64,
}

impl Subscribers {
    fn new() -> Self {
        Self {
            callbacks: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        }
    }

    fn subscribe(&self, callback: SubscriberCallback) -> SubscriberId {
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
        let snapshot: Vec<SubscriberCallback> =
            { self.callbacks.read().values().cloned().collect() };
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
    provider: RwLock<Option<ArcProtocol>>,
    /// Aggregated hooks (tool executor, before/after hooks, converters, etc.).
    hooks: RwLock<AgentHooks>,
    /// Maximum turns per prompt.
    max_turns: RwLock<usize>,
    /// Steering message queue.
    steering_queue: Mutex<VecDeque<AgentMessage>>,
    /// Follow-up message queue.
    follow_up_queue: Mutex<VecDeque<AgentMessage>>,
    /// Event subscribers (HashMap-based, no tombstone leak).
    subscribers: Arc<Subscribers>,
    /// Abort flag.
    abort_flag: Arc<AtomicBool>,
    /// API key for the provider.
    api_key: RwLock<Option<String>>,
    /// Session ID for caching.
    session_id: RwLock<Option<String>>,
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
            hooks: RwLock::new(AgentHooks::default()),
            max_turns: RwLock::new(DEFAULT_MAX_TURNS),
            steering_queue: Mutex::new(VecDeque::new()),
            follow_up_queue: Mutex::new(VecDeque::new()),
            subscribers: Arc::new(Subscribers::new()),
            abort_flag: Arc::new(AtomicBool::new(false)),
            api_key: RwLock::new(None),
            session_id: RwLock::new(None),
        }
    }

    /// Create an agent with a model.
    pub fn with_model(model: Model) -> Self {
        let agent = Self::new();
        agent.set_model(model.clone());
        *agent.config.write() = AgentConfig::new(model);
        agent
    }

    // ============================================================================
    // Provider & API Key
    // ============================================================================

    /// Set the LLM provider explicitly.
    pub fn set_provider(&self, provider: ArcProtocol) {
        *self.provider.write() = Some(provider);
    }

    /// Set a static API key.
    pub fn set_api_key(&self, key: impl Into<String>) {
        *self.api_key.write() = Some(key.into());
    }

    /// Set a dynamic API key resolver.
    ///
    /// Called before each LLM request. Useful for short-lived OAuth tokens
    /// that may expire during long-running tool execution phases.
    pub fn set_get_api_key<F, Fut>(&self, resolver: F)
    where
        F: Fn(&str) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Option<String>> + Send + 'static,
    {
        let resolver = Arc::new(move |provider: &str| {
            let fut = resolver(provider);
            Box::pin(fut)
                as std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>
        });
        self.hooks.write().get_api_key = Some(resolver);
    }

    // ============================================================================
    // Tool Executor & Hooks
    // ============================================================================

    /// Set the tool executor callback.
    ///
    /// The executor receives `(tool_name, tool_call_id, arguments, update_callback)`.
    /// The `update_callback` can be called during execution to push streaming
    /// partial results (emitted as `ToolExecutionUpdate` events).
    pub fn set_tool_executor<F, Fut>(&self, executor: F)
    where
        F: Fn(&str, &str, &serde_json::Value, Option<ToolUpdateCallback>) -> Fut
            + Send
            + Sync
            + 'static,
        Fut: std::future::Future<Output = AgentToolResult> + Send + 'static,
    {
        let executor = Arc::new(
            move |name: &str,
                  id: &str,
                  args: &serde_json::Value,
                  update_cb: Option<ToolUpdateCallback>| {
                let fut = executor(name, id, args, update_cb);
                Box::pin(fut)
                    as std::pin::Pin<Box<dyn std::future::Future<Output = AgentToolResult> + Send>>
            },
        );
        self.hooks.write().tool_executor = Some(executor);
    }

    /// Set the tool executor callback (simple version without update callback).
    ///
    /// Convenience method for tools that don't need streaming updates.
    pub fn set_tool_executor_simple<F, Fut>(&self, executor: F)
    where
        F: Fn(&str, &str, &serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = AgentToolResult> + Send + 'static,
    {
        let executor = Arc::new(
            move |name: &str,
                  id: &str,
                  args: &serde_json::Value,
                  _update_cb: Option<ToolUpdateCallback>| {
                let fut = executor(name, id, args);
                Box::pin(fut)
                    as std::pin::Pin<Box<dyn std::future::Future<Output = AgentToolResult> + Send>>
            },
        );
        self.hooks.write().tool_executor = Some(executor);
    }

    /// Set the `before_tool_call` hook.
    ///
    /// Called after arguments are validated but before tool execution.
    /// Return `BeforeToolCallResult { block: true, .. }` to prevent execution.
    pub fn set_before_tool_call<F, Fut>(&self, hook: F)
    where
        F: Fn(BeforeToolCallContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Option<BeforeToolCallResult>> + Send + 'static,
    {
        let hook = Arc::new(move |ctx: BeforeToolCallContext| {
            let fut = hook(ctx);
            Box::pin(fut)
                as std::pin::Pin<
                    Box<dyn std::future::Future<Output = Option<BeforeToolCallResult>> + Send>,
                >
        });
        self.hooks.write().before_tool_call = Some(hook);
    }

    /// Set the `after_tool_call` hook.
    ///
    /// Called after tool execution, before the result is committed.
    /// Return `AfterToolCallResult` to override content, details, or is_error.
    pub fn set_after_tool_call<F, Fut>(&self, hook: F)
    where
        F: Fn(AfterToolCallContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Option<crate::agent::AfterToolCallResult>> + Send + 'static,
    {
        let hook = Arc::new(move |ctx: AfterToolCallContext| {
            let fut = hook(ctx);
            Box::pin(fut)
                as std::pin::Pin<
                    Box<
                        dyn std::future::Future<
                                Output = Option<crate::agent::AfterToolCallResult>,
                            > + Send,
                    >,
                >
        });
        self.hooks.write().after_tool_call = Some(hook);
    }

    // ============================================================================
    // Context Pipeline
    // ============================================================================

    /// Set the custom `AgentMessage[]` → `Message[]` conversion function.
    ///
    /// Called before each LLM request. The default filters out `Custom` messages
    /// and maps User/Assistant/ToolResult directly.
    pub fn set_convert_to_llm<F, Fut>(&self, converter: F)
    where
        F: Fn(Vec<AgentMessage>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<Message>> + Send + 'static,
    {
        let converter = Arc::new(move |msgs: Vec<AgentMessage>| {
            let fut = converter(msgs);
            Box::pin(fut)
                as std::pin::Pin<Box<dyn std::future::Future<Output = Vec<Message>> + Send>>
        });
        self.hooks.write().convert_to_llm = Some(converter);
    }

    /// Set the context transformation function (applied BEFORE `convert_to_llm`).
    ///
    /// Use this for context window management, message pruning, injecting
    /// external context, etc.
    pub fn set_transform_context<F, Fut>(&self, transform: F)
    where
        F: Fn(Vec<AgentMessage>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<AgentMessage>> + Send + 'static,
    {
        let transform = Arc::new(move |msgs: Vec<AgentMessage>| {
            let fut = transform(msgs);
            Box::pin(fut)
                as std::pin::Pin<Box<dyn std::future::Future<Output = Vec<AgentMessage>> + Send>>
        });
        self.hooks.write().transform_context = Some(transform);
    }

    // ============================================================================
    // Payload & Stream Hooks
    // ============================================================================

    /// Set the payload inspection / replacement hook.
    ///
    /// Called with the serialized request body before it is sent to the provider.
    pub fn set_on_payload<F, Fut>(&self, hook: F)
    where
        F: Fn(serde_json::Value, Model) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Option<serde_json::Value>> + Send + 'static,
    {
        let hook = Arc::new(move |payload: serde_json::Value, model: Model| {
            let fut = hook(payload, model);
            Box::pin(fut)
                as std::pin::Pin<
                    Box<dyn std::future::Future<Output = Option<serde_json::Value>> + Send>,
                >
        });
        self.hooks.write().on_payload = Some(hook);
    }

    /// Set a custom stream function to replace the default provider streaming.
    ///
    /// Useful for proxy backends, custom routing, etc.
    pub fn set_stream_fn<F, Fut>(&self, stream_fn: F)
    where
        F: Fn(&Model, &Context, StreamOptions) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = AssistantMessageEventStream> + Send + 'static,
    {
        let stream_fn = Arc::new(
            move |model: &Model, context: &Context, options: StreamOptions| {
                let fut = stream_fn(model, context, options);
                Box::pin(fut)
                    as std::pin::Pin<
                        Box<dyn std::future::Future<Output = AssistantMessageEventStream> + Send>,
                    >
            },
        );
        self.hooks.write().stream_fn = Some(stream_fn);
    }

    // ============================================================================
    // Configuration Setters
    // ============================================================================

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

    /// Set tool execution mode.
    pub fn set_tool_execution(&self, mode: ToolExecutionMode) {
        self.config.write().tool_execution = mode;
    }

    /// Set the steering queue mode.
    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.config.write().steering_mode = mode;
    }

    /// Get the steering queue mode.
    pub fn steering_mode(&self) -> QueueMode {
        self.config.read().steering_mode
    }

    /// Set the follow-up queue mode.
    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.config.write().follow_up_mode = mode;
    }

    /// Get the follow-up queue mode.
    pub fn follow_up_mode(&self) -> QueueMode {
        self.config.read().follow_up_mode
    }

    /// Set custom thinking budgets.
    pub fn set_thinking_budgets(&self, budgets: ThinkingBudgets) {
        self.config.write().thinking_budgets = Some(budgets);
    }

    /// Get the current thinking budgets.
    pub fn thinking_budgets(&self) -> Option<ThinkingBudgets> {
        self.config.read().thinking_budgets.clone()
    }

    /// Set the preferred transport.
    pub fn set_transport(&self, transport: Transport) {
        self.config.write().transport = transport;
    }

    /// Get the preferred transport.
    pub fn transport(&self) -> Transport {
        self.config.read().transport
    }

    /// Set the maximum retry delay in milliseconds.
    ///
    /// If the server requests a retry delay exceeding this value, the request
    /// fails immediately so higher-level retry logic can handle it with user
    /// visibility. `None` = use default (60_000ms). `Some(0)` = disable cap.
    pub fn set_max_retry_delay_ms(&self, ms: Option<u64>) {
        self.config.write().max_retry_delay_ms = ms;
    }

    /// Get the current max retry delay.
    pub fn max_retry_delay_ms(&self) -> Option<u64> {
        self.config.read().max_retry_delay_ms
    }

    /// Set the session ID for caching.
    pub fn set_session_id(&self, id: impl Into<String>) {
        *self.session_id.write() = Some(id.into());
    }

    /// Get the current session ID.
    pub fn session_id(&self) -> Option<String> {
        self.session_id.read().clone()
    }

    /// Clear the session ID.
    pub fn clear_session_id(&self) {
        *self.session_id.write() = None;
    }

    // ============================================================================
    // Event Subscription
    // ============================================================================

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

    /// Emit an event to all subscribers.
    fn emit(&self, event: AgentEvent) {
        self.subscribers.emit(&event);
    }

    // ============================================================================
    // State Management
    // ============================================================================

    /// Set the system prompt.
    pub fn set_system_prompt(&self, prompt: impl Into<String>) {
        self.state.set_system_prompt(prompt);
    }

    /// Set the model.
    pub fn set_model(&self, model: Model) {
        self.config.write().model = model;
    }

    /// Set the thinking level.
    pub fn set_thinking_level(&self, level: ThinkingLevel) {
        self.config.write().thinking_level = level;
    }

    /// Set the tools.
    pub fn set_tools(&self, tools: Vec<AgentTool>) {
        self.state.set_tools(tools);
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
        *self.session_id.write() = None;
    }

    // ============================================================================
    // Steering and Follow-up
    // ============================================================================

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

    /// Dequeue steering messages respecting the configured mode.
    fn dequeue_steering_messages(&self) -> Vec<AgentMessage> {
        let mode = self.config.read().steering_mode;
        let mut queue = self.steering_queue.lock();
        match mode {
            QueueMode::All => queue.drain(..).collect(),
            QueueMode::OneAtATime => {
                if let Some(first) = queue.pop_front() {
                    vec![first]
                } else {
                    Vec::new()
                }
            }
        }
    }

    /// Dequeue follow-up messages respecting the configured mode.
    fn dequeue_follow_up_messages(&self) -> Vec<AgentMessage> {
        let mode = self.config.read().follow_up_mode;
        let mut queue = self.follow_up_queue.lock();
        match mode {
            QueueMode::All => queue.drain(..).collect(),
            QueueMode::OneAtATime => {
                if let Some(first) = queue.pop_front() {
                    vec![first]
                } else {
                    Vec::new()
                }
            }
        }
    }

    // ============================================================================
    // Core Agent Loop
    // ============================================================================

    /// Default `convert_to_llm`: filters out Custom messages and maps directly.
    fn default_convert_to_llm(messages: Vec<AgentMessage>) -> Vec<Message> {
        messages
            .into_iter()
            .filter_map(|m| {
                let opt: Option<Message> = m.into();
                opt
            })
            .collect()
    }

    /// Build the context from current agent state using the full pipeline:
    /// `messages → transform_context → convert_to_llm → Context`
    async fn build_context(&self) -> Context {
        let system_prompt = self.state.system_prompt.read().clone();
        let messages = self.state.messages.read().clone();
        let tools = self.state.tools.read().clone();

        // Step 1: transform_context (if set)
        let transform = self.hooks.read().transform_context.clone();
        let messages = if let Some(ref transform) = transform {
            transform(messages).await
        } else {
            messages
        };

        // Step 2: convert_to_llm
        let converter = self.hooks.read().convert_to_llm.clone();
        let llm_messages = if let Some(ref converter) = converter {
            converter(messages).await
        } else {
            Self::default_convert_to_llm(messages)
        };

        // Step 3: Build Context
        let mut context = if system_prompt.is_empty() {
            Context::new()
        } else {
            Context::with_system_prompt(&system_prompt)
        };

        for msg in llm_messages {
            context.add_message(msg);
        }

        // Add tools
        if !tools.is_empty() {
            let tool_defs: Vec<Tool> = tools.iter().map(|t| t.as_tool()).collect();
            context.set_tools(tool_defs);
        }

        context
    }

    /// Resolve the provider to use.
    fn resolve_provider(&self) -> Result<ArcProtocol, AgentError> {
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

    /// Build stream options, resolving API key dynamically if configured.
    async fn build_stream_options(&self) -> StreamOptions {
        let security = self.config.read().security.clone();
        let model = self.config.read().model.clone();
        let on_payload = self.hooks.read().on_payload.clone();
        let transport = self.config.read().transport;
        let max_retry_delay_ms = self.config.read().max_retry_delay_ms;
        let session_id = self.session_id.read().clone();

        // Dynamic API key resolution: getApiKey > static api_key
        let get_api_key = self.hooks.read().get_api_key.clone();
        let api_key = if let Some(ref resolver) = get_api_key {
            let dynamic = resolver(model.provider.as_str()).await;
            dynamic.or_else(|| self.api_key.read().clone())
        } else {
            self.api_key.read().clone()
        };

        StreamOptions {
            api_key,
            security: Some(security),
            session_id,
            on_payload,
            transport: Some(transport),
            max_retry_delay_ms,
            ..Default::default()
        }
    }

    /// Build SimpleStreamOptions with thinking level/budget resolution.
    async fn build_simple_stream_options(&self) -> SimpleStreamOptions {
        let base = self.build_stream_options().await;
        let thinking_level = self.config.read().thinking_level;

        let (reasoning, thinking_budget_tokens) = if thinking_level != ThinkingLevel::Off {
            let budget = self
                .config
                .read()
                .thinking_budgets
                .as_ref()
                .and_then(|b| b.budget_for(thinking_level))
                .or_else(|| Some(crate::thinking::ThinkingConfig::default_budget(thinking_level)));
            (Some(thinking_level), budget)
        } else {
            (None, None)
        };

        SimpleStreamOptions {
            base,
            reasoning,
            thinking_budget_tokens,
        }
    }

    /// Run a single LLM turn: call provider, consume stream, return AssistantMessage.
    async fn run_turn(&self, provider: &ArcProtocol) -> Result<AssistantMessage, AgentError> {
        let context = self.build_context().await;
        let model = self.config.read().model.clone();
        let options = self.build_simple_stream_options().await;
        let stream_timeout = self.config.read().security.stream.result_timeout();

        // Create the stream (custom stream_fn or default provider via stream_simple)
        let stream_fn = self.hooks.read().stream_fn.clone();
        let mut stream: AssistantMessageEventStream =
            if let Some(ref custom_stream) = stream_fn {
                custom_stream(&model, &context, options.base).await
            } else {
                provider.stream_simple(&model, &context, options)
            };

        // Process stream events
        while let Some(event) = stream.next().await {
            // Check abort
            if self.abort_flag.load(Ordering::SeqCst) {
                return Err(AgentError::Other("Aborted".to_string()));
            }

            // Check for steering messages
            let steering = self.dequeue_steering_messages();
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
                        assistant_event: Box::new(event.clone()),
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
                            assistant_event: Box::new(event.clone()),
                        });
                    }
                }
                _ => {
                    if let Some(partial) = event.partial_message() {
                        self.emit(AgentEvent::MessageUpdate {
                            message: AgentMessage::Assistant(partial.clone()),
                            assistant_event: Box::new(event.clone()),
                        });
                    }
                }
            }
        }

        // Get the final result with timeout to prevent infinite blocking
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
    /// Supports: beforeToolCall/afterToolCall hooks, tool validation,
    /// streaming ToolExecutionUpdate events, bounded parallel exec + timeout,
    /// and abort-aware execution.
    async fn execute_tool_calls(
        &self,
        assistant_msg: &AssistantMessage,
        context: &Context,
    ) -> Vec<ToolResultMessage> {
        let tool_calls = assistant_msg.tool_calls();
        if tool_calls.is_empty() {
            return Vec::new();
        }

        let executor = self.hooks.read().tool_executor.clone();
        let execution_mode = self.config.read().tool_execution;
        let security = self.config.read().security.clone();
        let tool_timeout = security.agent.tool_execution_timeout();
        let before_hook = self.hooks.read().before_tool_call.clone();
        let after_hook = self.hooks.read().after_tool_call.clone();

        // Build Tool list for validation
        let agent_tools = self.state.tools.read().clone();
        let tool_defs: Vec<Tool> = agent_tools.iter().map(|t| t.as_tool()).collect();

        let mut results = Vec::new();

        match execution_mode {
            ToolExecutionMode::Parallel => {
                let max_parallel = security.agent.max_parallel_tool_calls;
                let abort_flag = Arc::clone(&self.abort_flag);

                let mut tool_futures = Vec::new();

                for tc in &tool_calls {
                    let tc_id = tc.id.clone();
                    let tc_name = tc.name.clone();
                    let tc_args = tc.arguments.clone();
                    let tc_clone = (*tc).clone();

                    self.emit(AgentEvent::ToolExecutionStart {
                        tool_call_id: tc_id.clone(),
                        tool_name: tc_name.clone(),
                        args: tc_args.clone(),
                    });

                    self.state.pending_tool_calls.write().insert(tc_id.clone());

                    // Validate tool call before execution
                    if let Some(result) = validate_tool_call_or_error(
                        &tc_id, &tc_name, &tc_args, &tool_defs, &security,
                    ) {
                        self.emit(AgentEvent::ToolExecutionEnd {
                            tool_call_id: tc_id.clone(),
                            tool_name: tc_name.clone(),
                            result: serde_json::json!({"error": result.text_content()}),
                            is_error: true,
                        });
                        self.state.pending_tool_calls.write().remove(&tc_id);
                        results.push(result);
                        continue;
                    }

                    // beforeToolCall hook
                    if let Some(result) = run_before_hook(
                        &before_hook, assistant_msg, &tc_clone, &tc_args, context,
                        &tc_id, &tc_name,
                    ).await {
                        self.emit(AgentEvent::ToolExecutionEnd {
                            tool_call_id: tc_id.clone(),
                            tool_name: tc_name.clone(),
                            result: serde_json::json!({"error": result.text_content()}),
                            is_error: true,
                        });
                        self.state.pending_tool_calls.write().remove(&tc_id);
                        results.push(result);
                        continue;
                    }

                    let executor = executor.clone();
                    let abort = abort_flag.clone();
                    let after_hook = after_hook.clone();
                    let assistant_msg_clone = assistant_msg.clone();
                    let context_clone = context.clone();
                    let subscribers = Arc::clone(&self.subscribers);

                    tool_futures.push(async move {
                        let (final_content, final_is_error) = execute_and_apply_after_hook(
                            ToolExecCtx {
                                executor: &executor,
                                after_hook: &after_hook,
                                subscribers: &subscribers,
                                tc_id: &tc_id,
                                tc_name: &tc_name,
                                tc_args: &tc_args,
                                tc: &tc_clone,
                                assistant_msg: &assistant_msg_clone,
                                context: &context_clone,
                                tool_timeout,
                                abort_flag: abort,
                            },
                        ).await;

                        (tc_id, tc_name, final_content, final_is_error)
                    });
                }

                // Use buffer_unordered for bounded parallel execution
                let mut buffered =
                    futures::stream::iter(tool_futures).buffer_unordered(max_parallel);

                while let Some((tc_id, tc_name, content, is_error)) = buffered.next().await {
                    let result_json =
                        serde_json::to_value(&content).unwrap_or(serde_json::Value::Null);
                    self.emit(AgentEvent::ToolExecutionEnd {
                        tool_call_id: tc_id.clone(),
                        tool_name: tc_name.clone(),
                        result: result_json,
                        is_error,
                    });

                    self.state.pending_tool_calls.write().remove(&tc_id);

                    results.push(ToolResultMessage::new(tc_id, tc_name, content, is_error));
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
                    let tc_clone = (*tc).clone();

                    self.emit(AgentEvent::ToolExecutionStart {
                        tool_call_id: tc_id.clone(),
                        tool_name: tc_name.clone(),
                        args: tc_args.clone(),
                    });

                    self.state.pending_tool_calls.write().insert(tc_id.clone());

                    // Validate tool call before execution
                    if let Some(result) = validate_tool_call_or_error(
                        &tc_id, &tc_name, &tc_args, &tool_defs, &security,
                    ) {
                        self.emit(AgentEvent::ToolExecutionEnd {
                            tool_call_id: tc_id.clone(),
                            tool_name: tc_name.clone(),
                            result: serde_json::json!({"error": result.text_content()}),
                            is_error: true,
                        });
                        self.state.pending_tool_calls.write().remove(&tc_id);
                        results.push(result);
                        continue;
                    }

                    // beforeToolCall hook
                    if let Some(result) = run_before_hook(
                        &before_hook, assistant_msg, &tc_clone, &tc_args, context,
                        &tc_id, &tc_name,
                    ).await {
                        self.emit(AgentEvent::ToolExecutionEnd {
                            tool_call_id: tc_id.clone(),
                            tool_name: tc_name.clone(),
                            result: serde_json::json!({"error": result.text_content()}),
                            is_error: true,
                        });
                        self.state.pending_tool_calls.write().remove(&tc_id);
                        results.push(result);
                        continue;
                    }

                    let abort_flag = Arc::clone(&self.abort_flag);
                    let (final_content, final_is_error) = execute_and_apply_after_hook(
                        ToolExecCtx {
                            executor: &executor,
                            after_hook: &after_hook,
                            subscribers: &self.subscribers,
                            tc_id: &tc_id,
                            tc_name: &tc_name,
                            tc_args: &tc_args,
                            tc: &tc_clone,
                            assistant_msg,
                            context,
                            tool_timeout,
                            abort_flag,
                        },
                    ).await;

                    let result_json = serde_json::to_value(&final_content)
                        .unwrap_or(serde_json::Value::Null);
                    self.emit(AgentEvent::ToolExecutionEnd {
                        tool_call_id: tc_id.clone(),
                        tool_name: tc_name.clone(),
                        result: result_json,
                        is_error: final_is_error,
                    });

                    self.state.pending_tool_calls.write().remove(&tc_id);

                    results.push(ToolResultMessage::new(
                        tc_id,
                        tc_name,
                        final_content,
                        final_is_error,
                    ));

                    // Check for steering messages after each sequential tool
                    let steering = self.dequeue_steering_messages();
                    if !steering.is_empty() {
                        for steer_msg in steering {
                            self.state.add_message(steer_msg);
                        }
                        // Break out of remaining tool calls
                        break;
                    }
                }
            }
        }

        results
    }

    /// Run the agent loop: stream LLM → check tool calls → execute → loop.
    async fn run_loop(&self) -> Result<Vec<AgentMessage>, AgentError> {
        let provider = if self.hooks.read().stream_fn.is_some() {
            // When a custom stream function is set, we don't need a provider.
            // Create a dummy Arc for the loop (won't be used).
            None
        } else {
            Some(self.resolve_provider()?)
        };

        let max_turns = *self.max_turns.read();
        let mut new_messages = Vec::new();
        let mut turn_count = 0;

        // Sync message limit from security config
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
            let dummy_provider: ArcProtocol = Arc::new(DummyProvider);
            let active_provider = provider.as_ref().unwrap_or(&dummy_provider);
            let assistant_result = self.run_turn(active_provider).await;

            match assistant_result {
                Ok(assistant_msg) => {
                    // Build context snapshot for tool hook use
                    let context = self.build_context().await;

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
                        let tool_results =
                            self.execute_tool_calls(&assistant_msg, &context).await;

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
                        let follow_ups = self.dequeue_follow_up_messages();
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
                        let follow_ups = self.dequeue_follow_up_messages();
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
    /// Uses atomic compare_exchange to prevent TOCTOU race condition.
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
    /// Uses atomic compare_exchange to prevent TOCTOU race condition.
    pub async fn continue_(&self) -> Result<Vec<AgentMessage>, AgentError> {
        if self
            .state
            .is_streaming
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(AgentError::AlreadyStreaming);
        }

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

    /// Take a consistent point-in-time snapshot of the agent's full state.
    ///
    /// Combines runtime state from [`AgentState`] with configuration
    /// (model, thinking_level) from [`AgentConfig`].
    pub fn snapshot(&self) -> AgentStateSnapshot {
        let config = self.config.read();
        let system_prompt = self.state.system_prompt.read().clone();
        let messages = self.state.messages.read().clone();
        let is_streaming = self.state.is_streaming();
        let stream_message = self.state.stream_message.read().clone();
        let pending_tool_calls = self.state.pending_tool_calls.read().clone();
        let error = self.state.error.read().clone();
        let max_messages = self.state.get_max_messages();
        let message_count = messages.len();

        AgentStateSnapshot {
            system_prompt,
            model: config.model.clone(),
            thinking_level: config.thinking_level,
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

/// Helper: wait until the abort flag is set.
async fn wait_for_abort(flag: Arc<AtomicBool>) {
    loop {
        if flag.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

// ============================================================================
// Extracted helpers for execute_tool_calls deduplication
// ============================================================================

/// Validate a tool call against the tool definitions and security config.
///
/// Returns `Some(ToolResultMessage)` (an error result) if validation fails,
/// or `None` if the tool call is valid and execution should proceed.
/// Skips validation when disabled in config or when no tools are registered.
fn validate_tool_call_or_error(
    tc_id: &str,
    tc_name: &str,
    tc_args: &serde_json::Value,
    tool_defs: &[Tool],
    security: &SecurityConfig,
) -> Option<ToolResultMessage> {
    if !security.agent.validate_tool_calls || tool_defs.is_empty() {
        return None;
    }

    let tc = ToolCall::new(tc_id, tc_name, tc_args.clone());
    match crate::validation::validate_tool_call(tool_defs, &tc) {
        Ok(_) => None,
        Err(e) => Some(ToolResultMessage::error(tc_id, tc_name, e.to_string())),
    }
}

/// Run the `before_tool_call` hook if set.
///
/// Returns `Some(ToolResultMessage)` (a blocked/error result) if the hook
/// blocked execution, or `None` if execution should proceed.
async fn run_before_hook(
    before_hook: &Option<BeforeToolCallFn>,
    assistant_msg: &AssistantMessage,
    tc: &ToolCall,
    tc_args: &serde_json::Value,
    context: &Context,
    tc_id: &str,
    tc_name: &str,
) -> Option<ToolResultMessage> {
    let hook = before_hook.as_ref()?;
    let ctx = BeforeToolCallContext {
        assistant_message: assistant_msg.clone(),
        tool_call: tc.clone(),
        args: tc_args.clone(),
        context: context.clone(),
    };
    match hook(ctx).await {
        Some(result) if result.block => {
            let reason = result
                .reason
                .unwrap_or_else(|| "Tool call blocked by before_tool_call hook".to_string());
            Some(ToolResultMessage::error(tc_id, tc_name, reason))
        }
        _ => None,
    }
}

/// Context for a single tool call execution.
///
/// Groups the parameters needed by `execute_and_apply_after_hook` to avoid
/// exceeding clippy's `too_many_arguments` limit.
struct ToolExecCtx<'a> {
    executor: &'a Option<ToolExecutor>,
    after_hook: &'a Option<AfterToolCallFn>,
    subscribers: &'a Arc<Subscribers>,
    tc_id: &'a str,
    tc_name: &'a str,
    tc_args: &'a serde_json::Value,
    tc: &'a ToolCall,
    assistant_msg: &'a AssistantMessage,
    context: &'a Context,
    tool_timeout: std::time::Duration,
    abort_flag: Arc<AtomicBool>,
}

/// Execute a tool call and apply the `after_tool_call` hook if set.
///
/// Handles: executor invocation with timeout, abort-awareness,
/// streaming `ToolExecutionUpdate` events, error detection,
/// and after-hook overrides.
///
/// Returns `(final_content, final_is_error)`.
async fn execute_and_apply_after_hook(ctx: ToolExecCtx<'_>) -> (Vec<ContentBlock>, bool) {
    let ToolExecCtx {
        executor,
        after_hook,
        subscribers,
        tc_id,
        tc_name,
        tc_args,
        tc,
        assistant_msg,
        context,
        tool_timeout,
        abort_flag,
    } = ctx;
    // Execute the tool
    let tool_result = if let Some(ref exec) = executor {
        // Build update callback for streaming partial results
        let subs = Arc::clone(subscribers);
        let update_tc_id = tc_id.to_string();
        let update_tc_name = tc_name.to_string();
        let update_cb: ToolUpdateCallback = Arc::new(move |partial: serde_json::Value| {
            subs.emit(&AgentEvent::ToolExecutionUpdate {
                tool_call_id: update_tc_id.clone(),
                tool_name: update_tc_name.clone(),
                partial_result: partial,
            });
        });

        let exec_future = exec(tc_name, tc_id, tc_args, Some(update_cb));

        // Race: tool execution vs timeout vs abort
        tokio::select! {
            result = exec_future => result,
            _ = tokio::time::sleep(tool_timeout) => {
                AgentToolResult::error(format!(
                    "Tool '{}' timed out after {:?}",
                    tc_name, tool_timeout
                ))
            }
            _ = wait_for_abort(abort_flag) => {
                AgentToolResult::error(format!("Tool '{}' aborted", tc_name))
            }
        }
    } else {
        AgentToolResult::error(format!(
            "No tool executor configured for tool '{}'",
            tc_name
        ))
    };

    // Detect is_error from content
    let mut is_error = tool_result.content.iter().any(|block| {
        if let Some(text) = block.as_text() {
            text.text.starts_with("Error:") || text.text.starts_with("error:")
        } else {
            false
        }
    });

    let mut final_content = tool_result.content.clone();

    // Apply after_tool_call hook
    if let Some(ref hook) = after_hook {
        let after_ctx = AfterToolCallContext {
            assistant_message: assistant_msg.clone(),
            tool_call: tc.clone(),
            args: tc_args.clone(),
            result: tool_result,
            is_error,
            context: context.clone(),
        };
        if let Some(overrides) = hook(after_ctx).await {
            if let Some(content_override) = overrides.content {
                final_content = content_override;
            }
            if let Some(error_override) = overrides.is_error {
                is_error = error_override;
            }
        }
    }

    (final_content, is_error)
}

impl Default for Agent {
    fn default() -> Self {
        Self::new()
    }
}

/// Minimal dummy provider used when a custom `stream_fn` is set.
/// This should never actually be called.
struct DummyProvider;

#[async_trait::async_trait]
impl crate::provider::LLMProtocol for DummyProvider {
    fn provider_type(&self) -> Provider {
        Provider::Custom("dummy".to_string())
    }

    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let stream = AssistantMessageEventStream::new_assistant_stream();
        let error_msg = AssistantMessage::builder()
            .provider(Provider::Custom("dummy".to_string()))
            .model("dummy")
            .stop_reason(StopReason::Error)
            .error_message("DummyProvider should not be called when stream_fn is set")
            .build()
            .unwrap();
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: error_msg,
        });
        stream.end(None);
        stream
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.stream(model, context, options.base)
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

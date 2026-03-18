# Agent Module

Stateful LLM conversation manager with autonomous tool execution loop. Thread-safe, streaming-first, fully configurable.

## Architecture

```
                         +-----------+
                         |   Agent   |  <-- prompt() / continue_() / steer() / follow_up()
                         +-----+-----+
                               |
            +------------------+------------------+
            |                  |                  |
      +-----------+    +-------------+    +-----------+
      |AgentState |    | AgentConfig |    | Subscribers|
      | (RwLock)  |    |  (RwLock)   |    | (HashMap) |
      +-----------+    +-------------+    +-----------+
            |
            |  run_loop()
            v
     +------+------+     Each turn:
     |  Turn Loop  | <-- max_turns guard (default: 25)
     +------+------+
            |
            v
   +--------+--------+
   | Context Pipeline |
   |  transform_ctx   |
   |  convert_to_llm  |
   |  build Context   |
   +--------+---------+
            |
            v
   +--------+--------+
   | Stream Options   |
   | get_api_key      |  <-- dynamic key resolution
   | on_payload       |  <-- request body hook
   | session_id       |
   | transport        |
   | thinking budget  |
   +--------+---------+
            |
            v
   +--------+---------+    custom stream_fn
   |  LLM Provider    | <--- OR --->  StreamFn
   | stream_simple()  |
   +--------+---------+
            |
            v
   +--------+---------+
   | Stream Consume   |  <-- steering check per event
   | TextDelta, etc.  |  <-- MessageUpdate events
   +--------+---------+
            |
            v
   +--------+---------+
   | Tool Execution   |  parallel (bounded) / sequential
   | beforeToolCall   |  <-- block / allow
   | validate args    |  <-- JSON Schema
   | execute (timeout)|  <-- abort-aware
   | afterToolCall    |  <-- override result
   +--------+---------+
            |
            v
   +--------+---------+
   | Follow-up Queue  |  <-- continue or break
   +------------------+
```

### Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `Agent` | `agent.rs` | Top-level API, orchestrates the loop, hooks, queues, events |
| `AgentState` | `state.rs` | Thread-safe conversation state (messages, tools, streaming status) |
| `AgentConfig` | `types.rs` | Configuration: model, thinking, security, transport, queue modes |
| `AgentHooks` | `types.rs` | Aggregated hook container (tool executor, before/after hooks, pipeline fns) |
| `AgentEvent` | `types.rs` | Event enum for the observer pattern (10 event types) |
| `AgentMessage` | `types.rs` | Tagged enum wrapping User/Assistant/ToolResult/Custom |

### Thread Safety Model

All mutable state uses `parking_lot` locks (non-poisoning). Key concurrency patterns:

- **CAS mutual exclusion** for `prompt()` / `continue_()` via `AtomicBool::compare_exchange`
- **RwLock** for config, hooks, state fields (concurrent reads, exclusive writes)
- **Mutex** for steering / follow-up queues
- **Arc wrapping** for all callback types and shared state
- **Abort-aware** tool execution via `tokio::select!` racing against an atomic flag

## API Reference

### Construction

```rust
use tiy_core::agent::Agent;
use tiy_core::types::*;

// Default agent (gpt-4o-mini)
let agent = Agent::new();

// Agent with specific model
let model = Model::builder()
    .id("claude-sonnet-4-20250514")
    .name("Claude Sonnet 4")
    .provider(Provider::Anthropic)
    .context_window(200000)
    .max_tokens(8192)
    .build()
    .unwrap();
let agent = Agent::with_model(model);
```

### Prompting

```rust
// Send a prompt (async, blocks until agent loop completes)
let messages = agent.prompt("What is 2 + 2?").await?;

// Send typed message
let msg = AgentMessage::User(UserMessage::text("Hello"));
let messages = agent.prompt(msg).await?;

// Continue from current state (e.g., after injecting tool results externally)
let messages = agent.continue_().await?;

// Abort current operation
agent.abort();

// Wait for agent to finish
agent.wait_for_idle().await;
```

### State Management

```rust
agent.set_system_prompt("You are a helpful assistant.");
agent.set_model(my_model);
agent.set_thinking_level(ThinkingLevel::Medium);
agent.set_tools(vec![my_tool]);

// Message operations
agent.append_message(AgentMessage::from("user input"));
agent.replace_messages(vec![...]);
agent.clear_messages();

// Full reset (messages, queues, session_id, streaming state)
agent.reset();

// Access underlying state
let state = agent.state();
let snapshot = agent.snapshot();   // consistent point-in-time view
println!("Messages: {}", state.message_count());
```

### Provider & API Key

```rust
use std::sync::Arc;
use tiy_core::provider::{openai::OpenAIProvider, register_provider};

// Option 1: Explicit provider
agent.set_provider(Arc::new(OpenAIProvider::new()));

// Option 2: Registry (auto-resolved from model.provider)
register_provider(Arc::new(OpenAIProvider::new()));

// Static API key
agent.set_api_key("sk-...");

// Dynamic API key (called before each LLM request)
agent.set_get_api_key(|provider_name: &str| async move {
    // Useful for expiring OAuth tokens
    fetch_token_for(provider_name).await
});
```

### Tool Execution

```rust
use tiy_core::agent::*;

// Define tools
let tool = AgentTool::new(
    "get_weather",
    "Get Weather",                         // human-readable label
    "Get current weather for a location",
    serde_json::json!({
        "type": "object",
        "properties": {
            "location": { "type": "string", "description": "City name" }
        },
        "required": ["location"]
    }),
);
agent.set_tools(vec![tool]);

// Simple executor (no streaming updates)
agent.set_tool_executor_simple(|name, id, args| async move {
    match name {
        "get_weather" => {
            let location = args["location"].as_str().unwrap_or("unknown");
            AgentToolResult::text(format!("Weather in {}: 22C, sunny", location))
        }
        _ => AgentToolResult::error(format!("Unknown tool: {}", name)),
    }
});

// Full executor with streaming progress updates
agent.set_tool_executor(|name, id, args, on_update| async move {
    if let Some(cb) = &on_update {
        cb(serde_json::json!({"status": "starting..."}));
    }
    // ... do work ...
    if let Some(cb) = &on_update {
        cb(serde_json::json!({"status": "50% complete"}));
    }
    AgentToolResult::text("Done!")
});

// Execution mode
agent.set_tool_execution(ToolExecutionMode::Parallel);   // default, bounded concurrency
agent.set_tool_execution(ToolExecutionMode::Sequential);  // one at a time, checks steering between tools
```

### Hooks

#### beforeToolCall

Called after argument validation, before tool execution. Can block a tool call.

```rust
agent.set_before_tool_call(|ctx: BeforeToolCallContext| async move {
    if ctx.tool_call.name == "dangerous_tool" {
        Some(BeforeToolCallResult::blocked("Tool is restricted"))
    } else {
        None  // allow
    }
});
```

#### afterToolCall

Called after tool execution, before the result is committed. Can override content, details, or is_error.

```rust
agent.set_after_tool_call(|ctx: AfterToolCallContext| async move {
    if ctx.is_error {
        Some(AfterToolCallResult {
            content: Some(vec![ContentBlock::Text(TextContent::new("Sanitized error"))]),
            is_error: Some(true),
            ..Default::default()
        })
    } else {
        None  // keep original
    }
});
```

#### onPayload

Inspect or replace the serialized HTTP request body before it's sent to the provider.

```rust
agent.set_on_payload(|payload: serde_json::Value, model: Model| async move {
    println!("Request to {}: {}", model.id, payload);
    // Return Some(modified) to replace, None to keep original
    None
});
```

### Context Pipeline

The pipeline runs before each LLM call:

```
state.messages  -->  transformContext  -->  convertToLlm  -->  Context
```

#### transformContext

Pre-processing on `AgentMessage[]`. Use for pruning, injecting external context, context window management.

```rust
agent.set_transform_context(|messages: Vec<AgentMessage>| async move {
    // Keep only the last 50 messages to fit context window
    if messages.len() > 50 {
        messages[messages.len() - 50..].to_vec()
    } else {
        messages
    }
});
```

#### convertToLlm

Converts `AgentMessage[]` to `Message[]`. The default filters out `Custom` messages.

```rust
agent.set_convert_to_llm(|messages: Vec<AgentMessage>| async move {
    messages.into_iter().filter_map(|m| {
        match &m {
            AgentMessage::Custom { message_type, data } if message_type == "context_note" => {
                // Convert custom note into a user message for the LLM
                Some(Message::User(UserMessage::text(
                    data["text"].as_str().unwrap_or("")
                )))
            }
            _ => {
                let opt: Option<Message> = m.into();
                opt
            }
        }
    }).collect()
});
```

#### Custom streamFn

Replace the default provider streaming entirely.

```rust
agent.set_stream_fn(|model, context, options| async move {
    // Route to a custom backend, proxy, etc.
    my_proxy_stream(model, context, options).await
});
```

### Event System

Subscribe to agent events for UI updates, logging, telemetry.

```rust
let unsub = agent.subscribe(|event: &AgentEvent| {
    match event {
        AgentEvent::AgentStart => println!("Agent started"),
        AgentEvent::TurnStart => println!("New turn"),
        AgentEvent::MessageUpdate { message, assistant_event } => {
            if let AssistantMessageEvent::TextDelta { delta, .. } = assistant_event.as_ref() {
                print!("{}", delta);  // stream text to UI
            }
        }
        AgentEvent::ToolExecutionStart { tool_name, .. } => {
            println!("Executing tool: {}", tool_name);
        }
        AgentEvent::ToolExecutionUpdate { partial_result, .. } => {
            println!("Progress: {}", partial_result);
        }
        AgentEvent::ToolExecutionEnd { tool_name, is_error, .. } => {
            println!("Tool {} finished (error={})", tool_name, is_error);
        }
        AgentEvent::AgentEnd { messages } => {
            println!("Agent finished with {} new messages", messages.len());
        }
        _ => {}
    }
});

// Unsubscribe when done
unsub();
```

**Event lifecycle within a single turn:**

```
AgentStart
  TurnStart
    MessageUpdate (Start)
    MessageUpdate (TextDelta) ...
    MessageUpdate (ToolCallDelta) ...
    MessageStart  (finalized assistant message)
    MessageEnd
    ToolExecutionStart
      ToolExecutionUpdate ...
    ToolExecutionEnd
  TurnEnd { message, tool_results }
  TurnStart          <-- next turn if tool calls or follow-ups
    ...
AgentEnd { messages }
```

### Steering & Follow-up Queues

#### Steering (interruption)

Inject messages mid-run. Checked during stream consumption and between sequential tool calls.

```rust
// From another thread/task while agent is running:
agent.steer(AgentMessage::from("Actually, focus on X instead."));

// Queue mode
agent.set_steering_mode(QueueMode::OneAtATime);  // deliver one per check
agent.set_steering_mode(QueueMode::All);          // deliver all at once (default)
```

#### Follow-up (continuation)

Queue messages processed after the current work completes. Checked when no more tool calls remain.

```rust
agent.follow_up(AgentMessage::from("Now summarize the results."));

agent.set_follow_up_mode(QueueMode::OneAtATime);
agent.set_follow_up_mode(QueueMode::All);          // default
```

#### Queue Management

```rust
agent.clear_steering_queue();
agent.clear_follow_up_queue();
agent.clear_all_queues();
agent.has_queued_messages();  // true if either queue has items
```

### Configuration

#### Thinking Budgets

Custom token budgets per thinking level. Flows through `SimpleStreamOptions` to the provider.

```rust
use tiy_core::agent::ThinkingBudgets;
use tiy_core::thinking::ThinkingLevel;

agent.set_thinking_level(ThinkingLevel::High);
agent.set_thinking_budgets(ThinkingBudgets {
    minimal: Some(128),
    low:     Some(512),
    medium:  Some(2048),
    high:    Some(8192),
});
```

#### Transport

```rust
use tiy_core::agent::Transport;

agent.set_transport(Transport::Sse);        // default
agent.set_transport(Transport::WebSocket);
agent.set_transport(Transport::Auto);
```

#### Session ID

For provider-side caching (e.g., OpenAI prompt caching).

```rust
agent.set_session_id("my-session-42");
assert_eq!(agent.session_id(), Some("my-session-42".to_string()));
agent.clear_session_id();
```

#### Max Retry Delay

Cap how long the agent waits for server-requested retries.

```rust
agent.set_max_retry_delay_ms(Some(5000));   // 5 seconds max
agent.set_max_retry_delay_ms(Some(0));      // disable cap
agent.set_max_retry_delay_ms(None);         // use provider default
```

#### Max Turns

Prevent runaway loops.

```rust
agent.set_max_turns(10);  // default is 25
```

#### Security Config

Comprehensive resource limits for HTTP, agent behavior, and streaming.

```rust
use tiy_core::types::SecurityConfig;

let mut security = SecurityConfig::default();
security.agent.max_parallel_tool_calls = 8;
security.agent.tool_execution_timeout_secs = 60;
security.agent.validate_tool_calls = true;
security.agent.max_messages = 500;
security.http.connect_timeout_secs = 15;
security.stream.result_timeout_secs = 300;
agent.set_security_config(security);
```

| Limit | Default | Description |
|-------|---------|-------------|
| `http.connect_timeout_secs` | 30 | TCP connect timeout |
| `http.request_timeout_secs` | 1800 | Total request timeout incl. streaming |
| `agent.max_messages` | 1000 | Max conversation history (FIFO eviction) |
| `agent.max_parallel_tool_calls` | 16 | Bounded concurrency for parallel mode |
| `agent.tool_execution_timeout_secs` | 120 | Per-tool timeout |
| `agent.validate_tool_calls` | true | JSON Schema validation before execution |
| `stream.result_timeout_secs` | 600 | Timeout waiting for stream result |

### Custom Messages

Inject application-specific messages into the conversation.

```rust
// Create a custom message
let custom = AgentMessage::Custom {
    message_type: "artifact".to_string(),
    data: serde_json::json!({
        "title": "Generated Chart",
        "content": "<svg>...</svg>",
    }),
};
agent.append_message(custom);

// Custom messages are filtered out by the default convertToLlm.
// Provide a custom converter to handle them:
agent.set_convert_to_llm(|messages| async move {
    messages.into_iter().filter_map(|m| {
        match m {
            AgentMessage::Custom { message_type, data } => {
                // Optionally convert to LLM context
                None  // or Some(Message::User(...))
            }
            other => {
                let opt: Option<Message> = other.into();
                opt
            }
        }
    }).collect()
});
```

### Error Handling

```rust
use tiy_core::agent::AgentError;

match agent.prompt("hello").await {
    Ok(messages) => { /* success */ }
    Err(AgentError::AlreadyStreaming) => {
        // Another prompt() or continue_() is running.
        // Use steer() or follow_up() instead.
    }
    Err(AgentError::NoMessages) => {
        // continue_() called with empty history
    }
    Err(AgentError::CannotContinueFromAssistant) => {
        // continue_() called when last message is Assistant
    }
    Err(AgentError::ProviderError(msg)) => {
        // LLM returned an error (e.g., rate limit, invalid key)
    }
    Err(AgentError::Other(msg)) => {
        // "Aborted", stream timeout, etc.
    }
    _ => {}
}
```

## Complete Example

```rust
use std::sync::Arc;
use tiy_core::agent::*;
use tiy_core::thinking::ThinkingLevel;
use tiy_core::types::*;
use tiy_core::provider::{register_provider, openai::OpenAIProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Register provider
    register_provider(Arc::new(OpenAIProvider::new()));

    // Create agent
    let model = Model::builder()
        .id("gpt-4o-mini")
        .name("GPT-4o Mini")
        .provider(Provider::OpenAI)
        .context_window(128000)
        .max_tokens(16384)
        .build()?;
    let agent = Agent::with_model(model);

    // Configure
    agent.set_system_prompt("You are a helpful coding assistant.");
    agent.set_api_key(std::env::var("OPENAI_API_KEY")?);
    agent.set_thinking_level(ThinkingLevel::Medium);
    agent.set_max_turns(10);

    // Define a tool
    let tool = AgentTool::new(
        "run_code",
        "Run Code",
        "Execute a code snippet and return the output",
        serde_json::json!({
            "type": "object",
            "properties": {
                "language": { "type": "string", "enum": ["python", "javascript"] },
                "code": { "type": "string" }
            },
            "required": ["language", "code"]
        }),
    );
    agent.set_tools(vec![tool]);

    // Register tool executor
    agent.set_tool_executor_simple(|name, _id, args| async move {
        match name {
            "run_code" => {
                let lang = args["language"].as_str().unwrap_or("unknown");
                let code = args["code"].as_str().unwrap_or("");
                // In reality, run the code in a sandbox...
                AgentToolResult::text(format!("[{}] Output: executed successfully", lang))
            }
            _ => AgentToolResult::error(format!("Unknown tool: {}", name)),
        }
    });

    // Subscribe to events for real-time UI
    let _unsub = agent.subscribe(|event| {
        match event {
            AgentEvent::MessageUpdate { assistant_event, .. } => {
                if let AssistantMessageEvent::TextDelta { delta, .. } = assistant_event.as_ref() {
                    print!("{}", delta);
                }
            }
            AgentEvent::ToolExecutionStart { tool_name, .. } => {
                println!("\n[Calling tool: {}]", tool_name);
            }
            AgentEvent::ToolExecutionEnd { tool_name, is_error, .. } => {
                println!("[Tool {} {}]", tool_name, if *is_error { "failed" } else { "done" });
            }
            _ => {}
        }
    });

    // Run a prompt
    let result = agent.prompt("Write a Python function to compute fibonacci numbers, then test it.").await?;
    println!("\n\nAgent produced {} messages", result.len());

    // Queue a follow-up
    agent.follow_up(AgentMessage::from("Now optimize it with memoization."));

    // Continue handles the follow-up automatically on next prompt
    // Or we can continue explicitly after adding tool results

    Ok(())
}
```

## Type Reference

### Core Types

| Type | Description |
|------|-------------|
| `Agent` | Main entry point. Thread-safe, all methods take `&self`. |
| `AgentState` | Thread-safe conversation state. Access via `agent.state()`. |
| `AgentStateSnapshot` | Serializable point-in-time view. Get via `agent.snapshot()`. |
| `AgentConfig` | Model, thinking level, security, transport, queue modes. |
| `AgentHooks` | Aggregated hook container for all Agent callbacks. |
| `AgentMessage` | `User` / `Assistant` / `ToolResult` / `Custom` |
| `AgentEvent` | 10-variant event enum for the observer pattern. |
| `AgentTool` | Tool definition with name, label, description, JSON Schema parameters. |
| `AgentToolResult` | Tool execution result: content blocks + optional details. |
| `AgentError` | Error enum: `AlreadyStreaming`, `NoMessages`, `ProviderError`, etc. |

### Hook Types

| Type | Signature | Purpose |
|------|-----------|---------|
| `BeforeToolCallFn` | `(BeforeToolCallContext) -> Future<Option<BeforeToolCallResult>>` | Gate tool execution |
| `AfterToolCallFn` | `(AfterToolCallContext) -> Future<Option<AfterToolCallResult>>` | Override tool results |
| `OnPayloadFn` | `(Value, Model) -> Future<Option<Value>>` | Inspect/replace HTTP body |
| `ConvertToLlmFn` | `(Vec<AgentMessage>) -> Future<Vec<Message>>` | Custom message conversion |
| `TransformContextFn` | `(Vec<AgentMessage>) -> Future<Vec<AgentMessage>>` | Context pre-processing |
| `GetApiKeyFn` | `(&str) -> Future<Option<String>>` | Dynamic API key resolution |
| `StreamFn` | `(&Model, &Context, StreamOptions) -> Future<EventStream>` | Custom stream implementation |
| `ToolUpdateCallback` | `(Value) -> ()` | Streaming tool progress |

### Configuration Types

| Type | Variants / Fields |
|------|-------------------|
| `ToolExecutionMode` | `Parallel` (default) / `Sequential` |
| `QueueMode` | `All` (default) / `OneAtATime` |
| `ThinkingLevel` | `Off` / `Minimal` / `Low` / `Medium` / `High` / `XHigh` |
| `ThinkingBudgets` | `{ minimal, low, medium, high }` (all `Option<u32>`) |
| `Transport` | `Sse` (default) / `WebSocket` / `Auto` |
| `SecurityConfig` | `{ http, agent, stream, header_policy, url_policy }` |

### Event Types

| Event | Payload | Trigger |
|-------|---------|---------|
| `AgentStart` | -- | `prompt()` / `continue_()` begins |
| `AgentEnd` | `messages: Vec<AgentMessage>` | Loop completes (success or error) |
| `TurnStart` | -- | Each LLM call begins |
| `TurnEnd` | `message, tool_results` | LLM call + tools complete |
| `MessageStart` | `message` | Finalized assistant message committed |
| `MessageUpdate` | `message, assistant_event` | Streaming deltas (text, thinking, tool call) |
| `MessageEnd` | `message` | After MessageStart |
| `ToolExecutionStart` | `tool_call_id, tool_name, args` | Before tool runs |
| `ToolExecutionUpdate` | `tool_call_id, tool_name, partial_result` | Streaming tool progress |
| `ToolExecutionEnd` | `tool_call_id, tool_name, result, is_error` | Tool finished |

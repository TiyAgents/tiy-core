//! Agent example for tiycore.
//!
//! Demonstrates the full Agent capability set:
//!   1. Basic prompt & streaming events
//!   2. Tool definition, registration, and execution
//!   3. Hooks: beforeToolCall / afterToolCall / onPayload
//!   4. Context pipeline: transformContext / convertToLlm
//!   5. Thinking budgets & transport config
//!   6. Session ID, dynamic API key, max turns, security
//!   7. Steering & follow-up queues
//!   8. Custom messages
//!   9. State management: snapshot, reset, continue_
//!
//! Environment variables:
//!   LLM_API_KEY    — API key (required for live calls)
//!   LLM_BASE_URL   — Base URL override (optional)
//!   LLM_MODEL      — Model ID (default: gpt-4o-mini)
//!
//! Run:
//!   cargo run --example agent_example
//!   RUST_LOG=info cargo run --example agent_example

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tiycore::{agent::*, models::get_model, thinking::ThinkingLevel, types::*};

/// Resolve an env var with fallback.
fn env_or(primary: &str, fallback: &str) -> Option<String> {
    std::env::var(primary)
        .or_else(|_| std::env::var(fallback))
        .ok()
        .filter(|v| !v.is_empty())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("=== tiycore Agent Example ===\n");

    // ========================================================================
    // 1. Create Agent with model
    // ========================================================================
    println!("--- 1. Create Agent ---");

    let model_id = std::env::var("LLM_MODEL")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "gpt-4o-mini".to_string());

    let model = get_model("openai", &model_id).unwrap_or_else(|| {
        Model::builder()
            .id(&model_id)
            .name(&model_id)
            .provider(Provider::Zenmux)
            .context_window(128000)
            .max_tokens(4096)
            .build()
            .expect("Failed to build model")
    });

    let agent = Agent::with_model(model.clone());
    println!("  Model: {} ({})", model.name, model.id);
    println!("  Provider: {}", model.provider);

    // ========================================================================
    // 2. Configure the Agent
    // ========================================================================
    println!("\n--- 2. Configure Agent ---");

    // System prompt
    agent.set_system_prompt("You are a helpful coding assistant. Be concise.");

    // API key
    let api_key = env_or("LLM_API_KEY", "OPENAI_API_KEY");
    if let Some(ref key) = api_key {
        agent.set_api_key(key.clone());
        println!("  API key: [set]");
    } else {
        println!("  API key: [not set — live calls will be skipped]");
    }

    // Base URL override (via custom stream_fn or provider config)
    let base_url = env_or("LLM_BASE_URL", "OPENAI_BASE_URL");
    if let Some(ref url) = base_url {
        println!("  Base URL: {}", url);
    }

    // Thinking level & budgets
    agent.set_thinking_level(ThinkingLevel::Medium);
    agent.set_thinking_budgets(ThinkingBudgets {
        minimal: Some(128),
        low: Some(512),
        medium: Some(2048),
        high: Some(8192),
    });
    println!("  Thinking: Medium (custom budgets)");

    // Transport
    agent.set_transport(Transport::Sse);
    println!("  Transport: SSE");

    // Session ID for caching
    agent.set_session_id("example-session-001");
    println!("  Session ID: {:?}", agent.session_id());

    // Max retry delay
    agent.set_max_retry_delay_ms(Some(5000));
    println!("  Max retry delay: 5000ms");

    // Max turns (prevent runaway loops)
    agent.set_max_turns(10);
    println!("  Max turns: 10");

    // Security config
    let mut security = SecurityConfig::default();
    security.agent.max_parallel_tool_calls = 4;
    security.agent.tool_execution_timeout_secs = 30;
    agent.set_security_config(security);
    println!("  Max parallel tools: 4, tool timeout: 30s");

    // ========================================================================
    // 3. Define tools
    // ========================================================================
    println!("\n--- 3. Define Tools ---");

    let calc_tool = AgentTool::new(
        "calculate",
        "Calculator",
        "Evaluate a math expression and return the result",
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '2 + 3 * 4'"
                }
            },
            "required": ["expression"]
        }),
    );

    let search_tool = AgentTool::new(
        "search",
        "Web Search",
        "Search the web for information",
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }),
    );

    agent.set_tools(vec![calc_tool, search_tool]);
    println!("  Registered 2 tools: calculate, search");

    // ========================================================================
    // 4. Register tool executor
    // ========================================================================
    println!("\n--- 4. Register Tool Executor ---");

    agent.set_tool_executor_simple(|name, _id, args| {
        let name = name.to_string();
        let args = args.clone();
        async move {
            match name.as_str() {
                "calculate" => {
                    let expr = args["expression"].as_str().unwrap_or("0");
                    // Simple mock calculation
                    AgentToolResult::text(format!("Result of '{}' = 42", expr))
                }
                "search" => {
                    let query = args["query"].as_str().unwrap_or("");
                    AgentToolResult::text(format!(
                        "Search results for '{}': [1] Example result from web search.",
                        query
                    ))
                }
                _ => AgentToolResult::error(format!("Unknown tool: {}", name)),
            }
        }
    });
    println!("  Tool executor registered (simple mode)");

    // ========================================================================
    // 5. Set up hooks
    // ========================================================================
    println!("\n--- 5. Set Up Hooks ---");

    // beforeToolCall: log and optionally block
    agent.set_before_tool_call(|ctx: BeforeToolCallContext| async move {
        println!(
            "    [beforeToolCall] tool={}, args={}",
            ctx.tool_call.name,
            serde_json::to_string(&ctx.args).unwrap_or_default()
        );
        // Example: block a hypothetical "dangerous_tool"
        if ctx.tool_call.name == "dangerous_tool" {
            Some(BeforeToolCallResult::blocked("Blocked for safety"))
        } else {
            None // allow
        }
    });
    println!("  beforeToolCall hook: set");

    // afterToolCall: log results
    agent.set_after_tool_call(|ctx: AfterToolCallContext| async move {
        println!(
            "    [afterToolCall] tool={}, is_error={}",
            ctx.tool_call.name, ctx.is_error
        );
        None // keep original result
    });
    println!("  afterToolCall hook: set");

    // onPayload: log the outgoing request body
    let payload_count = Arc::new(AtomicUsize::new(0));
    let pc = payload_count.clone();
    agent.set_on_payload(move |payload, model| {
        let pc = pc.clone();
        async move {
            let n = pc.fetch_add(1, Ordering::SeqCst) + 1;
            println!(
                "    [onPayload #{}] model={}, keys={:?}",
                n,
                model.id,
                payload.as_object().map(|o| o.keys().collect::<Vec<_>>())
            );
            None // pass through unchanged
        }
    });
    println!("  onPayload hook: set");

    // ========================================================================
    // 6. Context pipeline
    // ========================================================================
    println!("\n--- 6. Context Pipeline ---");

    // transformContext: log + pass through
    agent.set_transform_context(|messages| async move {
        println!(
            "    [transformContext] {} messages in pipeline",
            messages.len()
        );
        messages // pass through (could prune here)
    });
    println!("  transformContext: set (logging)");

    // convertToLlm: handle custom messages
    agent.set_convert_to_llm(|messages| async move {
        println!(
            "    [convertToLlm] converting {} agent messages",
            messages.len()
        );
        messages
            .into_iter()
            .filter_map(|m| match m {
                AgentMessage::Custom { message_type, data } => {
                    // Convert "note" custom messages to user messages
                    if message_type == "note" {
                        let text = data["text"].as_str().unwrap_or("[note]");
                        Some(Message::User(UserMessage::text(format!(
                            "[Note: {}]",
                            text
                        ))))
                    } else {
                        None // skip other custom types
                    }
                }
                other => {
                    let opt: Option<Message> = other.into();
                    opt
                }
            })
            .collect()
    });
    println!("  convertToLlm: set (with custom message handling)");

    // ========================================================================
    // 7. Event subscription
    // ========================================================================
    println!("\n--- 7. Event Subscription ---");

    let event_count = Arc::new(AtomicUsize::new(0));
    let ec = event_count.clone();
    let _unsub = agent.subscribe(move |event: &AgentEvent| {
        ec.fetch_add(1, Ordering::SeqCst);
        match event {
            AgentEvent::AgentStart => println!("  [event] AgentStart"),
            AgentEvent::TurnStart => println!("  [event] TurnStart"),
            AgentEvent::MessageUpdate {
                assistant_event, ..
            } => {
                if let AssistantMessageEvent::TextDelta { delta, .. } = assistant_event.as_ref() {
                    print!("{}", delta); // Stream text to stdout
                }
            }
            AgentEvent::MessageStart { .. } => println!("  [event] MessageStart"),
            AgentEvent::MessageEnd { .. } => println!("  [event] MessageEnd"),
            AgentEvent::MessageDiscarded { reason, .. } => {
                println!("  [event] MessageDiscarded: {}", reason);
            }
            AgentEvent::ToolExecutionStart {
                tool_name, args, ..
            } => {
                println!("  [event] ToolExecutionStart: {} args={}", tool_name, args);
            }
            AgentEvent::ToolExecutionUpdate {
                tool_name,
                partial_result,
                ..
            } => {
                println!(
                    "  [event] ToolExecutionUpdate: {} partial={}",
                    tool_name, partial_result
                );
            }
            AgentEvent::ToolExecutionEnd {
                tool_name,
                is_error,
                ..
            } => {
                println!(
                    "  [event] ToolExecutionEnd: {} error={}",
                    tool_name, is_error
                );
            }
            AgentEvent::TurnEnd { tool_results, .. } => {
                println!("  [event] TurnEnd (tool_results={})", tool_results.len());
            }
            AgentEvent::TurnRetrying { delay_ms, .. } => {
                println!("  [event] TurnRetrying (delay_ms={})", delay_ms);
            }
            AgentEvent::AgentEnd { messages } => {
                println!("  [event] AgentEnd ({} new messages)", messages.len());
            }
        }
    });
    println!("  Subscribed to all events");

    // ========================================================================
    // 8. Custom messages
    // ========================================================================
    println!("\n--- 8. Custom Messages ---");

    // Inject a custom note that our convertToLlm will handle
    agent.append_message(AgentMessage::Custom {
        message_type: "note".to_string(),
        data: serde_json::json!({
            "text": "The user prefers Python for code examples."
        }),
    });
    println!("  Injected custom 'note' message");

    // ========================================================================
    // 9. Provider (auto-registered)
    // ========================================================================
    println!("\n--- 9. Provider ---");
    // Built-in providers are auto-registered on first access via get_provider().
    // No manual register_provider() calls needed.
    println!("  Providers auto-registered on demand (no manual setup required)");

    // ========================================================================
    // 10. Run the agent
    // ========================================================================
    println!("\n--- 10. Run Agent ---");

    if api_key.is_some() {
        println!("  Sending prompt...\n");
        match agent
            .prompt("What is 15 * 7? Use the calculator tool.")
            .await
        {
            Ok(messages) => {
                println!("\n\n  Prompt completed: {} new messages", messages.len());
            }
            Err(e) => {
                println!("  Prompt error: {}", e);
            }
        }
    } else {
        println!("  [Skipped] No API key set.");
        println!("  Set LLM_API_KEY to run with a live provider.");
    }

    // ========================================================================
    // 11. State inspection
    // ========================================================================
    println!("\n--- 11. State Inspection ---");

    let state = agent.state();
    println!("  Messages in history: {}", state.message_count());
    println!("  Is streaming: {}", state.is_streaming());
    println!(
        "  Error: {:?}",
        state.error.read().as_deref().unwrap_or("none")
    );

    // Consistent snapshot (combines state + config)
    let snapshot = agent.snapshot();
    println!("  Snapshot message count: {}", snapshot.message_count);
    println!("  Snapshot model: {}", snapshot.model.id);
    println!("  Snapshot thinking: {:?}", snapshot.thinking_level);

    // Session ID
    println!("  Session ID: {:?}", agent.session_id());

    // Queues
    println!("  Has queued messages: {}", agent.has_queued_messages());

    // ========================================================================
    // 12. Steering & follow-up (demonstration)
    // ========================================================================
    println!("\n--- 12. Steering & Follow-up Queues ---");

    // These would be used from another task/thread while the agent is running
    agent.steer(AgentMessage::from("Actually, use metric units."));
    agent.follow_up(AgentMessage::from("Now summarize the results."));
    println!("  Steering queue: 1 message");
    println!("  Follow-up queue: 1 message");
    println!("  Has queued: {}", agent.has_queued_messages());

    // Clear for demonstration
    agent.clear_all_queues();
    println!("  Cleared all queues: {}", !agent.has_queued_messages());

    // ========================================================================
    // 13. Dynamic API key resolver (demonstration)
    // ========================================================================
    println!("\n--- 13. Dynamic API Key ---");

    agent.set_get_api_key(|provider_name: &str| {
        let provider_name = provider_name.to_string();
        async move {
            println!("    [getApiKey] resolving for provider: {}", provider_name);
            // In practice: fetch from a vault, refresh OAuth token, etc.
            None // fall back to the static key
        }
    });
    println!("  Dynamic API key resolver: set");

    // ========================================================================
    // 14. Reset
    // ========================================================================
    println!("\n--- 14. Reset ---");

    agent.reset();
    println!("  Agent reset. Messages: {}", state.message_count());
    println!("  Session ID after reset: {:?}", agent.session_id());

    // ========================================================================
    // Summary
    // ========================================================================
    println!("\n--- Summary ---");
    println!(
        "  Total events received: {}",
        event_count.load(Ordering::SeqCst)
    );
    println!(
        "  Payload hooks fired: {}",
        payload_count.load(Ordering::SeqCst)
    );
    println!("\n=== Agent Example Complete ===");

    Ok(())
}

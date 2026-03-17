//! Basic usage example for tiy-core.
//!
//! This example demonstrates:
//! - All supported Provider types
//! - All supported API types
//! - Creating models with custom API BASE, KEY, and MODEL ID
//! - Making actual LLM API requests
//!
//! Environment variables (in priority order):
//!   LLM_API_KEY    > OPENAI_API_KEY         — API key
//!   LLM_BASE_URL   > OPENAI_BASE_URL        — Base URL override
//!   LLM_MODEL      > (default: gpt-4o-mini) — Model ID
//!
//! Run with logging: RUST_LOG=info cargo run --example basic_usage
//! For request body:  RUST_LOG=debug cargo run --example basic_usage

use futures::StreamExt;
use std::sync::Arc;
use tiy_core::{
    models::get_model,
    provider::{
        anthropic::AnthropicProvider, get_provider, google::GoogleProvider, groq::GroqProvider,
        kimi_coding::KimiCodingProvider, minimax::MiniMaxProvider, ollama::OllamaProvider,
        openai_completions::OpenAICompletionsProvider, openai_responses::OpenAIResponsesProvider,
        openrouter::OpenRouterProvider, register_provider, xai::XAIProvider, zai::ZAIProvider,
        zenmux::ZenmuxProvider,
    },
    stream::AssistantMessageEventStream,
    types::{Context, Model, Provider, StreamOptions, UserMessage},
};

/// Resolve an env var with fallback: try `primary`, then `fallback`.
fn env_or(primary: &str, fallback: &str) -> Option<String> {
    std::env::var(primary)
        .or_else(|_| std::env::var(fallback))
        .ok()
        .filter(|v| !v.is_empty())
}

fn main() {
    // Initialize tracing subscriber (controlled by RUST_LOG env var)
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("=== tiy-core Basic Usage Example ===\n");

    // ============================================
    // Part 2: Make Actual LLM Request
    // ============================================
    println!("--- Making LLM Request ---");

    // Resolve configuration from environment variables
    let api_key = env_or("LLM_API_KEY", "OPENAI_API_KEY");
    let base_url = env_or("LLM_BASE_URL", "OPENAI_BASE_URL");
    let model_id = std::env::var("LLM_MODEL")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "gpt-4o-mini".to_string());

    // Look up predefined model, or fall back to a custom model definition
    let model = get_model("openai", &model_id).unwrap_or_else(|| {
        Model::builder()
            .id(&model_id)
            .name(&model_id)
            .provider(Provider::Zenmux)
            .context_window(128000)
            .max_tokens(4096)
            .build()
            .expect("Failed to build custom model")
    });

    // Create context with messages
    let context = Context {
        system_prompt: Some("You are a helpful assistant. Answer in short sentences.".to_string()),
        messages: vec![tiy_core::types::Message::User(UserMessage::text(
            "What is the capital of France?",
        ))],
        tools: None,
    };

    println!("  Model: {} ({})", model.name, model.id);
    println!("  Provider: {}", model.provider);
    println!(
        "  Base URL: {}",
        base_url
            .as_deref()
            .or(model.base_url.as_deref())
            .unwrap_or("(default)")
    );
    println!("  Prompt: \"{}\"", "What is the capital of France?");

    // Register all supported providers into the global registry.
    // After this, providers are resolved automatically from model.provider.
    register_provider(Arc::new(OpenAICompletionsProvider::new()));
    register_provider(Arc::new(OpenAIResponsesProvider::new()));
    register_provider(Arc::new(AnthropicProvider::new()));
    register_provider(Arc::new(GoogleProvider::new()));
    register_provider(Arc::new(OllamaProvider::new()));
    register_provider(Arc::new(GroqProvider::new()));
    register_provider(Arc::new(XAIProvider::new()));
    register_provider(Arc::new(OpenRouterProvider::new()));
    register_provider(Arc::new(MiniMaxProvider::new()));
    register_provider(Arc::new(KimiCodingProvider::new()));
    register_provider(Arc::new(ZAIProvider::new()));
    register_provider(Arc::new(ZenmuxProvider::new()));

    // ============================================
    // List all registered Providers
    // ============================================
    println!("--- Registered Providers ---");
    let providers = tiy_core::provider::get_registered_providers();
    for provider in &providers {
        println!("  - {}", provider);
    }
    println!();

    // Resolve provider from the registry using model.provider
    let provider = get_provider(&model.provider)
        .expect(&format!("No provider registered for: {}", model.provider));

    match api_key {
        Some(key) => {
            println!("\n  Making request...");

            // base_url in StreamOptions overrides model.base_url
            let options = StreamOptions {
                temperature: Some(0.7),
                max_tokens: Some(8192),
                api_key: Some(key),
                base_url,
                headers: None,
                session_id: None,
            };

            // Process streaming response
            println!("\n  Response:");
            println!("  --------");

            // Use blocking async runtime for simplicity
            // NOTE: provider.stream() calls tokio::spawn internally,
            // so it must be called within an active Tokio runtime.
            let rt = tokio::runtime::Runtime::new().unwrap();
            let result = rt.block_on(async {
                // Make the request (must be inside async block for tokio::spawn)
                let stream = provider.stream(&model, &context, options);

                let mut full_response = String::new();

                // Collect events from stream
                let events: Vec<_> = stream.collect().await;

                for event in events {
                    match event {
                        tiy_core::types::AssistantMessageEvent::TextDelta { delta, .. } => {
                            print!("{}", delta);
                            full_response.push_str(&delta);
                        }
                        tiy_core::types::AssistantMessageEvent::Done { message, .. } => {
                            println!("\n  --------");
                            println!("  Stop reason: {:?}", message.stop_reason);
                            println!(
                                "  Usage: {} input, {} output tokens",
                                message.usage.input, message.usage.output
                            );
                        }
                        tiy_core::types::AssistantMessageEvent::Error { error, .. } => {
                            println!("\n  Error: {:?}", error.error_message);
                        }
                        _ => {}
                    }
                }

                Ok::<(), Box<dyn std::error::Error>>(())
            });

            if let Err(e) = result {
                println!("  Request error: {}", e);
            }
        }
        None => {
            println!("\n  Note: No API key set, skipping actual API call.");
            println!("  To make actual requests, set environment variables:");
            println!("    export LLM_API_KEY=your_key");
            println!("    export LLM_BASE_URL=https://your-proxy.com/v1  # optional");
            println!("    export LLM_MODEL=gpt-4o-mini                   # optional");
            println!("  Or use provider-specific variables:");
            println!("    export OPENAI_API_KEY=your_key");
            println!("    export OPENAI_BASE_URL=https://your-proxy.com/v1");
        }
    }

    println!("\n=== Example Complete ===");
}

// Helper function to process stream without async
#[allow(dead_code)]
fn process_stream_sync(
    stream: AssistantMessageEventStream,
) -> Result<String, Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let mut full_response = String::new();

        // Collect all events
        let events: Vec<_> = stream.collect().await;

        for event in events {
            match event {
                tiy_core::types::AssistantMessageEvent::TextDelta { delta, .. } => {
                    full_response.push_str(&delta);
                }
                tiy_core::types::AssistantMessageEvent::Done { message, .. } => {
                    println!("  Stop reason: {:?}", message.stop_reason);
                    println!(
                        "  Usage: {} input, {} output tokens",
                        message.usage.input, message.usage.output
                    );
                }
                tiy_core::types::AssistantMessageEvent::Error { error, .. } => {
                    return Err(format!("API error: {:?}", error.error_message).into());
                }
                _ => {}
            }
        }

        Ok(full_response)
    })
}

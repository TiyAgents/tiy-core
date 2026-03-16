//! Basic usage example for tiy-core.
//!
//! This example demonstrates:
//! - All supported Provider types
//! - All supported API types
//! - Creating models with custom API BASE, KEY, and MODEL ID
//! - Making actual LLM API requests

use futures::StreamExt;
use tiy_core::{
    models::get_model,
    provider::{openai_completions::OpenAICompletionsProvider, LLMProvider},
    stream::AssistantMessageEventStream,
    types::{Api, Context, Cost, InputType, Model, Provider, StreamOptions, UserMessage},
};

fn main() {
    println!("=== tiy-core Basic Usage Example ===\n");

    // ============================================
    // Part 1: List all supported Providers
    // ============================================
    println!("--- Supported Providers ---");
    let providers = tiy_core::models::get_providers();
    for provider in &providers {
        println!("  - {}", provider);
    }
    println!();

    // ============================================
    // Part 2: Predefined Models
    // ============================================
    println!("--- Predefined Models ---");
    if let Some(model) = get_model("openai", "gpt-4o-mini") {
        println!(
            "  OpenAI gpt-4o-mini: {} tokens (${:.4}/1M in)",
            model.context_window, model.cost.input
        );
    }
    if let Some(model) = get_model("anthropic", "claude-sonnet-4-20250514") {
        println!(
            "  Anthropic claude-sonnet-4: {} tokens",
            model.context_window
        );
    }
    if let Some(model) = get_model("google", "gemini-2.5-flash") {
        println!("  Google gemini-2.5-flash: {} tokens", model.context_window);
    }
    println!();

    // ============================================
    // Part 3: Custom Models with API BASE, KEY, MODEL ID
    // ============================================
    println!("--- Custom Models ---");

    // Example: Groq model
    let groq_model = Model::builder()
        .id("llama-3.3-70b-versatile")
        .name("Llama 3.3 70B (Groq)")
        .api(Api::OpenAICompletions)
        .provider(Provider::Groq)
        .base_url("https://api.groq.com/openai/v1")
        .reasoning(false)
        .input(vec![InputType::Text])
        .cost(Cost::free())
        .context_window(32768)
        .max_tokens(4096)
        .build()
        .unwrap();
    println!(
        "  Groq: id={}, base_url={}",
        groq_model.id, groq_model.base_url
    );

    // Example: OpenAI (local)
    let openai_model = Model::builder()
        .id("kimi-k2.5")
        .name("Kimi K2.5")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url("https://api.lkeap.cloud.tencent.com/v3")
        .reasoning(false)
        .input(vec![InputType::Text])
        .cost(Cost::free())
        .context_window(128000)
        .max_tokens(4096)
        .build()
        .unwrap();
    println!(
        "  OpenAI: id={}, base_url={}",
        openai_model.id, openai_model.base_url
    );
    println!();

    // ============================================
    // Part 4: Make Actual LLM Request
    // ============================================
    println!("--- Making LLM Request ---");

    // Use Groq as example (fast and has free tier)
    let model = openai_model;

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
    println!("  Base URL: {}", model.base_url);
    println!("  Prompt: \"{}\"", "What is the capital of France?");

    // Create provider
    let provider = OpenAICompletionsProvider::new();

    // Set API key from environment or hardcode (for demo purposes)
    // In production, use: std::env::var("GROQ_API_KEY").unwrap()
    let api_key = std::env::var("OPENAI_API_KEY");

    match api_key {
        Ok(key) if !key.is_empty() => {
            println!("\n  Making request to OPENAI API...");

            // Create stream options with API key
            let options = StreamOptions {
                temperature: Some(0.7),
                max_tokens: Some(100),
                api_key: Some(key),
                headers: None,
                session_id: None,
            };

            // Make the request
            let stream = provider.stream(&model, &context, options);

            // Process streaming response
            println!("\n  Response:");
            println!("  --------");

            // Use blocking async runtime for simplicity
            let rt = tokio::runtime::Runtime::new().unwrap();
            let result = rt.block_on(async {
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
        _ => {
            println!("\n  Note: GROQ_API_KEY not set, skipping actual API call.");
            println!("  To make actual requests:");
            println!("    1. Get an API key from https://console.groq.com");
            println!("    2. Set it: export GROQ_API_KEY=your_key");
            println!("    3. Or modify this example to use another provider");
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

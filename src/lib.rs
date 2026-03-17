//! # tiy-core
//!
//! A unified LLM API and stateful Agent runtime in Rust.
//!
//! This library provides:
//! - Unified API for multiple LLM providers (OpenAI, Anthropic, Google, Ollama, etc.)
//! - Streaming event-based responses
//! - Tool/function calling support
//! - Stateful agent with tool execution
//!
//! ## Example
//!
//! ```rust,no_run
//! use tiy_core::{provider::openai_completions::OpenAICompletionsProvider, types::*};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a model
//!     let model = Model {
//!         id: "gpt-4o-mini".to_string(),
//!         name: "GPT-4o Mini".to_string(),
//!         api: None,
//!         provider: Provider::OpenAI,
//!         base_url: Some("https://api.openai.com/v1".to_string()),
//!         reasoning: false,
//!         input: vec![InputType::Text, InputType::Image],
//!         cost: Cost::default(),
//!         context_window: 128000,
//!         max_tokens: 16384,
//!         headers: None,
//!         compat: None,
//!     };
//!
//!     // Create context
//!     let context = Context {
//!         system_prompt: Some("You are a helpful assistant.".to_string()),
//!         messages: vec![Message::User(UserMessage {
//!             role: Role::User,
//!             content: UserContent::Text("Hello!".to_string()),
//!             timestamp: chrono::Utc::now().timestamp_millis(),
//!         })],
//!         tools: None,
//!     };
//!
//!     Ok(())
//! }
//! ```

pub mod types;
pub mod provider;
pub mod stream;
pub mod transform;
pub mod validation;
pub mod thinking;
pub mod models;
pub mod agent;

// Re-export commonly used types
pub use types::{
    Api, Provider, Model, Context, Tool, Message, UserMessage, AssistantMessage,
    ToolResultMessage, ContentBlock, TextContent, ThinkingContent, ImageContent, ToolCall,
    StopReason, Role, Usage, Cost, InputType,
};

pub use stream::EventStream;

pub use agent::Agent;

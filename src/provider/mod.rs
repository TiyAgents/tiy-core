//! LLM service provider facades.
//!
//! Each provider represents a specific LLM service vendor.
//! Providers internally delegate to protocol implementations in `crate::protocol`.

pub mod openai;
pub mod anthropic;
pub mod google;
pub mod ollama;
pub mod xai;
pub mod groq;
pub mod openrouter;
pub mod zai;
pub mod minimax;
pub mod kimi_coding;
pub mod zenmux;

// Re-export protocol infrastructure for backward compatibility
pub use crate::protocol::{
    global_registry, register_provider, get_provider, get_registered_providers, clear_providers,
    LLMProtocol, BoxedProtocol, ArcProtocol, ProtocolRegistry,
};

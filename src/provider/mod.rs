//! Provider implementations for various LLM APIs.

mod registry;
mod traits;

pub mod openai_completions;
pub mod openai_responses;
pub mod anthropic;
pub mod google;
pub mod ollama;
pub mod minimax;
pub mod kimi_coding;
pub mod xai;
pub mod groq;
pub mod openrouter;
pub mod zai;
pub mod zenmux;

pub use registry::*;
pub use traits::*;

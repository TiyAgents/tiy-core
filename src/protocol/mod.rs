//! Wire-format protocol implementations.
//!
//! Four base protocols for LLM API communication:
//! - `openai_completions` — OpenAI Chat Completions (`/chat/completions`)
//! - `openai_responses` — OpenAI Responses API (`/responses`)
//! - `anthropic` — Anthropic Messages API (`/messages`)
//! - `google` — Google Generative AI / Vertex AI
//!
//! Most users should use the provider-level types (e.g., `provider::OpenAIProvider`,
//! `provider::AnthropicProvider`) rather than protocol modules directly.

mod traits;
pub mod common;

pub mod openai_completions;
pub mod openai_responses;
pub mod anthropic;
pub mod google;

pub use traits::*;   // LLMProtocol, BoxedProtocol, ArcProtocol

// Backward-compatible re-exports — registry & delegation now live in provider/
// but existing `crate::protocol::{register_provider, get_provider, ...}` paths still work.
pub use crate::provider::{
    global_registry, register_provider, get_provider, get_registered_providers, clear_providers,
    ProtocolRegistry,
};

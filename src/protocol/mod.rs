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
mod registry;
pub mod common;

#[macro_use]
pub(crate) mod delegation;

pub mod openai_completions;
pub mod openai_responses;
pub mod anthropic;
pub mod google;

pub use traits::*;   // LLMProtocol, BoxedProtocol, ArcProtocol
pub use registry::*; // ProtocolRegistry, register_provider, get_provider, etc.

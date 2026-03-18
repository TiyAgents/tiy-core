//! xAI provider (reuses OpenAI Completions protocol).
//!
//! xAI (Grok models) uses the OpenAI Chat Completions API at:
//! - `https://api.x.ai/v1`
//!
//! Compat customizations from pi-mono:
//! - `supports_store: false` (non-standard)
//! - `supports_developer_role: false` (non-standard)
//! - `supports_reasoning_effort: false` (xAI/Grok does not support reasoning_effort)
//! - `thinking_format: "openai"` (standard)
//!
//! This provider delegates to `OpenAICompletionsProvider` with xAI-specific defaults.

use crate::stream::AssistantMessageEventStream;
use crate::types::*;

define_openai_delegation_provider! {
    name: XAIProvider,
    doc: "xAI provider (OpenAI-compatible, Grok models).",
    provider_type: Provider::XAI,
    env_var: "XAI_API_KEY",
    default_compat: || OpenAICompletionsCompat {
        supports_store: false,
        supports_developer_role: false,
        supports_reasoning_effort: false,
        thinking_format: "openai".to_string(),
        ..Default::default()
    },
}

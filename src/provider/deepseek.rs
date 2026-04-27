//! DeepSeek provider (reuses OpenAI Completions protocol).
//!
//! DeepSeek uses the OpenAI Chat Completions API at:
//! - `https://api.deepseek.com`
//!
//! Compat customizations from pi-mono:
//! - `supports_store: false` (non-standard)
//! - `supports_developer_role: false` (non-standard)
//! - `supports_reasoning_effort: true` (DeepSeek supports reasoning)
//! - `thinking_format: "openai"` (standard)
//!
//! This provider delegates to `OpenAICompletionsProtocol` with DeepSeek-specific defaults.

use crate::stream::AssistantMessageEventStream;
use crate::types::*;

define_openai_delegation_provider! {
    name: DeepSeekProvider,
    doc: "DeepSeek provider (OpenAI-compatible, DeepSeek-R1/V3 models).",
    provider_type: Provider::DeepSeek,
    env_var: "DEEPSEEK_API_KEY",
    default_compat: || OpenAICompletionsCompat {
        supports_store: false,
        supports_developer_role: false,
        thinking_format: "openai".to_string(),
        reasoning_content_constrained: true,
        ..Default::default()
    },
}

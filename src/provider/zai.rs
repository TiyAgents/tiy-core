//! ZAI provider (reuses OpenAI Completions protocol with thinking customization).
//!
//! ZAI uses the OpenAI Chat Completions API at:
//! - `https://api.z.ai/api/coding/paas/v4`
//!
//! Thinking customizations from pi-mono:
//! - `thinking_format: "zai"` — uses `enable_thinking: true/false` top-level parameter
//!   instead of OpenAI's `reasoning_effort`
//! - `supports_developer_role: false` (non-standard)
//! - `supports_reasoning_effort: false` (uses `enable_thinking` instead)
//! - `supports_store: false` (non-standard)
//!
//! This provider delegates to `OpenAICompletionsProvider` with ZAI-specific compat.

use crate::stream::AssistantMessageEventStream;
use crate::types::*;

define_openai_delegation_provider! {
    name: ZAIProvider,
    doc: "ZAI provider (OpenAI-compatible with custom thinking format).",
    provider_type: Provider::ZAI,
    env_var: "ZAI_API_KEY",
    default_compat: || OpenAICompletionsCompat {
        supports_store: false,
        supports_developer_role: false,
        supports_reasoning_effort: false,
        thinking_format: "zai".to_string(),
        ..Default::default()
    },
}

//! Kimi Coding provider (reuses Anthropic Messages protocol).
//!
//! Kimi Coding exposes an Anthropic-compatible API endpoint at:
//! - `https://api.kimi.com/coding`
//!
//! This provider delegates all streaming to `AnthropicProvider`.

use crate::stream::AssistantMessageEventStream;
use crate::types::*;

define_anthropic_delegation_provider! {
    name: KimiCodingProvider,
    doc: "Kimi Coding provider (Anthropic-compatible).",
    provider_type: Provider::KimiCoding,
    env_var: "KIMI_API_KEY",
}

//! OpenRouter provider (reuses OpenAI Completions protocol with routing extensions).
//!
//! OpenRouter uses the OpenAI Chat Completions API at:
//! - `https://openrouter.ai/api/v1`
//!
//! Routing extensions from pi-mono:
//! - `provider` field in request body: `model.compat.open_router_routing` preferences
//!   for controlling which upstream providers to route to (e.g., `{"only": ["anthropic"]}`)
//! - Anthropic cache control: When model ID starts with `anthropic/`, adds
//!   `cache_control: { type: "ephemeral" }` to the last user/assistant text part
//! - Error metadata: OpenRouter may include `error.metadata.raw` with additional
//!   upstream provider error details
//!
//! This provider delegates to `OpenAICompletionsProvider`.

use crate::stream::AssistantMessageEventStream;
use crate::types::*;

define_openai_delegation_provider! {
    name: OpenRouterProvider,
    doc: "OpenRouter provider (OpenAI-compatible with routing extensions).",
    provider_type: Provider::OpenRouter,
    env_var: "OPENROUTER_API_KEY",
}

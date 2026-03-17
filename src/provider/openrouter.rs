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

use crate::provider::LLMProvider;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// OpenRouter provider (OpenAI-compatible with routing extensions).
pub struct OpenRouterProvider {
    default_api_key: Option<String>,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider.
    pub fn new() -> Self {
        Self {
            default_api_key: None,
        }
    }

    /// Create a provider with a default API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            default_api_key: Some(api_key.into()),
        }
    }

    /// Resolve API key from options, self, or environment.
    fn resolve_api_key(&self, options: &StreamOptions) -> Option<String> {
        if let Some(ref key) = options.api_key {
            return Some(key.clone());
        }
        if let Some(ref key) = self.default_api_key {
            return Some(key.clone());
        }
        std::env::var("OPENROUTER_API_KEY").ok()
    }
}

impl Default for OpenRouterProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for OpenRouterProvider {
    fn api_type(&self) -> Api {
        Api::OpenAICompletions
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let mut opts = options;
        if opts.api_key.is_none() {
            opts.api_key = self.resolve_api_key(&opts);
        }

        let provider = super::openai_completions::OpenAICompletionsProvider::new();
        provider.stream(model, context, opts)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let mut opts = options;
        if opts.base.api_key.is_none() {
            opts.base.api_key = self.resolve_api_key(&opts.base);
        }

        let provider = super::openai_completions::OpenAICompletionsProvider::new();
        provider.stream_simple(model, context, opts)
    }
}

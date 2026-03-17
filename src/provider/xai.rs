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

use crate::provider::LLMProvider;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// xAI provider (OpenAI-compatible, Grok models).
pub struct XAIProvider {
    default_api_key: Option<String>,
}

impl XAIProvider {
    /// Create a new xAI provider.
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
        std::env::var("XAI_API_KEY").ok()
    }

    /// Get xAI-specific compat settings.
    /// xAI is a non-standard provider: no store, no developer role, no reasoning_effort.
    pub fn default_compat() -> OpenAICompletionsCompat {
        OpenAICompletionsCompat {
            supports_store: false,
            supports_developer_role: false,
            supports_reasoning_effort: false,
            thinking_format: "openai".to_string(),
            ..Default::default()
        }
    }
}

impl Default for XAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for XAIProvider {
    fn provider_type(&self) -> Provider {
        Provider::XAI
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

        // Apply xAI compat defaults if model doesn't have explicit compat
        let model = if model.compat.is_none() {
            let mut m = model.clone();
            m.compat = Some(Self::default_compat());
            m
        } else {
            model.clone()
        };

        let provider = super::openai_completions::OpenAICompletionsProvider::new();
        provider.stream(&model, context, opts)
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

        let model = if model.compat.is_none() {
            let mut m = model.clone();
            m.compat = Some(Self::default_compat());
            m
        } else {
            model.clone()
        };

        let provider = super::openai_completions::OpenAICompletionsProvider::new();
        provider.stream_simple(&model, context, opts)
    }
}

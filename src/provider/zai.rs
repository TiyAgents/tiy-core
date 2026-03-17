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

use crate::provider::LLMProvider;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// ZAI provider (OpenAI-compatible with custom thinking format).
pub struct ZAIProvider {
    default_api_key: Option<String>,
}

impl ZAIProvider {
    /// Create a new ZAI provider.
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
        std::env::var("ZAI_API_KEY").ok()
    }

    /// Get ZAI-specific compat settings.
    /// ZAI uses `enable_thinking` parameter instead of `reasoning_effort`,
    /// does not support developer role or store.
    pub fn default_compat() -> OpenAICompletionsCompat {
        OpenAICompletionsCompat {
            supports_store: false,
            supports_developer_role: false,
            supports_reasoning_effort: false,
            thinking_format: "zai".to_string(),
            ..Default::default()
        }
    }
}

impl Default for ZAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for ZAIProvider {
    fn provider_type(&self) -> Provider {
        Provider::ZAI
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

        // Apply ZAI compat defaults if model doesn't have explicit compat
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

//! Kimi Coding provider (reuses Anthropic Messages protocol).
//!
//! Kimi Coding exposes an Anthropic-compatible API endpoint at:
//! - `https://api.kimi.com/coding`
//!
//! This provider delegates all streaming to `AnthropicProvider`.

use crate::provider::LLMProvider;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// Kimi Coding provider (Anthropic-compatible).
pub struct KimiCodingProvider {
    default_api_key: Option<String>,
}

impl KimiCodingProvider {
    /// Create a new Kimi Coding provider.
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
        std::env::var("KIMI_API_KEY").ok()
    }
}

impl Default for KimiCodingProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for KimiCodingProvider {
    fn provider_type(&self) -> Provider {
        Provider::KimiCoding
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

        let provider = super::anthropic::AnthropicProvider::new();
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

        let provider = super::anthropic::AnthropicProvider::new();
        provider.stream_simple(model, context, opts)
    }
}

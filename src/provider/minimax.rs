//! MiniMax provider (reuses Anthropic Messages protocol).
//!
//! MiniMax exposes an Anthropic-compatible API endpoint at:
//! - `https://api.minimax.io/anthropic` (international)
//! - `https://api.minimaxi.com/anthropic` (minimax-cn, China mainland)
//!
//! This provider delegates all streaming to `AnthropicProvider`.
//!
//! Note: MiniMax has a dual env var for API key resolution based on
//! the provider variant (MiniMax vs MiniMaxCN), which requires a
//! custom `resolve_api_key` instead of using the delegation macro.

use crate::provider::LLMProvider;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// MiniMax provider (Anthropic-compatible).
pub struct MiniMaxProvider {
    default_api_key: Option<String>,
}

impl MiniMaxProvider {
    /// Create a new MiniMax provider.
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
    ///
    /// Uses `MINIMAX_CN_API_KEY` for `MiniMaxCN` provider variant,
    /// `MINIMAX_API_KEY` for all others.
    fn resolve_api_key(&self, options: &StreamOptions, provider: &Provider) -> Option<String> {
        if let Some(ref key) = options.api_key {
            return Some(key.clone());
        }
        if let Some(ref key) = self.default_api_key {
            return Some(key.clone());
        }
        let env_var = match provider {
            Provider::MiniMaxCN => "MINIMAX_CN_API_KEY",
            _ => "MINIMAX_API_KEY",
        };
        std::env::var(env_var).ok()
    }
}

impl Default for MiniMaxProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for MiniMaxProvider {
    fn provider_type(&self) -> Provider {
        Provider::MiniMax
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let mut opts = options;
        if opts.api_key.is_none() {
            opts.api_key = self.resolve_api_key(&opts, &model.provider);
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
            opts.base.api_key = self.resolve_api_key(&opts.base, &model.provider);
        }
        let provider = super::anthropic::AnthropicProvider::new();
        provider.stream_simple(model, context, opts)
    }
}

//! Zenmux provider (routes to Anthropic Messages or Google Generative AI based on model ID).
//!
//! Zenmux is a multi-protocol proxy that supports:
//! - Anthropic Messages protocol at `https://zenmux.ai/api/anthropic`
//! - Google Generative AI protocol at `https://zenmux.ai/api/vertex-ai`
//!
//! Routing logic:
//! - If the model ID contains "google" or "gemini" (case-insensitive),
//!   routes to Google Generative AI protocol
//! - Otherwise, routes to Anthropic Messages protocol
//!
//! API key environment variable: `ZENMUX_API_KEY`

use crate::provider::LLMProvider;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// Default Anthropic endpoint for Zenmux.
const ZENMUX_ANTHROPIC_BASE_URL: &str = "https://zenmux.ai/api/anthropic";

/// Default Google endpoint for Zenmux.
const ZENMUX_GOOGLE_BASE_URL: &str = "https://zenmux.ai/api/vertex-ai";

/// Zenmux provider (multi-protocol proxy).
pub struct ZenmuxProvider {
    default_api_key: Option<String>,
}

impl ZenmuxProvider {
    /// Create a new Zenmux provider.
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
        std::env::var("ZENMUX_API_KEY").ok()
    }

    /// Check if a model ID should be routed to Google Generative AI.
    /// Models with "google" or "gemini" in their ID use the Google protocol.
    fn is_google_model(model_id: &str) -> bool {
        let lower = model_id.to_lowercase();
        lower.contains("google") || lower.contains("gemini")
    }
}

impl Default for ZenmuxProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for ZenmuxProvider {
    fn provider_type(&self) -> Provider {
        Provider::Zenmux
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

        if Self::is_google_model(&model.id) {
            // Route to Google Generative AI protocol
            let mut m = model.clone();
            let needs_default = m.base_url.as_ref()
                .is_none_or(|url| url.contains("zenmux.ai") || url.is_empty());
            if needs_default {
                m.base_url = Some(ZENMUX_GOOGLE_BASE_URL.to_string());
            }
            m.api = Some(Api::GoogleGenerativeAi);

            let provider = super::google::GoogleProvider::new();
            provider.stream(&m, context, opts)
        } else {
            // Route to Anthropic Messages protocol
            let mut m = model.clone();
            let needs_default = m.base_url.as_ref()
                .is_none_or(|url| url.contains("zenmux.ai") || url.is_empty());
            if needs_default {
                m.base_url = Some(ZENMUX_ANTHROPIC_BASE_URL.to_string());
            }
            m.api = Some(Api::AnthropicMessages);

            let provider = super::anthropic::AnthropicProvider::new();
            provider.stream(&m, context, opts)
        }
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

        if Self::is_google_model(&model.id) {
            let mut m = model.clone();
            let needs_default = m.base_url.as_ref()
                .is_none_or(|url| url.contains("zenmux.ai") || url.is_empty());
            if needs_default {
                m.base_url = Some(ZENMUX_GOOGLE_BASE_URL.to_string());
            }
            m.api = Some(Api::GoogleGenerativeAi);

            let provider = super::google::GoogleProvider::new();
            provider.stream_simple(&m, context, opts)
        } else {
            let mut m = model.clone();
            let needs_default = m.base_url.as_ref()
                .is_none_or(|url| url.contains("zenmux.ai") || url.is_empty());
            if needs_default {
                m.base_url = Some(ZENMUX_ANTHROPIC_BASE_URL.to_string());
            }
            m.api = Some(Api::AnthropicMessages);

            let provider = super::anthropic::AnthropicProvider::new();
            provider.stream_simple(&m, context, opts)
        }
    }
}

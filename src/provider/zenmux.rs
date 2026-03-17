//! Zenmux provider (adaptive multi-protocol proxy based on model ID).
//!
//! Zenmux is a multi-protocol proxy that supports:
//! - OpenAI Responses protocol at `https://zenmux.ai/api/v1`
//! - Google Vertex AI protocol at `https://zenmux.ai/api/vertex-ai`
//! - Anthropic Messages protocol at `https://zenmux.ai/api/anthropic/v1`
//!
//! Adaptive routing logic (when base_url is empty or starts with `https://zenmux.ai`):
//! - If the model ID contains "google" or "gemini" (case-insensitive),
//!   routes to Google Vertex AI protocol
//! - If the model ID contains "openai" or "gpt" (case-insensitive),
//!   routes to OpenAI Responses protocol
//! - Otherwise, routes to Anthropic Messages protocol
//!
//! When a custom base_url is provided (not empty and not starting with
//! `https://zenmux.ai`), the provider uses OpenAI Completions protocol
//! with the given base_url as-is.
//!
//! API key environment variable: `ZENMUX_API_KEY`

use crate::provider::LLMProvider;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// Zenmux base URL prefix used to detect adaptive routing mode.
const ZENMUX_HOST_PREFIX: &str = "https://zenmux.ai";

/// Default OpenAI Responses endpoint for Zenmux.
const ZENMUX_OPENAI_BASE_URL: &str = "https://zenmux.ai/api/v1";

/// Default Google Vertex AI endpoint for Zenmux.
const ZENMUX_GOOGLE_BASE_URL: &str = "https://zenmux.ai/api/vertex-ai";

/// Default Anthropic Messages endpoint for Zenmux.
const ZENMUX_ANTHROPIC_BASE_URL: &str = "https://zenmux.ai/api/anthropic/v1";

/// Protocol routing decision for a model.
enum ProtocolRoute {
    Google,
    OpenAI,
    Anthropic,
}

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

    /// Check if adaptive routing should be enabled.
    ///
    /// Resolves the effective base_url (options.base_url > model.base_url),
    /// then returns true when it is None, empty, or starts with `https://zenmux.ai`.
    fn should_adapt(options_base_url: &Option<String>, model_base_url: &Option<String>) -> bool {
        let effective = options_base_url.as_deref().or(model_base_url.as_deref());
        match effective {
            None => true,
            Some(url) => url.is_empty() || url.starts_with(ZENMUX_HOST_PREFIX),
        }
    }

    /// Determine protocol route based on model ID.
    fn detect_route(model_id: &str) -> ProtocolRoute {
        let lower = model_id.to_lowercase();
        if lower.contains("google") || lower.contains("gemini") {
            ProtocolRoute::Google
        } else if lower.contains("openai") || lower.contains("gpt") {
            ProtocolRoute::OpenAI
        } else {
            ProtocolRoute::Anthropic
        }
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

        let mut m = model.clone();

        if Self::should_adapt(&opts.base_url, &m.base_url) {
            // Adaptive mode: choose protocol and endpoint based on model ID.
            // Clear options.base_url so the routed endpoint in model.base_url takes effect.
            opts.base_url = None;
            match Self::detect_route(&m.id) {
                ProtocolRoute::Google => {
                    m.base_url = Some(ZENMUX_GOOGLE_BASE_URL.to_string());
                    m.api = Some(Api::GoogleVertex);
                    let provider = super::google::GoogleProvider::new();
                    provider.stream(&m, context, opts)
                }
                ProtocolRoute::OpenAI => {
                    m.base_url = Some(ZENMUX_OPENAI_BASE_URL.to_string());
                    m.api = Some(Api::OpenAIResponses);
                    let provider = super::openai_responses::OpenAIResponsesProvider::new();
                    provider.stream(&m, context, opts)
                }
                ProtocolRoute::Anthropic => {
                    m.base_url = Some(ZENMUX_ANTHROPIC_BASE_URL.to_string());
                    m.api = Some(Api::AnthropicMessages);
                    let provider = super::anthropic::AnthropicProvider::new();
                    provider.stream(&m, context, opts)
                }
            }
        } else {
            // Custom base_url: use OpenAI Completions protocol as-is
            m.api = Some(Api::OpenAICompletions);
            let provider = super::openai_completions::OpenAICompletionsProvider::new();
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

        let mut m = model.clone();

        if Self::should_adapt(&opts.base.base_url, &m.base_url) {
            opts.base.base_url = None;
            match Self::detect_route(&m.id) {
                ProtocolRoute::Google => {
                    m.base_url = Some(ZENMUX_GOOGLE_BASE_URL.to_string());
                    m.api = Some(Api::GoogleVertex);
                    let provider = super::google::GoogleProvider::new();
                    provider.stream_simple(&m, context, opts)
                }
                ProtocolRoute::OpenAI => {
                    m.base_url = Some(ZENMUX_OPENAI_BASE_URL.to_string());
                    m.api = Some(Api::OpenAIResponses);
                    let provider = super::openai_responses::OpenAIResponsesProvider::new();
                    provider.stream_simple(&m, context, opts)
                }
                ProtocolRoute::Anthropic => {
                    m.base_url = Some(ZENMUX_ANTHROPIC_BASE_URL.to_string());
                    m.api = Some(Api::AnthropicMessages);
                    let provider = super::anthropic::AnthropicProvider::new();
                    provider.stream_simple(&m, context, opts)
                }
            }
        } else {
            // Custom base_url: use OpenAI Completions protocol as-is
            m.api = Some(Api::OpenAICompletions);
            let provider = super::openai_completions::OpenAICompletionsProvider::new();
            provider.stream_simple(&m, context, opts)
        }
    }
}

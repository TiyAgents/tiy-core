//! OpenCode Go provider (adaptive multi-protocol proxy based on model ID).
//!
//! OpenCode Go is a multi-protocol proxy that supports:
//! - OpenAI Completions protocol for GLM, Kimi, and Mimo models
//! - Anthropic Messages protocol for MiniMax models
//!
//! Adaptive routing logic:
//! - If the model ID contains "minimax" (case-insensitive),
//!   routes to Anthropic Messages protocol
//! - Otherwise, routes to OpenAI Completions protocol
//!
//! Default base URL: `https://opencode.ai/zen/go/v1`
//! API key environment variable: `OPENCODE_GO_API_KEY`

use crate::protocol::LLMProtocol;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// Default base URL for OpenCode Go.
const DEFAULT_BASE_URL: &str = "https://opencode.ai/zen/go/v1";

/// Protocol routing decision for a model.
enum ProtocolRoute {
    OpenAICompletions,
    Anthropic,
}

/// OpenCode Go provider (multi-protocol proxy).
pub struct OpenCodeGoProvider {
    default_api_key: Option<String>,
}

impl OpenCodeGoProvider {
    /// Create a new OpenCode Go provider.
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
        std::env::var("OPENCODE_GO_API_KEY").ok()
    }

    /// Determine protocol route based on model ID.
    fn detect_route(model_id: &str) -> ProtocolRoute {
        let lower = model_id.to_lowercase();
        if lower.contains("minimax") {
            ProtocolRoute::Anthropic
        } else {
            ProtocolRoute::OpenAICompletions
        }
    }
}

impl Default for OpenCodeGoProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProtocol for OpenCodeGoProvider {
    fn provider_type(&self) -> Provider {
        Provider::OpenCodeGo
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

        // Set default base_url if not provided
        if m.base_url.is_none() && opts.base_url.is_none() {
            m.base_url = Some(DEFAULT_BASE_URL.to_string());
        }

        match Self::detect_route(&m.id) {
            ProtocolRoute::Anthropic => {
                m.api = Some(Api::AnthropicMessages);
                let provider = crate::protocol::anthropic::AnthropicProtocol::new();
                provider.stream(&m, context, opts)
            }
            ProtocolRoute::OpenAICompletions => {
                m.api = Some(Api::OpenAICompletions);
                let provider =
                    crate::protocol::openai_completions::OpenAICompletionsProtocol::new();
                provider.stream(&m, context, opts)
            }
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

        // Set default base_url if not provided
        if m.base_url.is_none() && opts.base.base_url.is_none() {
            m.base_url = Some(DEFAULT_BASE_URL.to_string());
        }

        match Self::detect_route(&m.id) {
            ProtocolRoute::Anthropic => {
                m.api = Some(Api::AnthropicMessages);
                let provider = crate::protocol::anthropic::AnthropicProtocol::new();
                provider.stream_simple(&m, context, opts)
            }
            ProtocolRoute::OpenAICompletions => {
                m.api = Some(Api::OpenAICompletions);
                let provider =
                    crate::protocol::openai_completions::OpenAICompletionsProtocol::new();
                provider.stream_simple(&m, context, opts)
            }
        }
    }
}

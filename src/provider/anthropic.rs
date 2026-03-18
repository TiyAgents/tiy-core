//! Anthropic provider — uses Anthropic Messages protocol.

use crate::protocol::anthropic::AnthropicProtocol;
use crate::protocol::LLMProtocol;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// Anthropic provider.
///
/// Internally uses the Anthropic Messages protocol.
pub struct AnthropicProvider {
    inner: AnthropicProtocol,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider.
    pub fn new() -> Self {
        Self {
            inner: AnthropicProtocol::new(),
        }
    }

    /// Create a provider with a default API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            inner: AnthropicProtocol::with_api_key(api_key),
        }
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProtocol for AnthropicProvider {
    fn provider_type(&self) -> Provider {
        Provider::Anthropic
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        self.inner.stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.inner.stream_simple(model, context, options)
    }
}

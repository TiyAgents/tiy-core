//! Unified OpenAI provider — uses Responses protocol.
//!
//! OpenAI Completions protocol is for third-party compatible providers
//! (xAI, Groq, OpenRouter, etc.), not for direct OpenAI usage.

use crate::protocol::openai_responses::OpenAIResponsesProtocol;
use crate::protocol::LLMProtocol;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// Unified OpenAI provider.
///
/// Internally uses the OpenAI Responses protocol.
pub struct OpenAIProvider {
    inner: OpenAIResponsesProtocol,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider.
    pub fn new() -> Self {
        Self {
            inner: OpenAIResponsesProtocol::new(),
        }
    }

    /// Create a provider with a default API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAIResponsesProtocol::with_api_key(api_key),
        }
    }
}

impl Default for OpenAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProtocol for OpenAIProvider {
    fn provider_type(&self) -> Provider {
        Provider::OpenAI
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

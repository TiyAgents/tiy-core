//! Standalone OpenAI Responses provider facade.
//!
//! This provider exposes the OpenAI Responses protocol under the
//! `Provider::OpenAIResponses` key, allowing upper-layer applications to
//! explicitly select the Responses API when custom base URLs are provided.
//!
//! For direct OpenAI usage, prefer `Provider::OpenAI` (which also uses the
//! Responses protocol internally).

use crate::protocol::openai_responses::OpenAIResponsesProtocol;
use crate::protocol::LLMProtocol;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// Standalone OpenAI Responses provider.
///
/// Delegates to [`OpenAIResponsesProtocol`] but registers itself as
/// `Provider::OpenAIResponses` so it can be resolved independently from
/// `Provider::OpenAI`.
pub struct OpenAIResponsesProvider {
    inner: OpenAIResponsesProtocol,
}

impl OpenAIResponsesProvider {
    /// Create a new OpenAI Responses provider.
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

impl Default for OpenAIResponsesProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProtocol for OpenAIResponsesProvider {
    fn provider_type(&self) -> Provider {
        Provider::OpenAIResponses
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

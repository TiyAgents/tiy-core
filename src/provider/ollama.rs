//! Ollama API provider (OpenAI compatible).

use crate::protocol::LLMProtocol;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use crate::types::{SimpleStreamOptions, StreamOptions};
use async_trait::async_trait;

/// Ollama API provider.
pub struct OllamaProvider {
    base_url: String,
}

impl OllamaProvider {
    /// Create a new Ollama provider with default URL.
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:11434/v1".to_string(),
        }
    }

    /// Create an Ollama provider with custom URL.
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProtocol for OllamaProvider {
    fn provider_type(&self) -> Provider {
        Provider::Ollama
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        // Ollama is OpenAI-compatible, so we can use the OpenAI Completions provider
        // with the Ollama base URL
        let ollama_model = Model {
            base_url: Some(self.base_url.clone()),
            ..model.clone()
        };

        let provider = crate::protocol::openai_completions::OpenAICompletionsProtocol::new();
        provider.stream(&ollama_model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let ollama_model = Model {
            base_url: Some(self.base_url.clone()),
            ..model.clone()
        };

        let provider = crate::protocol::openai_completions::OpenAICompletionsProtocol::new();
        provider.stream_simple(&ollama_model, context, options)
    }
}

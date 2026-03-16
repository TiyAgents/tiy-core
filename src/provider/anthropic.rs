//! Anthropic Messages API provider.
//!
//! This is a placeholder module. Full implementation will be added later.

use crate::provider::LLMProvider;
use crate::types::{StreamOptions, SimpleStreamOptions};
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// Anthropic Messages API provider.
pub struct AnthropicProvider;

impl AnthropicProvider {
    /// Create a new Anthropic provider.
    pub fn new() -> Self {
        Self
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    fn api_type(&self) -> Api {
        Api::AnthropicMessages
    }

    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let stream = AssistantMessageEventStream::new_assistant_stream();
        let mut output = AssistantMessage::builder()
            .api(Api::AnthropicMessages)
            .provider(Provider::Anthropic)
            .model("unknown")
            .stop_reason(StopReason::Error)
            .build()
            .unwrap();
        output.error_message = Some("Anthropic Messages API not yet implemented".to_string());
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: output,
        });
        stream.end(None);
        stream
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.stream(model, context, options.base)
    }
}

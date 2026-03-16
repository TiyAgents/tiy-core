//! OpenAI Responses API provider (new API for o1, o3, gpt-5 models).
//!
//! This is a placeholder module. Full implementation will be added later.

use crate::provider::LLMProvider;
use crate::types::{StreamOptions, SimpleStreamOptions};
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// OpenAI Responses API provider.
pub struct OpenAIResponsesProvider;

impl OpenAIResponsesProvider {
    /// Create a new OpenAI Responses provider.
    pub fn new() -> Self {
        Self
    }
}

impl Default for OpenAIResponsesProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for OpenAIResponsesProvider {
    fn api_type(&self) -> Api {
        Api::OpenAIResponses
    }

    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let stream = AssistantMessageEventStream::new_assistant_stream();
        let mut output = AssistantMessage::builder()
            .api(Api::OpenAIResponses)
            .provider(Provider::OpenAI)
            .model("unknown")
            .stop_reason(StopReason::Error)
            .build()
            .unwrap();
        output.error_message = Some("OpenAI Responses API not yet implemented".to_string());
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

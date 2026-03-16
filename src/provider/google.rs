//! Google Generative AI provider.
//!
//! This is a placeholder module. Full implementation will be added later.

use crate::provider::LLMProvider;
use crate::types::{StreamOptions, SimpleStreamOptions};
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// Google Generative AI provider.
pub struct GoogleProvider;

impl GoogleProvider {
    /// Create a new Google provider.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GoogleProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for GoogleProvider {
    fn api_type(&self) -> Api {
        Api::GoogleGenerativeAi
    }

    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let stream = AssistantMessageEventStream::new_assistant_stream();
        let mut output = AssistantMessage::builder()
            .api(Api::GoogleGenerativeAi)
            .provider(Provider::Google)
            .model("unknown")
            .stop_reason(StopReason::Error)
            .build()
            .unwrap();
        output.error_message = Some("Google Generative AI not yet implemented".to_string());
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

//! Provider trait definitions.

use crate::stream::AssistantMessageEventStream;
use crate::types::{Context, Model, Provider, SimpleStreamOptions, StreamOptions};
use async_trait::async_trait;
use std::sync::Arc;

/// Provider trait for LLM API implementations.
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Get the provider type this implementation handles.
    fn provider_type(&self) -> Provider;

    /// Stream completion with full options.
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: StreamOptions,
    ) -> AssistantMessageEventStream;

    /// Stream completion with simplified options.
    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream;
}

/// Type alias for a boxed provider.
pub type BoxedProvider = Box<dyn LLMProvider>;

/// Type alias for an Arc-wrapped provider.
pub type ArcProvider = Arc<dyn LLMProvider>;

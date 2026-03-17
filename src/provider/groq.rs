//! Groq provider (reuses OpenAI Completions protocol).
//!
//! Groq uses the OpenAI Chat Completions API at:
//! - `https://api.groq.com/openai/v1`
//!
//! Compat customizations from pi-mono:
//! - Standard OpenAI compat (supports_store: true, supports_developer_role: true, etc.)
//! - Special reasoning_effort_map for `qwen/qwen3-32b` model: all levels map to "default"
//! - `supports_reasoning_effort: true`
//!
//! This provider delegates to `OpenAICompletionsProvider` with Groq-specific defaults.

use crate::provider::LLMProvider;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;
use std::collections::HashMap;

/// Groq provider (OpenAI-compatible).
pub struct GroqProvider {
    default_api_key: Option<String>,
}

impl GroqProvider {
    /// Create a new Groq provider.
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
        std::env::var("GROQ_API_KEY").ok()
    }

    /// Get Groq-specific compat settings.
    /// Special handling for qwen3-32b: all reasoning effort levels map to "default".
    pub fn default_compat(model_id: &str) -> OpenAICompletionsCompat {
        let reasoning_effort_map = if model_id == "qwen/qwen3-32b" {
            let mut map = HashMap::new();
            for level in &["minimal", "low", "medium", "high", "xhigh"] {
                map.insert(level.to_string(), "default".to_string());
            }
            map
        } else {
            HashMap::new()
        };

        OpenAICompletionsCompat {
            reasoning_effort_map,
            ..Default::default()
        }
    }
}

impl Default for GroqProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for GroqProvider {
    fn api_type(&self) -> Api {
        Api::OpenAICompletions
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

        // Apply Groq compat defaults if model doesn't have explicit compat
        let model = if model.compat.is_none() {
            let mut m = model.clone();
            m.compat = Some(Self::default_compat(&m.id));
            m
        } else {
            model.clone()
        };

        let provider = super::openai_completions::OpenAICompletionsProvider::new();
        provider.stream(&model, context, opts)
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

        let model = if model.compat.is_none() {
            let mut m = model.clone();
            m.compat = Some(Self::default_compat(&m.id));
            m
        } else {
            model.clone()
        };

        let provider = super::openai_completions::OpenAICompletionsProvider::new();
        provider.stream_simple(&model, context, opts)
    }
}

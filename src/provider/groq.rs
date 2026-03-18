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

use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use std::collections::HashMap;

define_openai_delegation_provider! {
    name: GroqProvider,
    doc: "Groq provider (OpenAI-compatible).",
    provider_type: Provider::Groq,
    env_var: "GROQ_API_KEY",
    model_aware_compat: |model_id: &str| {
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
    },
}

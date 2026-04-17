//! Predefined models and model registry.

mod predefined;

use crate::types::{Model, Provider};
use std::collections::HashMap;

/// Model registry for managing model definitions.
pub struct ModelRegistry {
    models: HashMap<String, HashMap<String, Model>>,
}

impl ModelRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Create a registry with predefined models.
    pub fn with_predefined() -> Self {
        let mut registry = Self::new();
        register_builtin_models(&mut registry);
        registry
    }

    /// Register a model.
    pub fn register(&mut self, model: Model) {
        let provider_key = model.provider.as_str().to_string();
        let model_key = model.id.clone();

        self.models
            .entry(provider_key)
            .or_insert_with(HashMap::new)
            .insert(model_key, model);
    }

    /// Get a model by provider and ID.
    pub fn get(&self, provider: &Provider, model_id: &str) -> Option<&Model> {
        self.models.get(provider.as_str())?.get(model_id)
    }

    /// Get all providers that have predefined models registered.
    pub fn providers(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// Get all models for a provider.
    pub fn models_for_provider(&self, provider: &Provider) -> Vec<&Model> {
        self.models
            .get(provider.as_str())
            .map(|m| m.values().collect())
            .unwrap_or_default()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::with_predefined()
    }
}

/// Register builtin models.
fn register_builtin_models(registry: &mut ModelRegistry) {
    // OpenAI models
    for model in get_openai_models() {
        registry.register(model);
    }

    // Anthropic models
    for model in get_anthropic_models() {
        registry.register(model);
    }

    // Google models
    for model in get_google_models() {
        registry.register(model);
    }

    // DeepSeek models
    for model in get_deepseek_models() {
        registry.register(model);
    }
}

fn get_openai_models() -> Vec<Model> {
    vec![
        Model::builder()
            .id("gpt-4o-mini")
            .name("GPT-4o Mini")
            .provider(Provider::OpenAI)
            .base_url("https://api.openai.com/v1")
            .reasoning(false)
            .input(vec![
                crate::types::InputType::Text,
                crate::types::InputType::Image,
            ])
            .cost(crate::types::Cost::new(0.15, 0.60, 0.075, 0.0))
            .context_window(128000)
            .max_tokens(16384)
            .build()
            .unwrap(),
        Model::builder()
            .id("gpt-4o")
            .name("GPT-4o")
            .provider(Provider::OpenAI)
            .base_url("https://api.openai.com/v1")
            .reasoning(false)
            .input(vec![
                crate::types::InputType::Text,
                crate::types::InputType::Image,
            ])
            .cost(crate::types::Cost::new(2.50, 10.00, 1.25, 0.0))
            .context_window(128000)
            .max_tokens(16384)
            .build()
            .unwrap(),
        Model::builder()
            .id("gpt-4.1")
            .name("GPT-4.1")
            .provider(Provider::OpenAI)
            .base_url("https://api.openai.com/v1")
            .reasoning(false)
            .input(vec![
                crate::types::InputType::Text,
                crate::types::InputType::Image,
            ])
            .cost(crate::types::Cost::new(2.0, 8.0, 0.5, 0.0))
            .context_window(1047576)
            .max_tokens(32768)
            .build()
            .unwrap(),
        Model::builder()
            .id("o3")
            .name("o3")
            .provider(Provider::OpenAI)
            .base_url("https://api.openai.com/v1")
            .reasoning(true)
            .input(vec![
                crate::types::InputType::Text,
                crate::types::InputType::Image,
            ])
            .cost(crate::types::Cost::new(10.0, 40.0, 2.5, 0.0))
            .context_window(200000)
            .max_tokens(100000)
            .build()
            .unwrap(),
    ]
}

fn get_anthropic_models() -> Vec<Model> {
    vec![
        Model::builder()
            .id("claude-sonnet-4-20250514")
            .name("Claude Sonnet 4")
            .provider(Provider::Anthropic)
            .base_url("https://api.anthropic.com/v1")
            .reasoning(true)
            .input(vec![
                crate::types::InputType::Text,
                crate::types::InputType::Image,
            ])
            .cost(crate::types::Cost::new(3.0, 15.0, 0.30, 3.75))
            .context_window(200000)
            .max_tokens(16000)
            .build()
            .unwrap(),
        Model::builder()
            .id("claude-opus-4-20250514")
            .name("Claude Opus 4")
            .provider(Provider::Anthropic)
            .base_url("https://api.anthropic.com/v1")
            .reasoning(true)
            .input(vec![
                crate::types::InputType::Text,
                crate::types::InputType::Image,
            ])
            .cost(crate::types::Cost::new(15.0, 75.0, 1.50, 18.75))
            .context_window(200000)
            .max_tokens(32000)
            .build()
            .unwrap(),
        Model::builder()
            .id("claude-opus-4-7")
            .name("Claude Opus 4.7")
            .provider(Provider::Anthropic)
            .base_url("https://api.anthropic.com/v1")
            .reasoning(true)
            .input(vec![
                crate::types::InputType::Text,
                crate::types::InputType::Image,
            ])
            .cost(crate::types::Cost::new(5.0, 25.0, 0.50, 6.25))
            .context_window(1000000)
            .max_tokens(128000)
            .build()
            .unwrap(),
    ]
}

fn get_google_models() -> Vec<Model> {
    vec![Model::builder()
        .id("gemini-2.5-flash")
        .name("Gemini 2.5 Flash")
        .provider(Provider::Google)
        .base_url("https://generativelanguage.googleapis.com/v1beta")
        .reasoning(true)
        .input(vec![
            crate::types::InputType::Text,
            crate::types::InputType::Image,
        ])
        .cost(crate::types::Cost::new(0.075, 0.30, 0.01875, 0.0))
        .context_window(1048576)
        .max_tokens(65536)
        .build()
        .unwrap()]
}

fn get_deepseek_models() -> Vec<Model> {
    vec![
        Model::builder()
            .id("deepseek-r1")
            .name("DeepSeek-R1")
            .provider(Provider::DeepSeek)
            .base_url("https://api.deepseek.com")
            .reasoning(true)
            .input(vec![crate::types::InputType::Text])
            .cost(crate::types::Cost::new(0.55, 2.19, 0.14, 0.0))
            .context_window(131072)
            .max_tokens(65536)
            .build()
            .unwrap(),
        Model::builder()
            .id("deepseek-v3-0324")
            .name("DeepSeek-V3-0324")
            .provider(Provider::DeepSeek)
            .base_url("https://api.deepseek.com")
            .reasoning(true)
            .input(vec![crate::types::InputType::Text])
            .cost(crate::types::Cost::new(0.27, 1.10, 0.07, 0.0))
            .context_window(131072)
            .max_tokens(65536)
            .build()
            .unwrap(),
        Model::builder()
            .id("deepseek-chat")
            .name("DeepSeek-V3")
            .provider(Provider::DeepSeek)
            .base_url("https://api.deepseek.com")
            .reasoning(false)
            .input(vec![crate::types::InputType::Text])
            .cost(crate::types::Cost::new(0.27, 1.10, 0.07, 0.0))
            .context_window(131072)
            .max_tokens(8192)
            .build()
            .unwrap(),
    ]
}

/// Global model registry (single instance shared by all functions).
static GLOBAL_MODEL_REGISTRY: once_cell::sync::Lazy<ModelRegistry> =
    once_cell::sync::Lazy::new(ModelRegistry::with_predefined);

/// Get a model by provider and ID.
pub fn get_model(provider: impl Into<String>, model_id: impl Into<String>) -> Option<Model> {
    let provider = provider.into();
    let model_id = model_id.into();

    // Try to parse provider
    let provider_enum: Provider = provider.clone().into();

    GLOBAL_MODEL_REGISTRY
        .get(&provider_enum, &model_id)
        .cloned()
}

/// Get all providers that have predefined models in the global model registry.
pub fn get_providers() -> Vec<String> {
    GLOBAL_MODEL_REGISTRY.providers()
}

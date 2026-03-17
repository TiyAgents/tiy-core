//! Tests for models/registry module and provider/registry module.

use tiy_core::types::*;
use tiy_core::models::ModelRegistry;

// ============================================================================
// ModelRegistry tests
// ============================================================================

#[test]
fn test_model_registry_new_empty() {
    let registry = ModelRegistry::new();
    assert!(registry.providers().is_empty());
}

#[test]
fn test_model_registry_with_predefined() {
    let registry = ModelRegistry::with_predefined();
    let providers = registry.providers();

    // Should have at least OpenAI, Anthropic, Google
    assert!(providers.len() >= 3);
    assert!(providers.contains(&"openai".to_string()));
    assert!(providers.contains(&"anthropic".to_string()));
    assert!(providers.contains(&"google".to_string()));
}

#[test]
fn test_model_registry_register_and_get() {
    let mut registry = ModelRegistry::new();

    let model = Model::builder()
        .id("test-model")
        .name("Test Model")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url("http://test")
        .context_window(4096)
        .max_tokens(1024)
        .build()
        .unwrap();

    registry.register(model.clone());

    let retrieved = registry.get(&Provider::OpenAI, "test-model");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id, "test-model");
}

#[test]
fn test_model_registry_get_nonexistent() {
    let registry = ModelRegistry::new();
    assert!(registry.get(&Provider::OpenAI, "nonexistent").is_none());
}

#[test]
fn test_model_registry_get_wrong_provider() {
    let mut registry = ModelRegistry::new();
    let model = Model::builder()
        .id("test-model")
        .name("Test")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url("http://test")
        .context_window(4096)
        .max_tokens(1024)
        .build()
        .unwrap();
    registry.register(model);

    assert!(registry.get(&Provider::Anthropic, "test-model").is_none());
}

#[test]
fn test_model_registry_models_for_provider() {
    let registry = ModelRegistry::with_predefined();

    let openai_models = registry.models_for_provider(&Provider::OpenAI);
    assert!(openai_models.len() >= 4); // gpt-4o-mini, gpt-4o, gpt-4.1, o3

    let anthropic_models = registry.models_for_provider(&Provider::Anthropic);
    assert!(anthropic_models.len() >= 2); // claude-sonnet-4, claude-opus-4

    let google_models = registry.models_for_provider(&Provider::Google);
    assert!(google_models.len() >= 1); // gemini-2.5-flash
}

#[test]
fn test_model_registry_models_for_unknown_provider() {
    let registry = ModelRegistry::with_predefined();
    let models = registry.models_for_provider(&Provider::Custom("unknown".into()));
    assert!(models.is_empty());
}

#[test]
fn test_model_registry_default_is_predefined() {
    let registry = ModelRegistry::default();
    assert!(!registry.providers().is_empty());
}

// ============================================================================
// Predefined model validation
// ============================================================================

#[test]
fn test_predefined_openai_models() {
    let registry = ModelRegistry::with_predefined();

    // gpt-4o-mini
    let model = registry.get(&Provider::OpenAI, "gpt-4o-mini").unwrap();
    assert_eq!(model.api, Api::OpenAICompletions);
    assert!(!model.reasoning);
    assert!(model.supports_text());
    assert!(model.supports_image());
    assert_eq!(model.context_window, 128000);

    // gpt-4o
    let model = registry.get(&Provider::OpenAI, "gpt-4o").unwrap();
    assert!(!model.reasoning);

    // gpt-4.1
    let model = registry.get(&Provider::OpenAI, "gpt-4.1").unwrap();
    assert_eq!(model.context_window, 1047576);

    // o3
    let model = registry.get(&Provider::OpenAI, "o3").unwrap();
    assert!(model.reasoning);
}

#[test]
fn test_predefined_anthropic_models() {
    let registry = ModelRegistry::with_predefined();

    let sonnet = registry.get(&Provider::Anthropic, "claude-sonnet-4-20250514").unwrap();
    assert_eq!(sonnet.api, Api::AnthropicMessages);
    assert!(sonnet.reasoning);
    assert_eq!(sonnet.cost.input, 3.0);
    assert_eq!(sonnet.cost.output, 15.0);

    let opus = registry.get(&Provider::Anthropic, "claude-opus-4-20250514").unwrap();
    assert_eq!(opus.cost.input, 15.0);
    assert_eq!(opus.cost.output, 75.0);
}

#[test]
fn test_predefined_google_models() {
    let registry = ModelRegistry::with_predefined();

    let gemini = registry.get(&Provider::Google, "gemini-2.5-flash").unwrap();
    assert_eq!(gemini.api, Api::GoogleGenerativeAi);
    assert!(gemini.reasoning);
    assert_eq!(gemini.context_window, 1048576);
}

// ============================================================================
// Global model functions
// ============================================================================

#[test]
fn test_get_model_function() {
    let model = tiy_core::models::get_model("openai", "gpt-4o-mini");
    assert!(model.is_some());
    assert_eq!(model.unwrap().id, "gpt-4o-mini");
}

#[test]
fn test_get_model_nonexistent() {
    let model = tiy_core::models::get_model("openai", "nonexistent");
    assert!(model.is_none());
}

#[test]
fn test_get_providers_function() {
    let providers = tiy_core::models::get_providers();
    assert!(providers.len() >= 3);
}

// ============================================================================
// ProviderRegistry tests
// ============================================================================

use std::sync::Arc;
use tiy_core::provider::{ProviderRegistry, LLMProvider};
use tiy_core::stream::AssistantMessageEventStream;

struct MockProvider;

impl LLMProvider for MockProvider {
    fn api_type(&self) -> Api {
        Api::Custom("mock".to_string())
    }

    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: StreamOptions,
    ) -> AssistantMessageEventStream {
        AssistantMessageEventStream::new_assistant_stream()
    }

    fn stream_simple(
        &self,
        _model: &Model,
        _context: &Context,
        _options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        AssistantMessageEventStream::new_assistant_stream()
    }
}

#[test]
fn test_provider_registry_new() {
    let registry = ProviderRegistry::new();
    assert!(registry.api_types().is_empty());
}

#[test]
fn test_provider_registry_register_and_get() {
    let registry = ProviderRegistry::new();
    let provider: Arc<dyn LLMProvider> = Arc::new(MockProvider);
    registry.register(provider);

    let api = Api::Custom("mock".to_string());
    assert!(registry.contains(&api));

    let retrieved = registry.get(&api);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().api_type(), api);
}

#[test]
fn test_provider_registry_get_by_name() {
    let registry = ProviderRegistry::new();
    let provider: Arc<dyn LLMProvider> = Arc::new(MockProvider);
    registry.register(provider);

    let retrieved = registry.get_by_name("mock");
    assert!(retrieved.is_some());
}

#[test]
fn test_provider_registry_unregister() {
    let registry = ProviderRegistry::new();
    let provider: Arc<dyn LLMProvider> = Arc::new(MockProvider);
    registry.register(provider);

    let api = Api::Custom("mock".to_string());
    assert!(registry.contains(&api));

    registry.unregister(&api);
    assert!(!registry.contains(&api));
}

#[test]
fn test_provider_registry_clear() {
    let registry = ProviderRegistry::new();
    let provider: Arc<dyn LLMProvider> = Arc::new(MockProvider);
    registry.register(provider);

    registry.clear();
    assert!(registry.api_types().is_empty());
}

#[test]
fn test_provider_registry_api_types() {
    let registry = ProviderRegistry::new();
    let provider: Arc<dyn LLMProvider> = Arc::new(MockProvider);
    registry.register(provider);

    let types = registry.api_types();
    assert_eq!(types.len(), 1);
    assert!(types.contains(&"mock".to_string()));
}

#[test]
fn test_provider_registry_not_found() {
    let registry = ProviderRegistry::new();
    assert!(registry.get(&Api::OpenAICompletions).is_none());
    assert!(!registry.contains(&Api::OpenAICompletions));
}

//! Provider registry for managing LLM API providers.

use crate::provider::{ArcProvider, LLMProvider};
use crate::types::Api;
use parking_lot::RwLock;
use std::collections::HashMap;

/// Provider registry for managing LLM API providers.
pub struct ProviderRegistry {
    providers: RwLock<HashMap<String, ArcProvider>>,
}

impl ProviderRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
        }
    }

    /// Register a provider.
    pub fn register(&self, provider: ArcProvider) {
        let api = provider.api_type();
        let mut providers = self.providers.write();
        providers.insert(api.as_str().to_string(), provider);
    }

    /// Get a provider by API type.
    pub fn get(&self, api: &Api) -> Option<ArcProvider> {
        let providers = self.providers.read();
        providers.get(api.as_str()).cloned()
    }

    /// Get a provider by API type string.
    pub fn get_by_name(&self, api_name: &str) -> Option<ArcProvider> {
        let providers = self.providers.read();
        providers.get(api_name).cloned()
    }

    /// Unregister a provider by API type.
    pub fn unregister(&self, api: &Api) {
        let mut providers = self.providers.write();
        providers.remove(api.as_str());
    }

    /// Clear all providers.
    pub fn clear(&self) {
        let mut providers = self.providers.write();
        providers.clear();
    }

    /// Get all registered API types.
    pub fn api_types(&self) -> Vec<String> {
        let providers = self.providers.read();
        providers.keys().cloned().collect()
    }

    /// Check if a provider is registered.
    pub fn contains(&self, api: &Api) -> bool {
        let providers = self.providers.read();
        providers.contains_key(api.as_str())
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global provider registry.
static GLOBAL_REGISTRY: once_cell::sync::Lazy<ProviderRegistry> =
    once_cell::sync::Lazy::new(ProviderRegistry::new);

/// Get the global provider registry.
pub fn global_registry() -> &'static ProviderRegistry {
    &GLOBAL_REGISTRY
}

/// Register a provider globally.
pub fn register_provider(provider: ArcProvider) {
    GLOBAL_REGISTRY.register(provider);
}

/// Get a provider from the global registry.
pub fn get_provider(api: &Api) -> Option<ArcProvider> {
    GLOBAL_REGISTRY.get(api)
}

/// Clear all providers from the global registry.
pub fn clear_providers() {
    GLOBAL_REGISTRY.clear();
}

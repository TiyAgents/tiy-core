//! Provider registry for managing LLM providers.

use crate::provider::ArcProvider;
use crate::types::Provider;
use parking_lot::RwLock;
use std::collections::HashMap;

/// Provider registry for managing LLM providers.
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
        let provider_type = provider.provider_type();
        let mut providers = self.providers.write();
        providers.insert(provider_type.as_str().to_string(), provider);
    }

    /// Get a provider by provider type.
    pub fn get(&self, provider: &Provider) -> Option<ArcProvider> {
        let providers = self.providers.read();
        providers.get(provider.as_str()).cloned()
    }

    /// Get a provider by provider type string.
    pub fn get_by_name(&self, provider_name: &str) -> Option<ArcProvider> {
        let providers = self.providers.read();
        providers.get(provider_name).cloned()
    }

    /// Unregister a provider by provider type.
    pub fn unregister(&self, provider: &Provider) {
        let mut providers = self.providers.write();
        providers.remove(provider.as_str());
    }

    /// Clear all providers.
    pub fn clear(&self) {
        let mut providers = self.providers.write();
        providers.clear();
    }

    /// Get all registered provider types.
    pub fn provider_types(&self) -> Vec<String> {
        let providers = self.providers.read();
        providers.keys().cloned().collect()
    }

    /// Check if a provider is registered.
    pub fn contains(&self, provider: &Provider) -> bool {
        let providers = self.providers.read();
        providers.contains_key(provider.as_str())
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
pub fn get_provider(provider: &Provider) -> Option<ArcProvider> {
    GLOBAL_REGISTRY.get(provider)
}

/// Get all registered provider type names from the global registry.
pub fn get_registered_providers() -> Vec<String> {
    GLOBAL_REGISTRY.provider_types()
}

/// Clear all providers from the global registry.
pub fn clear_providers() {
    GLOBAL_REGISTRY.clear();
}

//! Macros for reducing boilerplate in delegation providers.
//!
//! Delegation providers share a common pattern: they hold an optional API key,
//! resolve it from options / self / env, optionally inject compat settings,
//! and delegate to a protocol provider (OpenAI Completions or Anthropic).

/// Generate a delegation provider that delegates to `OpenAICompletionsProvider`.
///
/// # Variants
///
/// **Without compat** (e.g., OpenRouter):
/// ```ignore
/// define_openai_delegation_provider! {
///     name: OpenRouterProvider,
///     doc: "OpenRouter provider (OpenAI-compatible with routing extensions).",
///     provider_type: Provider::OpenRouter,
///     env_var: "OPENROUTER_API_KEY",
/// }
/// ```
///
/// **With static compat** (e.g., xAI, ZAI):
/// ```ignore
/// define_openai_delegation_provider! {
///     name: XAIProvider,
///     doc: "xAI provider (OpenAI-compatible, Grok models).",
///     provider_type: Provider::XAI,
///     env_var: "XAI_API_KEY",
///     default_compat: || OpenAICompletionsCompat { ... },
/// }
/// ```
///
/// **With model-aware compat** (e.g., Groq):
/// ```ignore
/// define_openai_delegation_provider! {
///     name: GroqProvider,
///     doc: "Groq provider (OpenAI-compatible).",
///     provider_type: Provider::Groq,
///     env_var: "GROQ_API_KEY",
///     model_aware_compat: |model_id: &str| OpenAICompletionsCompat { ... },
/// }
/// ```
macro_rules! define_openai_delegation_provider {
    // Variant 1: No compat injection
    (
        name: $name:ident,
        doc: $doc:literal,
        provider_type: $provider_type:expr,
        env_var: $env_var:literal $(,)?
    ) => {
        #[doc = $doc]
        pub struct $name {
            default_api_key: Option<String>,
        }

        impl $name {
            /// Create a new provider.
            pub fn new() -> Self {
                Self { default_api_key: None }
            }

            /// Create a provider with a default API key.
            pub fn with_api_key(api_key: impl Into<String>) -> Self {
                Self { default_api_key: Some(api_key.into()) }
            }

            /// Resolve API key from options, self, or environment.
            fn resolve_api_key(&self, options: &StreamOptions) -> Option<String> {
                if let Some(ref key) = options.api_key {
                    return Some(key.clone());
                }
                if let Some(ref key) = self.default_api_key {
                    return Some(key.clone());
                }
                std::env::var($env_var).ok()
            }
        }

        impl Default for $name {
            fn default() -> Self { Self::new() }
        }

        #[async_trait::async_trait]
        impl crate::provider::LLMProvider for $name {
            fn provider_type(&self) -> Provider { $provider_type }

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
                let provider = super::openai_completions::OpenAICompletionsProvider::new();
                provider.stream(model, context, opts)
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
                let provider = super::openai_completions::OpenAICompletionsProvider::new();
                provider.stream_simple(model, context, opts)
            }
        }
    };

    // Variant 2: Static compat (compat = fn() -> OpenAICompletionsCompat)
    (
        name: $name:ident,
        doc: $doc:literal,
        provider_type: $provider_type:expr,
        env_var: $env_var:literal,
        default_compat: $compat_fn:expr $(,)?
    ) => {
        #[doc = $doc]
        pub struct $name {
            default_api_key: Option<String>,
        }

        impl $name {
            /// Create a new provider.
            pub fn new() -> Self {
                Self { default_api_key: None }
            }

            /// Create a provider with a default API key.
            pub fn with_api_key(api_key: impl Into<String>) -> Self {
                Self { default_api_key: Some(api_key.into()) }
            }

            /// Resolve API key from options, self, or environment.
            fn resolve_api_key(&self, options: &StreamOptions) -> Option<String> {
                if let Some(ref key) = options.api_key {
                    return Some(key.clone());
                }
                if let Some(ref key) = self.default_api_key {
                    return Some(key.clone());
                }
                std::env::var($env_var).ok()
            }

            /// Get provider-specific compat settings.
            pub fn default_compat() -> OpenAICompletionsCompat {
                ($compat_fn)()
            }
        }

        impl Default for $name {
            fn default() -> Self { Self::new() }
        }

        #[async_trait::async_trait]
        impl crate::provider::LLMProvider for $name {
            fn provider_type(&self) -> Provider { $provider_type }

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
                let model = if model.compat.is_none() {
                    let mut m = model.clone();
                    m.compat = Some(Self::default_compat());
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
                    m.compat = Some(Self::default_compat());
                    m
                } else {
                    model.clone()
                };
                let provider = super::openai_completions::OpenAICompletionsProvider::new();
                provider.stream_simple(&model, context, opts)
            }
        }
    };

    // Variant 3: Model-aware compat (compat = fn(&str) -> OpenAICompletionsCompat)
    (
        name: $name:ident,
        doc: $doc:literal,
        provider_type: $provider_type:expr,
        env_var: $env_var:literal,
        model_aware_compat: $compat_fn:expr $(,)?
    ) => {
        #[doc = $doc]
        pub struct $name {
            default_api_key: Option<String>,
        }

        impl $name {
            /// Create a new provider.
            pub fn new() -> Self {
                Self { default_api_key: None }
            }

            /// Create a provider with a default API key.
            pub fn with_api_key(api_key: impl Into<String>) -> Self {
                Self { default_api_key: Some(api_key.into()) }
            }

            /// Resolve API key from options, self, or environment.
            fn resolve_api_key(&self, options: &StreamOptions) -> Option<String> {
                if let Some(ref key) = options.api_key {
                    return Some(key.clone());
                }
                if let Some(ref key) = self.default_api_key {
                    return Some(key.clone());
                }
                std::env::var($env_var).ok()
            }

            /// Get provider-specific compat settings based on model ID.
            pub fn default_compat(model_id: &str) -> OpenAICompletionsCompat {
                ($compat_fn)(model_id)
            }
        }

        impl Default for $name {
            fn default() -> Self { Self::new() }
        }

        #[async_trait::async_trait]
        impl crate::provider::LLMProvider for $name {
            fn provider_type(&self) -> Provider { $provider_type }

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
    };
}

/// Generate a delegation provider that delegates to `AnthropicProvider`.
macro_rules! define_anthropic_delegation_provider {
    (
        name: $name:ident,
        doc: $doc:literal,
        provider_type: $provider_type:expr,
        env_var: $env_var:literal $(,)?
    ) => {
        #[doc = $doc]
        pub struct $name {
            default_api_key: Option<String>,
        }

        impl $name {
            /// Create a new provider.
            pub fn new() -> Self {
                Self { default_api_key: None }
            }

            /// Create a provider with a default API key.
            pub fn with_api_key(api_key: impl Into<String>) -> Self {
                Self { default_api_key: Some(api_key.into()) }
            }

            /// Resolve API key from options, self, or environment.
            fn resolve_api_key(&self, options: &StreamOptions) -> Option<String> {
                if let Some(ref key) = options.api_key {
                    return Some(key.clone());
                }
                if let Some(ref key) = self.default_api_key {
                    return Some(key.clone());
                }
                std::env::var($env_var).ok()
            }
        }

        impl Default for $name {
            fn default() -> Self { Self::new() }
        }

        #[async_trait::async_trait]
        impl crate::provider::LLMProvider for $name {
            fn provider_type(&self) -> Provider { $provider_type }

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
                let provider = super::anthropic::AnthropicProvider::new();
                provider.stream(model, context, opts)
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
                let provider = super::anthropic::AnthropicProvider::new();
                provider.stream_simple(model, context, opts)
            }
        }
    };
}

// Macros exported via #[macro_use] on the module declaration in mod.rs

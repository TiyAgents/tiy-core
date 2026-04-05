//! # tiycore
//!
//! A unified LLM API and stateful Agent runtime in Rust.
//!
//! This library provides:
//! - Unified API for multiple LLM providers (OpenAI, Anthropic, Google, Ollama, etc.)
//! - Streaming event-based responses
//! - Tool/function calling support
//! - Stateful agent with tool execution
//!
//! ## Example
//!
//! ```rust,no_run
//! use tiycore::{provider::openai::OpenAIProvider, types::*};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a model
//!     let model = Model {
//!         id: "gpt-4o-mini".to_string(),
//!         name: "GPT-4o Mini".to_string(),
//!         api: None,
//!         provider: Provider::OpenAI,
//!         base_url: Some("https://api.openai.com/v1".to_string()),
//!         reasoning: false,
//!         input: vec![InputType::Text, InputType::Image],
//!         cost: Cost::default(),
//!         context_window: 128000,
//!         max_tokens: 16384,
//!         headers: None,
//!         compat: None,
//!     };
//!
//!     // Create context
//!     let context = Context {
//!         system_prompt: Some("You are a helpful assistant.".to_string()),
//!         messages: vec![Message::User(UserMessage {
//!             role: Role::User,
//!             content: UserContent::Text("Hello!".to_string()),
//!             timestamp: chrono::Utc::now().timestamp_millis(),
//!         })],
//!         tools: None,
//!     };
//!
//!     Ok(())
//! }
//! ```

pub mod agent;
pub mod catalog;
pub mod models;
pub mod protocol;
pub mod provider;
pub mod stream;
pub mod thinking;
pub mod transform;
pub mod types;
pub mod validation;

// Re-export commonly used types
pub use types::{
    AgentLimits, Api, AssistantMessage, CacheRetention, ContentBlock, Context, Cost, HeaderPolicy,
    HttpLimits, ImageContent, InputType, Message, Model, OpenAIServiceTier, Provider, Role,
    SecurityConfig, StopReason, StreamLimits, TextContent, ThinkingContent, Tool, ToolCall,
    ToolChoice, ToolChoiceFunction, ToolChoiceMode, ToolChoiceNamed, ToolResultMessage, UrlPolicy,
    Usage, UserMessage,
};

pub use stream::EventStream;

pub use agent::{Agent, AgentStateSnapshot};
pub use catalog::{
    build_catalog_snapshot, build_catalog_snapshot_manifest, catalog_manifest_sidecar_path,
    enrich_manual_model, list_models, list_models_with_enrichment, load_catalog_metadata_store,
    refresh_catalog_snapshot, save_catalog_snapshot, CatalogMetadataStore, CatalogModelMatch,
    CatalogModelMetadata, CatalogRefreshResult, CatalogRemoteConfig, CatalogSnapshot,
    CatalogSnapshotError, CatalogSnapshotManifest, EmptyCatalogMetadataStore, FetchModelsRequest,
    FileCatalogMetadataStore, InMemoryCatalogMetadataStore, ListModelsResult, ModelCatalogError,
    ProviderExtractedModel, UnifiedModelInfo,
};

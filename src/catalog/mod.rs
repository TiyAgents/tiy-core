//! Model catalog fetching, normalization, and metadata enrichment.
//!
//! This module provides a display-oriented model listing flow:
//! 1. Fetch a provider's native model list
//! 2. Extract a shared intermediate model shape from heterogeneous payloads
//! 3. Enrich from an external catalog metadata store
//! 4. Return unified model information while preserving the provider raw ID

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::protocol::common::apply_custom_headers;
use crate::types::{HeaderPolicy, Provider};
use sha2::{Digest, Sha256};
use url::Url;

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const XAI_BASE_URL: &str = "https://api.x.ai/v1";
const GROQ_BASE_URL: &str = "https://api.groq.com/openai/v1";
const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
const ZAI_BASE_URL: &str = "https://api.z.ai/api/coding/paas/v4";
const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";
const ZENMUX_BASE_URL: &str = "https://zenmux.ai/api/v1";
const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";
const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";
const MINIMAX_BASE_URL: &str = "https://api.minimax.io/anthropic";
const MINIMAX_CN_BASE_URL: &str = "https://api.minimaxi.com/anthropic";
const KIMI_CODING_BASE_URL: &str = "https://api.kimi.com/coding";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_CATALOG_MANIFEST_URL: &str =
    "https://tiyagents.github.io/tiy-core/catalog/manifest.json";

/// Request to fetch models from a provider.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FetchModelsRequest {
    /// Provider to query.
    pub provider: Provider,
    /// API key override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Base URL override. Should point at the provider API base, such as `/v1`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Custom headers to add to the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

impl FetchModelsRequest {
    /// Create a request for the given provider.
    pub fn new(provider: Provider) -> Self {
        Self {
            provider,
            api_key: None,
            base_url: None,
            headers: None,
        }
    }
}

/// Provider-native extracted model fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProviderExtractedModel {
    pub provider: Provider,
    pub raw_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
    pub raw: Value,
}

/// External metadata used to enrich native provider models.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CatalogModelMetadata {
    pub canonical_model_key: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing: Option<Value>,
    pub source: String,
    pub raw: Value,
}

/// Metadata match result from a store.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CatalogModelMatch {
    pub metadata: CatalogModelMetadata,
    pub confidence: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_alias: Option<String>,
}

/// Snapshot manifest published to a remote catalog endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CatalogSnapshotManifest {
    pub version: String,
    pub generated_at: String,
    pub snapshot_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

/// Catalog snapshot containing normalized model metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CatalogSnapshot {
    pub version: String,
    pub generated_at: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<CatalogModelMetadata>,
}

/// Remote configuration for refreshing a snapshot from a published catalog.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CatalogRemoteConfig {
    pub manifest_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

impl Default for CatalogRemoteConfig {
    fn default() -> Self {
        Self {
            manifest_url: DEFAULT_CATALOG_MANIFEST_URL.to_string(),
            headers: None,
        }
    }
}

impl CatalogRemoteConfig {
    pub fn new(manifest_url: impl Into<String>) -> Self {
        Self {
            manifest_url: manifest_url.into(),
            headers: None,
        }
    }
}

/// Result of refreshing a local catalog snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CatalogRefreshResult {
    Updated {
        manifest: CatalogSnapshotManifest,
        bytes_written: u64,
        created: bool,
    },
    Unchanged {
        manifest: CatalogSnapshotManifest,
    },
}

/// Unified model data returned to applications.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UnifiedModelInfo {
    pub provider: Provider,
    pub raw_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_model_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub match_confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metadata_sources: Vec<String>,
    pub raw: Value,
}

/// Result of listing models for a provider.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ListModelsResult {
    pub models: Vec<UnifiedModelInfo>,
    pub raw_response: Value,
}

/// Error returned by model catalog operations.
#[derive(Debug, thiserror::Error)]
pub enum ModelCatalogError {
    #[error("provider {provider} does not expose a supported list-models adapter yet")]
    UnsupportedProvider { provider: Provider },
    #[error("provider {provider} requires a base_url override for list-models requests")]
    MissingBaseUrl { provider: Provider },
    #[error("provider {provider} returned an invalid models payload: {message}")]
    InvalidResponse { provider: Provider, message: String },
    #[error("provider {provider} returned a repeating pagination cursor `{cursor}` while listing models")]
    PaginationLoop { provider: Provider, cursor: String },
    #[error("provider {provider} returned HTTP {status}: {body}")]
    Http {
        provider: Provider,
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("request to provider {provider} failed: {source}")]
    Request {
        provider: Provider,
        #[source]
        source: reqwest::Error,
    },
}

/// Error returned by snapshot load, save, or refresh operations.
#[derive(Debug, thiserror::Error)]
pub enum CatalogSnapshotError {
    #[error("failed to read snapshot file {path}: {source}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write snapshot file {path}: {source}")]
    WriteFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse snapshot file {path}: {source}")]
    ParseSnapshot {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to serialize snapshot data: {source}")]
    SerializeSnapshot {
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to fetch catalog manifest from {url}: {source}")]
    FetchManifest {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("catalog manifest request to {url} returned HTTP {status}: {body}")]
    FetchManifestHttp {
        url: String,
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("failed to parse catalog manifest from {url}: {source}")]
    ParseManifest {
        url: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to fetch catalog snapshot from {url}: {source}")]
    FetchSnapshot {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("catalog snapshot request to {url} returned HTTP {status}: {body}")]
    FetchSnapshotHttp {
        url: String,
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("manifest URL is invalid: {url}")]
    InvalidManifestUrl { url: String },
    #[error("snapshot URL is invalid: {url}")]
    InvalidSnapshotUrl { url: String },
    #[error("snapshot checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },
    #[error("snapshot size mismatch: expected {expected} bytes, got {actual} bytes")]
    SizeMismatch { expected: u64, actual: u64 },
    #[error("snapshot version mismatch: manifest has {manifest_version}, snapshot has {snapshot_version}")]
    VersionMismatch {
        manifest_version: String,
        snapshot_version: String,
    },
}

/// Read-only source of catalog metadata.
pub trait CatalogMetadataStore: Send + Sync {
    fn find_by_raw_or_alias(
        &self,
        provider: &Provider,
        raw_id: &str,
        normalized_aliases: &[String],
    ) -> Option<CatalogModelMatch>;
}

/// Metadata store that never returns enrichment data.
#[derive(Debug, Default)]
pub struct EmptyCatalogMetadataStore;

impl CatalogMetadataStore for EmptyCatalogMetadataStore {
    fn find_by_raw_or_alias(
        &self,
        _provider: &Provider,
        _raw_id: &str,
        _normalized_aliases: &[String],
    ) -> Option<CatalogModelMatch> {
        None
    }
}

/// Simple in-memory metadata store for tests or embedded snapshots.
#[derive(Debug, Clone, Default)]
pub struct InMemoryCatalogMetadataStore {
    entries: Vec<CatalogModelMetadata>,
    alias_index: HashMap<String, usize>,
}

impl InMemoryCatalogMetadataStore {
    pub fn new(entries: Vec<CatalogModelMetadata>) -> Self {
        let mut alias_index = HashMap::new();

        for (idx, entry) in entries.iter().enumerate() {
            for alias in metadata_aliases(entry) {
                alias_index.entry(alias).or_insert(idx);
            }
        }

        Self {
            entries,
            alias_index,
        }
    }
}

impl CatalogMetadataStore for InMemoryCatalogMetadataStore {
    fn find_by_raw_or_alias(
        &self,
        _provider: &Provider,
        raw_id: &str,
        normalized_aliases: &[String],
    ) -> Option<CatalogModelMatch> {
        let mut candidates = Vec::with_capacity(normalized_aliases.len() + 1);
        candidates.extend(normalized_aliases.iter().cloned());
        candidates.extend(normalized_alias_candidates(raw_id, None));

        for candidate in candidates {
            if let Some(idx) = self.alias_index.get(&candidate) {
                let metadata = self.entries[*idx].clone();
                return Some(CatalogModelMatch {
                    metadata,
                    confidence: 1.0,
                    matched_alias: Some(candidate),
                });
            }
        }

        None
    }
}

/// File-backed metadata store loaded from a local snapshot file.
#[derive(Debug, Clone)]
pub struct FileCatalogMetadataStore {
    snapshot: CatalogSnapshot,
    inner: InMemoryCatalogMetadataStore,
}

impl FileCatalogMetadataStore {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CatalogSnapshotError> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|source| CatalogSnapshotError::ReadFile {
            path: path.to_path_buf(),
            source,
        })?;
        let snapshot = serde_json::from_slice::<CatalogSnapshot>(&bytes).map_err(|source| {
            CatalogSnapshotError::ParseSnapshot {
                path: path.to_path_buf(),
                source,
            }
        })?;
        Ok(Self::from_snapshot(snapshot))
    }

    pub fn try_load(path: impl AsRef<Path>) -> Result<Option<Self>, CatalogSnapshotError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(None);
        }
        Self::load(path).map(Some)
    }

    pub fn from_snapshot(snapshot: CatalogSnapshot) -> Self {
        let inner = InMemoryCatalogMetadataStore::new(snapshot.models.clone());
        Self { snapshot, inner }
    }

    pub fn snapshot(&self) -> &CatalogSnapshot {
        &self.snapshot
    }
}

impl CatalogMetadataStore for FileCatalogMetadataStore {
    fn find_by_raw_or_alias(
        &self,
        provider: &Provider,
        raw_id: &str,
        normalized_aliases: &[String],
    ) -> Option<CatalogModelMatch> {
        self.inner
            .find_by_raw_or_alias(provider, raw_id, normalized_aliases)
    }
}

/// Fetch native models without metadata enrichment.
pub async fn list_models(
    request: FetchModelsRequest,
) -> Result<ListModelsResult, ModelCatalogError> {
    list_models_with_enrichment(request, &EmptyCatalogMetadataStore).await
}

/// Fetch native models and enrich them from an external metadata store.
pub async fn list_models_with_enrichment(
    request: FetchModelsRequest,
    metadata_store: &dyn CatalogMetadataStore,
) -> Result<ListModelsResult, ModelCatalogError> {
    let adapter = adapter_for(&request.provider)?;
    let raw_response = adapter.fetch_raw(&request).await?;
    let extracted = adapter.extract_models(&raw_response)?;
    let models = extracted
        .into_iter()
        .map(|model| enrich_model(model, metadata_store))
        .collect();

    Ok(ListModelsResult {
        models,
        raw_response,
    })
}

/// Enrich a manually provided model ID using the same metadata snapshot used for
/// fetched provider models.
///
/// This is useful when an application allows users to type a model ID directly,
/// or when the upstream provider does not expose a list-models endpoint.
///
/// The returned [`UnifiedModelInfo`] preserves the caller-supplied `raw_id`.
/// Metadata fields are filled from the provided [`CatalogMetadataStore`] when a
/// matching snapshot entry is found.
pub fn enrich_manual_model(
    provider: Provider,
    raw_id: impl Into<String>,
    display_name: Option<String>,
    metadata_store: &dyn CatalogMetadataStore,
) -> UnifiedModelInfo {
    let raw_id = raw_id.into();
    enrich_model(
        ProviderExtractedModel {
            provider,
            raw_id,
            display_name,
            description: None,
            context_window: None,
            max_output_tokens: None,
            max_input_tokens: None,
            created_at: None,
            modalities: None,
            capabilities: None,
            raw: json!({}),
        },
        metadata_store,
    )
}

/// Load a local snapshot path into a file-backed metadata store, if it exists.
pub fn load_catalog_metadata_store(
    snapshot_path: impl AsRef<Path>,
) -> Result<Option<FileCatalogMetadataStore>, CatalogSnapshotError> {
    FileCatalogMetadataStore::try_load(snapshot_path)
}

/// Refresh a local snapshot file from a remote manifest and snapshot endpoint.
///
/// Applications can call this in the background while continuing to use an
/// already loaded local snapshot.
pub async fn refresh_catalog_snapshot(
    snapshot_path: impl AsRef<Path>,
    config: &CatalogRemoteConfig,
) -> Result<CatalogRefreshResult, CatalogSnapshotError> {
    let snapshot_path = snapshot_path.as_ref();
    let local_manifest_path = catalog_manifest_sidecar_path(snapshot_path);
    let client = build_client();

    let remote_manifest = fetch_remote_manifest(&client, config).await?;
    let local_manifest = read_local_manifest(&local_manifest_path)?;

    if snapshot_path.exists() {
        if let Some(local_manifest) = local_manifest.as_ref() {
            let same_version = local_manifest.version == remote_manifest.version;
            let same_checksum = local_manifest.sha256 == remote_manifest.sha256;
            if same_version && same_checksum {
                return Ok(CatalogRefreshResult::Unchanged {
                    manifest: remote_manifest,
                });
            }
        }
    }

    let snapshot_url = resolve_snapshot_url(&config.manifest_url, &remote_manifest.snapshot_url)?;
    let snapshot_bytes = fetch_remote_snapshot(&client, &snapshot_url, config).await?;

    if let Some(expected_size) = remote_manifest.size_bytes {
        let actual_size = snapshot_bytes.len() as u64;
        if actual_size != expected_size {
            return Err(CatalogSnapshotError::SizeMismatch {
                expected: expected_size,
                actual: actual_size,
            });
        }
    }

    if let Some(expected_sha) = remote_manifest.sha256.as_deref() {
        let actual_sha = sha256_hex(&snapshot_bytes);
        if actual_sha != expected_sha {
            return Err(CatalogSnapshotError::ChecksumMismatch {
                expected: expected_sha.to_string(),
                actual: actual_sha,
            });
        }
    }

    let snapshot: CatalogSnapshot = serde_json::from_slice(&snapshot_bytes).map_err(|source| {
        CatalogSnapshotError::ParseSnapshot {
            path: snapshot_path.to_path_buf(),
            source,
        }
    })?;

    if snapshot.version != remote_manifest.version {
        return Err(CatalogSnapshotError::VersionMismatch {
            manifest_version: remote_manifest.version.clone(),
            snapshot_version: snapshot.version,
        });
    }

    let manifest_bytes = serde_json::to_vec_pretty(&remote_manifest)
        .map_err(|source| CatalogSnapshotError::SerializeSnapshot { source })?;

    let created = !snapshot_path.exists();
    atomic_write(snapshot_path, &snapshot_bytes)?;
    atomic_write(&local_manifest_path, &manifest_bytes)?;

    Ok(CatalogRefreshResult::Updated {
        manifest: remote_manifest,
        bytes_written: snapshot_bytes.len() as u64,
        created,
    })
}

/// Build a snapshot document from metadata records.
pub fn build_catalog_snapshot(
    version: impl Into<String>,
    generated_at: impl Into<String>,
    models: Vec<CatalogModelMetadata>,
) -> CatalogSnapshot {
    CatalogSnapshot {
        version: version.into(),
        generated_at: generated_at.into(),
        models,
    }
}

/// Build a manifest document for a snapshot payload.
pub fn build_catalog_snapshot_manifest(
    version: impl Into<String>,
    generated_at: impl Into<String>,
    snapshot_url: impl Into<String>,
    snapshot_bytes: &[u8],
) -> CatalogSnapshotManifest {
    CatalogSnapshotManifest {
        version: version.into(),
        generated_at: generated_at.into(),
        snapshot_url: snapshot_url.into(),
        sha256: Some(sha256_hex(snapshot_bytes)),
        size_bytes: Some(snapshot_bytes.len() as u64),
    }
}

/// Save a snapshot and its sidecar manifest to disk.
pub fn save_catalog_snapshot(
    snapshot_path: impl AsRef<Path>,
    snapshot: &CatalogSnapshot,
    manifest: &CatalogSnapshotManifest,
) -> Result<(), CatalogSnapshotError> {
    let snapshot_path = snapshot_path.as_ref();
    let snapshot_bytes = serde_json::to_vec_pretty(snapshot)
        .map_err(|source| CatalogSnapshotError::SerializeSnapshot { source })?;
    let manifest_bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|source| CatalogSnapshotError::SerializeSnapshot { source })?;

    atomic_write(snapshot_path, &snapshot_bytes)?;
    atomic_write(
        &catalog_manifest_sidecar_path(snapshot_path),
        &manifest_bytes,
    )?;
    Ok(())
}

fn enrich_model(
    model: ProviderExtractedModel,
    metadata_store: &dyn CatalogMetadataStore,
) -> UnifiedModelInfo {
    let alias_candidates =
        normalized_alias_candidates(&model.raw_id, model.display_name.as_deref());
    let metadata_match =
        metadata_store.find_by_raw_or_alias(&model.provider, &model.raw_id, &alias_candidates);

    let metadata = metadata_match.as_ref().map(|m| &m.metadata);

    UnifiedModelInfo {
        provider: model.provider,
        raw_id: model.raw_id,
        canonical_model_key: metadata.map(|m| m.canonical_model_key.clone()),
        display_name: prefer_option(
            model.display_name,
            metadata.and_then(|m| m.display_name.clone()),
        ),
        description: prefer_option(
            model.description,
            metadata.and_then(|m| m.description.clone()),
        ),
        context_window: prefer_option(
            model.context_window,
            metadata.and_then(|m| m.context_window),
        ),
        max_output_tokens: prefer_option(
            model.max_output_tokens,
            metadata.and_then(|m| m.max_output_tokens),
        ),
        max_input_tokens: prefer_option(
            model.max_input_tokens,
            metadata.and_then(|m| m.max_input_tokens),
        ),
        created_at: model.created_at,
        modalities: prefer_option(
            model.modalities,
            metadata.and_then(|m| m.modalities.clone()),
        ),
        capabilities: prefer_option(
            model.capabilities,
            metadata.and_then(|m| m.capabilities.clone()),
        ),
        pricing: metadata.and_then(|m| m.pricing.clone()),
        match_confidence: metadata_match.as_ref().map(|m| m.confidence),
        metadata_sources: metadata.map(|m| vec![m.source.clone()]).unwrap_or_default(),
        raw: model.raw,
    }
}

fn prefer_option<T>(primary: Option<T>, fallback: Option<T>) -> Option<T> {
    primary.or(fallback)
}

fn metadata_aliases(metadata: &CatalogModelMetadata) -> Vec<String> {
    let mut aliases = Vec::new();
    aliases.extend(normalized_alias_candidates(
        &metadata.canonical_model_key,
        metadata.display_name.as_deref(),
    ));
    for alias in &metadata.aliases {
        aliases.extend(normalized_alias_candidates(
            alias,
            metadata.display_name.as_deref(),
        ));
    }
    dedupe_strings(aliases)
}

fn dedupe_strings(values: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        if seen.insert(value.clone()) {
            out.push(value);
        }
    }
    out
}

fn normalized_alias_candidates(raw_id: &str, display_name: Option<&str>) -> Vec<String> {
    let mut values = Vec::new();
    let raw_variants = [raw_id.to_string(), strip_vendor_prefix(raw_id)];

    for variant in raw_variants {
        let base = normalize_token(&variant);
        if base.is_empty() {
            continue;
        }
        values.push(base.clone());
        let dotted = collapse_separators(base.replace('.', "-"));
        if !dotted.is_empty() {
            values.push(dotted);
        }
    }

    if let Some(name) = display_name {
        let normalized_name = normalize_token(name);
        if !normalized_name.is_empty() {
            values.push(normalized_name);
        }
    }

    dedupe_strings(values)
}

fn normalize_token(input: &str) -> String {
    let lowered = input.trim().to_lowercase();
    let mut out = String::with_capacity(lowered.len());
    let mut last_dash = false;

    for ch in lowered.chars() {
        let mapped = match ch {
            'a'..='z' | '0'..='9' | '.' => Some(ch),
            '/' | '_' | ' ' | ':' => Some('-'),
            '-' => Some('-'),
            _ => None,
        };

        if let Some(ch) = mapped {
            if ch == '-' {
                if last_dash {
                    continue;
                }
                last_dash = true;
            } else {
                last_dash = false;
            }
            out.push(ch);
        }
    }

    collapse_separators(out)
}

fn collapse_separators(mut value: String) -> String {
    while value.contains("--") {
        value = value.replace("--", "-");
    }
    value.trim_matches('-').to_string()
}

fn strip_vendor_prefix(value: &str) -> String {
    for prefix in [
        "anthropic/",
        "anthropic:",
        "openai/",
        "openai:",
        "google/",
        "google:",
        "groq/",
        "groq:",
        "xai/",
        "xai:",
        "deepseek/",
        "deepseek:",
        "openrouter/",
        "openrouter:",
        "zai/",
        "zai:",
        "zenmux/",
        "zenmux:",
        "minimax/",
        "minimax:",
        "kimi/",
        "kimi:",
    ] {
        if let Some(stripped) = value.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    value.to_string()
}

#[async_trait]
trait ModelListAdapter: Send + Sync {
    async fn fetch_raw(&self, request: &FetchModelsRequest) -> Result<Value, ModelCatalogError>;

    fn extract_models(&self, raw: &Value)
        -> Result<Vec<ProviderExtractedModel>, ModelCatalogError>;
}

fn adapter_for(provider: &Provider) -> Result<Box<dyn ModelListAdapter>, ModelCatalogError> {
    match provider {
        Provider::OpenAI
        | Provider::OpenAICompatible
        | Provider::XAI
        | Provider::Groq
        | Provider::ZAI
        | Provider::DeepSeek
        | Provider::Ollama => Ok(Box::new(OpenAIModelsAdapter::new(provider.clone()))),
        Provider::OpenRouter => Ok(Box::new(OpenRouterModelsAdapter::new(provider.clone()))),
        Provider::Zenmux => Ok(Box::new(ZenmuxModelsAdapter::new(provider.clone()))),
        Provider::Anthropic | Provider::MiniMax | Provider::MiniMaxCN | Provider::KimiCoding => {
            Ok(Box::new(AnthropicModelsAdapter::new(provider.clone())))
        }
        _ => Err(ModelCatalogError::UnsupportedProvider {
            provider: provider.clone(),
        }),
    }
}

#[derive(Debug, Clone)]
struct OpenAIModelsAdapter {
    provider: Provider,
    client: Client,
}

impl OpenAIModelsAdapter {
    fn new(provider: Provider) -> Self {
        Self {
            provider,
            client: build_client(),
        }
    }
}

#[async_trait]
impl ModelListAdapter for OpenAIModelsAdapter {
    async fn fetch_raw(&self, request: &FetchModelsRequest) -> Result<Value, ModelCatalogError> {
        let url = join_url(&resolve_base_url(request)?, "models");
        let headers = build_openai_headers(&self.provider, request);
        send_json_request(&self.client, self.provider.clone(), &url, headers).await
    }

    fn extract_models(
        &self,
        raw: &Value,
    ) -> Result<Vec<ProviderExtractedModel>, ModelCatalogError> {
        extract_openai_models(&self.provider, raw)
    }
}

#[derive(Debug, Clone)]
struct OpenRouterModelsAdapter {
    provider: Provider,
    client: Client,
}

impl OpenRouterModelsAdapter {
    fn new(provider: Provider) -> Self {
        Self {
            provider,
            client: build_client(),
        }
    }
}

#[async_trait]
impl ModelListAdapter for OpenRouterModelsAdapter {
    async fn fetch_raw(&self, request: &FetchModelsRequest) -> Result<Value, ModelCatalogError> {
        let base_url = resolve_base_url(request)?;
        let headers = build_openai_headers(&self.provider, request);

        let models_url = join_url(&base_url, "models");
        let models_response = send_json_request(
            &self.client,
            self.provider.clone(),
            &models_url,
            headers.clone(),
        )
        .await?;

        let embeddings_url = join_url(&base_url, "embeddings/models");
        let embeddings_response = send_optional_json_request(
            &self.client,
            self.provider.clone(),
            &embeddings_url,
            headers,
        )
        .await?;

        let mut combined = value_array(&self.provider, &models_response, "data")?.clone();
        if let Some(ref embeddings) = embeddings_response {
            append_unique_models(
                &mut combined,
                value_array(&self.provider, embeddings, "data")?,
            );
        }

        Ok(json!({
            "data": combined,
            "sources": {
                "models": models_response,
                "embeddings": embeddings_response,
            }
        }))
    }

    fn extract_models(
        &self,
        raw: &Value,
    ) -> Result<Vec<ProviderExtractedModel>, ModelCatalogError> {
        extract_openai_models(&self.provider, raw)
    }
}

#[derive(Debug, Clone)]
struct ZenmuxModelsAdapter {
    provider: Provider,
    client: Client,
}

impl ZenmuxModelsAdapter {
    fn new(provider: Provider) -> Self {
        Self {
            provider,
            client: build_client(),
        }
    }
}

#[async_trait]
impl ModelListAdapter for ZenmuxModelsAdapter {
    async fn fetch_raw(&self, request: &FetchModelsRequest) -> Result<Value, ModelCatalogError> {
        let base_url = resolve_base_url(request)?;
        let headers = build_openai_headers(&self.provider, request);

        let models_url = join_url(&base_url, "models");
        let models_response = send_json_request(
            &self.client,
            self.provider.clone(),
            &models_url,
            headers.clone(),
        )
        .await?;

        let vertex_models_url = derive_zenmux_vertex_models_url(&base_url)?;
        let vertex_models_response = send_optional_json_request(
            &self.client,
            self.provider.clone(),
            &vertex_models_url,
            headers,
        )
        .await?;

        let mut combined = value_array(&self.provider, &models_response, "data")?.clone();
        if let Some(ref vertex_models) = vertex_models_response {
            append_unique_models(
                &mut combined,
                value_array(&self.provider, vertex_models, "models")?,
            );
        }

        Ok(json!({
            "data": combined,
            "sources": {
                "models": models_response,
                "vertex_models": vertex_models_response,
            }
        }))
    }

    fn extract_models(
        &self,
        raw: &Value,
    ) -> Result<Vec<ProviderExtractedModel>, ModelCatalogError> {
        extract_openai_models(&self.provider, raw)
    }
}

#[derive(Debug, Clone)]
struct AnthropicModelsAdapter {
    provider: Provider,
    client: Client,
}

impl AnthropicModelsAdapter {
    fn new(provider: Provider) -> Self {
        Self {
            provider,
            client: build_client(),
        }
    }
}

#[async_trait]
impl ModelListAdapter for AnthropicModelsAdapter {
    async fn fetch_raw(&self, request: &FetchModelsRequest) -> Result<Value, ModelCatalogError> {
        let mut after_id: Option<String> = None;
        let mut combined_pages = Vec::new();
        let mut combined_data = Vec::new();
        let mut seen_cursors = HashSet::new();

        loop {
            let url = join_url(&resolve_base_url(request)?, "models");
            let headers = build_anthropic_headers(&self.provider, request);
            let mut query = vec![("limit", "1000".to_string())];
            if let Some(ref cursor) = after_id {
                query.push(("after_id", cursor.clone()));
            }

            let response = send_json_request_with_query(
                &self.client,
                self.provider.clone(),
                &url,
                headers,
                &query,
            )
            .await?;

            combined_data.extend(
                value_array(&self.provider, &response, "data")?
                    .iter()
                    .cloned(),
            );
            let has_more = response
                .get("has_more")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let next_after_id = response
                .get("last_id")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            combined_pages.push(response);

            if !has_more {
                break;
            }

            let cursor = next_after_id.ok_or_else(|| ModelCatalogError::InvalidResponse {
                provider: self.provider.clone(),
                message: "paginated response is missing `last_id`".to_string(),
            })?;

            if !seen_cursors.insert(cursor.clone()) {
                return Err(ModelCatalogError::PaginationLoop {
                    provider: self.provider.clone(),
                    cursor,
                });
            }

            after_id = Some(cursor);
        }

        Ok(json!({
            "data": combined_data,
            "pages": combined_pages,
        }))
    }

    fn extract_models(
        &self,
        raw: &Value,
    ) -> Result<Vec<ProviderExtractedModel>, ModelCatalogError> {
        extract_anthropic_models(&self.provider, raw)
    }
}

fn build_client() -> Client {
    Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| Client::new())
}

fn derive_zenmux_vertex_models_url(base_url: &str) -> Result<String, ModelCatalogError> {
    let mut url = Url::parse(base_url).map_err(|_| ModelCatalogError::MissingBaseUrl {
        provider: Provider::Zenmux,
    })?;
    url.set_path("/api/vertex-ai/v1beta/models");
    url.set_query(None);
    url.set_fragment(None);
    Ok(url.to_string())
}

fn append_unique_models(target: &mut Vec<Value>, incoming: &Vec<Value>) {
    let mut seen_ids: HashSet<String> = target
        .iter()
        .filter_map(model_identifier)
        .map(ToString::to_string)
        .collect();

    for item in incoming {
        let Some(id) = model_identifier(item) else {
            continue;
        };
        if seen_ids.insert(id.to_string()) {
            target.push(item.clone());
        }
    }
}

fn model_identifier(item: &Value) -> Option<&str> {
    item.get("id")
        .and_then(Value::as_str)
        .or_else(|| item.get("name").and_then(Value::as_str))
}

fn resolve_base_url(request: &FetchModelsRequest) -> Result<String, ModelCatalogError> {
    if let Some(base_url) = request.base_url.as_ref() {
        return Ok(base_url.clone());
    }

    let base_url = match request.provider {
        Provider::OpenAI => OPENAI_BASE_URL,
        Provider::OpenAICompatible => {
            return Err(ModelCatalogError::MissingBaseUrl {
                provider: request.provider.clone(),
            })
        }
        Provider::XAI => XAI_BASE_URL,
        Provider::Groq => GROQ_BASE_URL,
        Provider::OpenRouter => OPENROUTER_BASE_URL,
        Provider::ZAI => ZAI_BASE_URL,
        Provider::DeepSeek => DEEPSEEK_BASE_URL,
        Provider::Zenmux => ZENMUX_BASE_URL,
        Provider::Ollama => OLLAMA_BASE_URL,
        Provider::Anthropic => ANTHROPIC_BASE_URL,
        Provider::MiniMax => MINIMAX_BASE_URL,
        Provider::MiniMaxCN => MINIMAX_CN_BASE_URL,
        Provider::KimiCoding => KIMI_CODING_BASE_URL,
        _ => {
            return Err(ModelCatalogError::UnsupportedProvider {
                provider: request.provider.clone(),
            })
        }
    };

    Ok(base_url.to_string())
}

fn join_url(base_url: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

fn build_openai_headers(provider: &Provider, request: &FetchModelsRequest) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    if let Some(api_key) = resolve_api_key(provider, request).filter(|key| !key.is_empty()) {
        let bearer = format!("Bearer {}", api_key);
        if let Ok(value) = HeaderValue::from_str(&bearer) {
            headers.insert(AUTHORIZATION, value);
        }
    }

    apply_custom_headers(&mut headers, &request.headers, &HeaderPolicy::default());

    headers
}

fn build_anthropic_headers(provider: &Provider, request: &FetchModelsRequest) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        "anthropic-version",
        HeaderValue::from_static(ANTHROPIC_VERSION),
    );

    if let Some(api_key) = resolve_api_key(provider, request).filter(|key| !key.is_empty()) {
        if let Ok(value) = HeaderValue::from_str(&api_key) {
            headers.insert("x-api-key", value);
        }
    }

    apply_custom_headers(&mut headers, &request.headers, &HeaderPolicy::default());

    headers
}

fn resolve_api_key(provider: &Provider, request: &FetchModelsRequest) -> Option<String> {
    if let Some(api_key) = request.api_key.as_ref() {
        return Some(api_key.clone());
    }

    let env_var = match provider {
        Provider::OpenAI => Some("OPENAI_API_KEY"),
        Provider::OpenAICompatible => Some("OPENAI_API_KEY"),
        Provider::XAI => Some("XAI_API_KEY"),
        Provider::Groq => Some("GROQ_API_KEY"),
        Provider::OpenRouter => Some("OPENROUTER_API_KEY"),
        Provider::ZAI => Some("ZAI_API_KEY"),
        Provider::DeepSeek => Some("DEEPSEEK_API_KEY"),
        Provider::Zenmux => Some("ZENMUX_API_KEY"),
        Provider::Anthropic => Some("ANTHROPIC_API_KEY"),
        Provider::MiniMax => Some("MINIMAX_API_KEY"),
        Provider::MiniMaxCN => Some("MINIMAX_CN_API_KEY"),
        Provider::KimiCoding => Some("KIMI_API_KEY"),
        Provider::Ollama => None,
        _ => None,
    };

    env_var.and_then(|name| std::env::var(name).ok())
}

fn read_local_manifest(
    manifest_path: &Path,
) -> Result<Option<CatalogSnapshotManifest>, CatalogSnapshotError> {
    if !manifest_path.exists() {
        return Ok(None);
    }

    let bytes = fs::read(manifest_path).map_err(|source| CatalogSnapshotError::ReadFile {
        path: manifest_path.to_path_buf(),
        source,
    })?;
    let manifest = serde_json::from_slice::<CatalogSnapshotManifest>(&bytes).map_err(|source| {
        CatalogSnapshotError::ParseManifest {
            url: manifest_path.display().to_string(),
            source,
        }
    })?;
    Ok(Some(manifest))
}

async fn fetch_remote_manifest(
    client: &Client,
    config: &CatalogRemoteConfig,
) -> Result<CatalogSnapshotManifest, CatalogSnapshotError> {
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
    apply_custom_headers(&mut headers, &config.headers, &HeaderPolicy::default());

    let response = client
        .get(&config.manifest_url)
        .headers(headers)
        .send()
        .await
        .map_err(|source| CatalogSnapshotError::FetchManifest {
            url: config.manifest_url.clone(),
            source,
        })?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(CatalogSnapshotError::FetchManifestHttp {
            url: config.manifest_url.clone(),
            status,
            body,
        });
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|source| CatalogSnapshotError::FetchManifest {
            url: config.manifest_url.clone(),
            source,
        })?;

    serde_json::from_slice::<CatalogSnapshotManifest>(&bytes).map_err(|source| {
        CatalogSnapshotError::ParseManifest {
            url: config.manifest_url.clone(),
            source,
        }
    })
}

async fn fetch_remote_snapshot(
    client: &Client,
    snapshot_url: &str,
    config: &CatalogRemoteConfig,
) -> Result<Vec<u8>, CatalogSnapshotError> {
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
    apply_custom_headers(&mut headers, &config.headers, &HeaderPolicy::default());

    let response = client
        .get(snapshot_url)
        .headers(headers)
        .send()
        .await
        .map_err(|source| CatalogSnapshotError::FetchSnapshot {
            url: snapshot_url.to_string(),
            source,
        })?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(CatalogSnapshotError::FetchSnapshotHttp {
            url: snapshot_url.to_string(),
            status,
            body,
        });
    }

    response
        .bytes()
        .await
        .map(|bytes| bytes.to_vec())
        .map_err(|source| CatalogSnapshotError::FetchSnapshot {
            url: snapshot_url.to_string(),
            source,
        })
}

fn resolve_snapshot_url(
    manifest_url: &str,
    snapshot_url: &str,
) -> Result<String, CatalogSnapshotError> {
    if let Ok(url) = Url::parse(snapshot_url) {
        return Ok(url.to_string());
    }

    let base = Url::parse(manifest_url).map_err(|_| CatalogSnapshotError::InvalidManifestUrl {
        url: manifest_url.to_string(),
    })?;
    let joined = base
        .join(snapshot_url)
        .map_err(|_| CatalogSnapshotError::InvalidSnapshotUrl {
            url: snapshot_url.to_string(),
        })?;
    Ok(joined.to_string())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), CatalogSnapshotError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| CatalogSnapshotError::WriteFile {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    let temp_path = temporary_path_for(path);
    fs::write(&temp_path, bytes).map_err(|source| CatalogSnapshotError::WriteFile {
        path: temp_path.clone(),
        source,
    })?;

    if path.exists() {
        fs::remove_file(path).map_err(|source| CatalogSnapshotError::WriteFile {
            path: path.to_path_buf(),
            source,
        })?;
    }

    fs::rename(&temp_path, path).map_err(|source| CatalogSnapshotError::WriteFile {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

fn temporary_path_for(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("catalog");
    path.with_file_name(format!("{}.tmp-{}", file_name, uuid::Uuid::new_v4()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Derive the local sidecar manifest path for a snapshot file.
pub fn catalog_manifest_sidecar_path(snapshot_path: impl AsRef<Path>) -> PathBuf {
    let snapshot_path = snapshot_path.as_ref();
    let stem = snapshot_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("catalog");

    let file_name = match snapshot_path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => format!("{stem}.manifest.{ext}"),
        None => format!("{stem}.manifest.json"),
    };

    snapshot_path.with_file_name(file_name)
}

async fn send_json_request(
    client: &Client,
    provider: Provider,
    url: &str,
    headers: HeaderMap,
) -> Result<Value, ModelCatalogError> {
    send_json_request_with_query(client, provider, url, headers, &[]).await
}

async fn send_optional_json_request(
    client: &Client,
    provider: Provider,
    url: &str,
    headers: HeaderMap,
) -> Result<Option<Value>, ModelCatalogError> {
    let response = client
        .get(url)
        .headers(headers)
        .send()
        .await
        .map_err(|source| ModelCatalogError::Request {
            provider: provider.clone(),
            source,
        })?;

    let status = response.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(ModelCatalogError::Http {
            provider,
            status,
            body,
        });
    }

    response
        .json::<Value>()
        .await
        .map(Some)
        .map_err(|source| ModelCatalogError::Request { provider, source })
}

async fn send_json_request_with_query(
    client: &Client,
    provider: Provider,
    url: &str,
    headers: HeaderMap,
    query: &[(&str, String)],
) -> Result<Value, ModelCatalogError> {
    let response = client
        .get(url)
        .headers(headers)
        .query(query)
        .send()
        .await
        .map_err(|source| ModelCatalogError::Request {
            provider: provider.clone(),
            source,
        })?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(ModelCatalogError::Http {
            provider,
            status,
            body,
        });
    }

    response
        .json::<Value>()
        .await
        .map_err(|source| ModelCatalogError::Request { provider, source })
}

fn extract_openai_models(
    provider: &Provider,
    raw: &Value,
) -> Result<Vec<ProviderExtractedModel>, ModelCatalogError> {
    let data = value_array(provider, raw, "data")?;
    data.iter()
        .map(|item| extract_model_record(provider, item))
        .collect()
}

fn extract_anthropic_models(
    provider: &Provider,
    raw: &Value,
) -> Result<Vec<ProviderExtractedModel>, ModelCatalogError> {
    let data = value_array(provider, raw, "data")?;
    data.iter()
        .map(|item| extract_model_record(provider, item))
        .collect()
}

fn extract_model_record(
    provider: &Provider,
    item: &Value,
) -> Result<ProviderExtractedModel, ModelCatalogError> {
    let raw_id = model_identifier(item).ok_or_else(|| ModelCatalogError::InvalidResponse {
        provider: provider.clone(),
        message: "model entry is missing string field `id` or `name`".to_string(),
    })?;

    Ok(ProviderExtractedModel {
        provider: provider.clone(),
        raw_id: raw_id.to_string(),
        display_name: optional_string(item, &["display_name", "displayName", "name"]),
        description: optional_string(item, &["description"]),
        context_window: optional_u64(
            item,
            &[
                "context_window",
                "context_length",
                "max_context_length",
                "inputTokenLimit",
            ],
        )
        .or_else(|| {
            item.get("top_provider")
                .and_then(|v| optional_u64(v, &["context_length"]))
        }),
        max_output_tokens: optional_u64(
            item,
            &[
                "max_output_tokens",
                "max_completion_tokens",
                "max_tokens",
                "output_token_limit",
                "outputTokenLimit",
            ],
        )
        .or_else(|| {
            item.get("top_provider")
                .and_then(|v| optional_u64(v, &["max_completion_tokens", "max_output_tokens"]))
        }),
        max_input_tokens: optional_u64(
            item,
            &["max_input_tokens", "input_token_limit", "inputTokenLimit"],
        ),
        created_at: optional_timestamp(item, &["created_at", "created"]),
        modalities: collect_declared_modalities(item)
            .or_else(|| collect_architecture_modalities(item))
            .or_else(|| collect_bool_keys(item.get("capabilities"), "supports_")),
        capabilities: optional_string_array(item, &["capabilities"])
            .or_else(|| collect_capabilities_object(item.get("capabilities")))
            .or_else(|| collect_thinking_capability(item))
            .or_else(|| collect_supported_parameter_capabilities(item)),
        raw: item.clone(),
    })
}

fn value_array<'a>(
    provider: &Provider,
    raw: &'a Value,
    field: &str,
) -> Result<&'a Vec<Value>, ModelCatalogError> {
    raw.get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| ModelCatalogError::InvalidResponse {
            provider: provider.clone(),
            message: format!("response is missing array field `{}`", field),
        })
}

fn optional_string(item: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| item.get(*key).and_then(Value::as_str))
        .map(ToString::to_string)
}

fn optional_u64(item: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| parse_u64(item.get(*key)?))
}

fn parse_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse::<u64>().ok(),
        _ => None,
    }
}

fn optional_timestamp(item: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter().find_map(|key| parse_timestamp(item.get(*key)?))
}

fn parse_timestamp(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => number.as_i64(),
        Value::String(text) => DateTime::parse_from_rfc3339(text)
            .map(|ts| ts.with_timezone(&Utc).timestamp_millis())
            .ok()
            .or_else(|| text.parse::<i64>().ok()),
        _ => None,
    }
}

fn optional_string_array(item: &Value, keys: &[&str]) -> Option<Vec<String>> {
    keys.iter()
        .find_map(|key| parse_string_array(item.get(*key)?))
}

fn collect_declared_modalities(item: &Value) -> Option<Vec<String>> {
    let mut values = Vec::new();
    for key in [
        "modalities",
        "supported_modalities",
        "input_modalities",
        "output_modalities",
        "inputModalities",
        "outputModalities",
    ] {
        if let Some(items) = item.get(key).and_then(parse_string_array) {
            values.extend(items);
        }
    }

    let values = dedupe_strings(values);
    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

fn parse_string_array(value: &Value) -> Option<Vec<String>> {
    match value {
        Value::Array(values) => {
            let items: Vec<String> = values
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect();
            if items.is_empty() {
                None
            } else {
                Some(items)
            }
        }
        Value::String(text) => Some(vec![text.to_string()]),
        _ => None,
    }
}

fn collect_architecture_modalities(item: &Value) -> Option<Vec<String>> {
    let architecture = item.get("architecture")?;
    let mut items = Vec::new();
    if let Some(inputs) = architecture
        .get("input_modalities")
        .and_then(parse_string_array)
    {
        items.extend(inputs);
    }
    if let Some(outputs) = architecture
        .get("output_modalities")
        .and_then(parse_string_array)
    {
        items.extend(outputs);
    }

    let items = dedupe_strings(items);
    if items.is_empty() {
        None
    } else {
        Some(items)
    }
}

fn collect_bool_keys(value: Option<&Value>, prefix_to_strip: &str) -> Option<Vec<String>> {
    let object = value?.as_object()?;
    let mut items = Vec::new();
    for (key, value) in object {
        if value.as_bool() == Some(true) {
            items.push(key.trim_start_matches(prefix_to_strip).to_string());
        }
    }
    if items.is_empty() {
        None
    } else {
        Some(items)
    }
}

fn collect_capabilities_object(value: Option<&Value>) -> Option<Vec<String>> {
    let object = value?.as_object()?;
    let mut items = Vec::new();
    for (key, value) in object {
        if value.as_bool() == Some(true) {
            items.push(key.to_string());
        }
    }
    if items.is_empty() {
        None
    } else {
        Some(items)
    }
}

fn collect_thinking_capability(item: &Value) -> Option<Vec<String>> {
    if item.get("thinking").and_then(Value::as_bool) == Some(true) {
        Some(vec!["reasoning".to_string()])
    } else {
        None
    }
}

fn collect_supported_parameter_capabilities(item: &Value) -> Option<Vec<String>> {
    let parameters = item
        .get("supported_parameters")
        .and_then(parse_string_array)?;

    let mut items = Vec::new();
    for parameter in parameters {
        match parameter.as_str() {
            "reasoning" | "include_reasoning" => items.push("reasoning".to_string()),
            "tools" | "tool_choice" | "parallel_tool_calls" => items.push("tools".to_string()),
            "response_format" | "structured_outputs" => {
                items.push("structured_outputs".to_string())
            }
            _ => {}
        }
    }

    let items = dedupe_strings(items);
    if items.is_empty() {
        None
    } else {
        Some(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_alias_candidates_across_provider_variants() {
        let aliases = normalized_alias_candidates("claude-opus-4-6", None);
        assert!(aliases.contains(&"claude-opus-4-6".to_string()));

        let metadata = CatalogModelMetadata {
            canonical_model_key: "anthropic:claude-opus:4.6".to_string(),
            aliases: vec!["anthropic/claude-opus-4.6".to_string()],
            display_name: Some("Claude Opus 4.6".to_string()),
            description: None,
            context_window: None,
            max_output_tokens: None,
            max_input_tokens: None,
            modalities: None,
            capabilities: None,
            pricing: None,
            source: "openrouter".to_string(),
            raw: json!({}),
        };
        let store = InMemoryCatalogMetadataStore::new(vec![metadata]);
        let matched = store
            .find_by_raw_or_alias(&Provider::Anthropic, "claude-opus-4-6", &aliases)
            .expect("should match normalized alias");

        assert_eq!(
            matched.metadata.canonical_model_key,
            "anthropic:claude-opus:4.6"
        );
    }

    #[test]
    fn extracts_capabilities_from_supported_parameters() {
        let item = json!({
            "supported_parameters": [
                "max_tokens",
                "include_reasoning",
                "reasoning",
                "tool_choice",
                "tools",
                "response_format",
                "seed"
            ]
        });

        assert_eq!(
            collect_supported_parameter_capabilities(&item),
            Some(vec![
                "reasoning".to_string(),
                "tools".to_string(),
                "structured_outputs".to_string()
            ])
        );
    }
}

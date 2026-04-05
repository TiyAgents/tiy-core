//! Tests for file-backed catalog snapshots and remote refresh.

use serde_json::json;
use std::fs;
use std::path::PathBuf;
use tiycore::catalog::{
    build_catalog_snapshot, build_catalog_snapshot_manifest, catalog_manifest_sidecar_path,
    refresh_catalog_snapshot, save_catalog_snapshot, CatalogMetadataStore, CatalogModelMetadata,
    CatalogRefreshResult, CatalogRemoteConfig, FileCatalogMetadataStore,
};
use tiycore::types::Provider;
use uuid::Uuid;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn temp_catalog_path(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tiycore-{test_name}-{}", Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("temp dir should be created");
    dir.join("catalog.json")
}

fn sample_metadata() -> CatalogModelMetadata {
    CatalogModelMetadata {
        canonical_model_key: "anthropic:claude-opus:4.6".to_string(),
        aliases: vec!["anthropic/claude-opus-4.6".to_string()],
        display_name: Some("Claude Opus 4.6".to_string()),
        description: Some("High-end reasoning model".to_string()),
        context_window: Some(200_000),
        max_output_tokens: Some(32_000),
        max_input_tokens: Some(200_000),
        modalities: Some(vec!["text".to_string(), "image".to_string()]),
        capabilities: Some(vec!["reasoning".to_string(), "tools".to_string()]),
        pricing: Some(json!({"input": "15.0", "output": "75.0"})),
        source: "openrouter".to_string(),
        raw: json!({"id": "anthropic/claude-opus-4.6"}),
    }
}

#[test]
fn test_file_catalog_metadata_store_loads_snapshot() {
    let snapshot_path = temp_catalog_path("load");
    let snapshot = build_catalog_snapshot(
        "2026-03-19",
        "2026-03-19T00:00:00Z",
        vec![sample_metadata()],
    );
    let snapshot_bytes = serde_json::to_vec_pretty(&snapshot).expect("snapshot should serialize");
    let manifest = build_catalog_snapshot_manifest(
        snapshot.version.clone(),
        snapshot.generated_at.clone(),
        "catalog.json",
        &snapshot_bytes,
    );

    save_catalog_snapshot(&snapshot_path, &snapshot, &manifest)
        .expect("snapshot should be written");

    let store = FileCatalogMetadataStore::load(&snapshot_path).expect("snapshot should load");
    assert_eq!(store.snapshot().version, "2026-03-19");

    let matched = store
        .find_by_raw_or_alias(
            &Provider::Anthropic,
            "claude-opus-4-6",
            &["claude-opus-4-6".to_string()],
        )
        .expect("metadata should match normalized alias");

    assert_eq!(matched.metadata.source, "openrouter");
}

#[tokio::test]
async fn test_refresh_catalog_snapshot_downloads_remote_files() {
    let server = MockServer::start().await;
    let snapshot_path = temp_catalog_path("refresh-update");
    let snapshot = build_catalog_snapshot(
        "2026-03-20",
        "2026-03-20T00:00:00Z",
        vec![sample_metadata()],
    );
    let snapshot_bytes = serde_json::to_vec_pretty(&snapshot).expect("snapshot should serialize");
    let manifest = build_catalog_snapshot_manifest(
        snapshot.version.clone(),
        snapshot.generated_at.clone(),
        "catalog.json",
        &snapshot_bytes,
    );

    Mock::given(method("GET"))
        .and(path("/catalog/manifest.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&manifest))
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/catalog/catalog.json"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(snapshot_bytes.clone()))
        .mount(&server)
        .await;

    let result = refresh_catalog_snapshot(
        &snapshot_path,
        &CatalogRemoteConfig::new(format!("{}/catalog/manifest.json", server.uri())),
    )
    .await
    .expect("refresh should succeed");

    match result {
        CatalogRefreshResult::Updated {
            manifest: returned_manifest,
            created,
            ..
        } => {
            assert!(created);
            assert_eq!(returned_manifest.version, "2026-03-20");
        }
        other => panic!("unexpected refresh result: {other:?}"),
    }

    let local_manifest_path = catalog_manifest_sidecar_path(&snapshot_path);
    assert!(snapshot_path.exists());
    assert!(local_manifest_path.exists());

    let store = FileCatalogMetadataStore::load(&snapshot_path).expect("store should load");
    assert_eq!(store.snapshot().models.len(), 1);
}

#[tokio::test]
async fn test_refresh_catalog_snapshot_skips_when_manifest_unchanged() {
    let server = MockServer::start().await;
    let snapshot_path = temp_catalog_path("refresh-unchanged");
    let snapshot = build_catalog_snapshot(
        "2026-03-21",
        "2026-03-21T00:00:00Z",
        vec![sample_metadata()],
    );
    let snapshot_bytes = serde_json::to_vec_pretty(&snapshot).expect("snapshot should serialize");
    let manifest = build_catalog_snapshot_manifest(
        snapshot.version.clone(),
        snapshot.generated_at.clone(),
        "catalog.json",
        &snapshot_bytes,
    );

    save_catalog_snapshot(&snapshot_path, &snapshot, &manifest)
        .expect("local snapshot should be seeded");

    Mock::given(method("GET"))
        .and(path("/catalog/manifest.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&manifest))
        .mount(&server)
        .await;

    let result = refresh_catalog_snapshot(
        &snapshot_path,
        &CatalogRemoteConfig::new(format!("{}/catalog/manifest.json", server.uri())),
    )
    .await
    .expect("refresh should succeed");

    match result {
        CatalogRefreshResult::Unchanged {
            manifest: returned_manifest,
        } => {
            assert_eq!(returned_manifest.version, "2026-03-21");
        }
        other => panic!("expected unchanged result, got {other:?}"),
    }
}

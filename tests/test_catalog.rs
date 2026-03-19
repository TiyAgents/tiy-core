//! Tests for model catalog fetching and enrichment.

use serde_json::json;
use tiy_core::catalog::{
    enrich_manual_model, list_models, list_models_with_enrichment, CatalogModelMetadata,
    FetchModelsRequest, InMemoryCatalogMetadataStore, ModelCatalogError,
};
use tiy_core::types::Provider;
use wiremock::matchers::{header, method, path, query_param, query_param_is_missing};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_list_models_with_openai_enrichment() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                {
                    "id": "gpt-4.1",
                    "created": 1710000000,
                    "owned_by": "openai"
                }
            ]
        })))
        .mount(&server)
        .await;

    let store = InMemoryCatalogMetadataStore::new(vec![CatalogModelMetadata {
        canonical_model_key: "openai:gpt-4.1".to_string(),
        aliases: vec!["openai/gpt-4.1".to_string()],
        display_name: Some("GPT-4.1".to_string()),
        description: Some("General-purpose flagship".to_string()),
        context_window: Some(1_000_000),
        max_output_tokens: Some(32_768),
        max_input_tokens: Some(1_000_000),
        modalities: Some(vec!["text".to_string(), "image".to_string()]),
        capabilities: Some(vec!["tools".to_string(), "reasoning".to_string()]),
        pricing: Some(json!({"input": "2.0", "output": "8.0"})),
        source: "openrouter".to_string(),
        raw: json!({}),
    }]);

    let result = list_models_with_enrichment(
        FetchModelsRequest {
            provider: Provider::OpenAI,
            api_key: Some("test-key".to_string()),
            base_url: Some(format!("{}/v1", server.uri())),
            headers: None,
        },
        &store,
    )
    .await
    .expect("openai list should succeed");

    assert_eq!(result.models.len(), 1);
    let model = &result.models[0];
    assert_eq!(model.raw_id, "gpt-4.1");
    assert_eq!(model.canonical_model_key.as_deref(), Some("openai:gpt-4.1"));
    assert_eq!(model.display_name.as_deref(), Some("GPT-4.1"));
    assert_eq!(model.context_window, Some(1_000_000));
    assert_eq!(model.max_output_tokens, Some(32_768));
    assert_eq!(model.match_confidence, Some(1.0));
    assert_eq!(model.metadata_sources, vec!["openrouter".to_string()]);
    assert_eq!(result.raw_response["data"][0]["id"], "gpt-4.1");
}

#[tokio::test]
async fn test_list_models_for_anthropic_paginates() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(query_param("limit", "1000"))
        .and(query_param_is_missing("after_id"))
        .and(header("x-api-key", "anth-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {
                    "id": "claude-sonnet-4-20250514",
                    "display_name": "Claude Sonnet 4",
                    "context_window": 200000,
                    "max_output_tokens": 16000,
                    "created_at": "2025-05-14T00:00:00Z"
                }
            ],
            "has_more": true,
            "last_id": "cursor-1"
        })))
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(query_param("limit", "1000"))
        .and(query_param("after_id", "cursor-1"))
        .and(header("x-api-key", "anth-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {
                    "id": "claude-opus-4-6",
                    "display_name": "Claude Opus 4.6",
                    "context_window": 200000,
                    "max_output_tokens": 32000
                }
            ],
            "has_more": false,
            "last_id": "cursor-2"
        })))
        .mount(&server)
        .await;

    let result = list_models(FetchModelsRequest {
        provider: Provider::Anthropic,
        api_key: Some("anth-key".to_string()),
        base_url: Some(format!("{}/v1", server.uri())),
        headers: None,
    })
    .await
    .expect("anthropic list should succeed");

    assert_eq!(result.models.len(), 2);
    assert_eq!(result.models[0].raw_id, "claude-sonnet-4-20250514");
    assert_eq!(result.models[0].max_output_tokens, Some(16000));
    assert_eq!(result.models[1].raw_id, "claude-opus-4-6");
    assert_eq!(result.models[1].context_window, Some(200000));
    assert_eq!(
        result.raw_response["pages"].as_array().map(Vec::len),
        Some(2)
    );
}

#[tokio::test]
async fn test_list_models_rejects_unsupported_provider() {
    let error = list_models(FetchModelsRequest::new(Provider::Google))
        .await
        .expect_err("google should not be supported yet");

    match error {
        ModelCatalogError::UnsupportedProvider { provider } => {
            assert_eq!(provider, Provider::Google);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn test_enrich_manual_model_uses_snapshot_metadata() {
    let store = InMemoryCatalogMetadataStore::new(vec![CatalogModelMetadata {
        canonical_model_key: "openai:gpt-4.1".to_string(),
        aliases: vec!["openai/gpt-4.1".to_string()],
        display_name: Some("GPT-4.1".to_string()),
        description: Some("General-purpose flagship".to_string()),
        context_window: Some(1_000_000),
        max_output_tokens: Some(32_768),
        max_input_tokens: Some(1_000_000),
        modalities: Some(vec!["text".to_string(), "image".to_string()]),
        capabilities: Some(vec!["tools".to_string(), "reasoning".to_string()]),
        pricing: Some(json!({"input": "2.0", "output": "8.0"})),
        source: "openrouter".to_string(),
        raw: json!({}),
    }]);

    let model = enrich_manual_model(Provider::OpenAI, "openai/gpt-4.1", None, &store);

    assert_eq!(model.raw_id, "openai/gpt-4.1");
    assert_eq!(model.canonical_model_key.as_deref(), Some("openai:gpt-4.1"));
    assert_eq!(model.display_name.as_deref(), Some("GPT-4.1"));
    assert_eq!(model.context_window, Some(1_000_000));
    assert_eq!(model.max_output_tokens, Some(32_768));
    assert_eq!(model.match_confidence, Some(1.0));
    assert_eq!(model.metadata_sources, vec!["openrouter".to_string()]);
}

#[test]
fn test_enrich_manual_model_preserves_manual_display_name_without_snapshot_match() {
    let store = InMemoryCatalogMetadataStore::new(vec![]);

    let model = enrich_manual_model(
        Provider::OpenAI,
        "custom-model-id",
        Some("My Custom Model".to_string()),
        &store,
    );

    assert_eq!(model.raw_id, "custom-model-id");
    assert_eq!(model.display_name.as_deref(), Some("My Custom Model"));
    assert!(model.canonical_model_key.is_none());
    assert!(model.context_window.is_none());
    assert!(model.metadata_sources.is_empty());
}

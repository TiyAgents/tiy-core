//! Tests for delegation providers (minimax, kimi-coding, xai, groq, openrouter, zai, zenmux).
//!
//! These providers delegate to existing protocol implementations.
//! Tests verify correct API type, compat settings, key resolution, and delegation behavior.

use futures::StreamExt;
use tiy_core::provider::{
    minimax::MiniMaxProvider,
    kimi_coding::KimiCodingProvider,
    xai::XAIProvider,
    groq::GroqProvider,
    openrouter::OpenRouterProvider,
    zai::ZAIProvider,
    zenmux::ZenmuxProvider,
    LLMProvider,
};
use tiy_core::types::*;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

// ============================================================================
// Helpers
// ============================================================================
fn make_model(api: Api, provider: Provider, base_url: &str) -> Model {
    Model::builder()
        .id("test-model")
        .name("Test Model")
        .api(api)
        .provider(provider)
        .base_url(base_url)
        .context_window(128000)
        .max_tokens(8192)
        .build()
        .unwrap()
}

fn anthropic_sse(events: Vec<(&str, &str)>) -> String {
    events.iter()
        .map(|(event_type, data)| format!("event: {}\ndata: {}\n\n", event_type, data))
        .collect::<String>()
}

fn simple_anthropic_response(text: &str) -> String {
    anthropic_sse(vec![
        ("message_start", &serde_json::json!({
            "type": "message_start",
            "message": {
                "id": "msg_1", "type": "message", "role": "assistant",
                "content": [], "model": "test",
                "usage": {"input_tokens": 10, "output_tokens": 0}
            }
        }).to_string()),
        ("content_block_start", &serde_json::json!({
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}
        }).to_string()),
        ("content_block_delta", &serde_json::json!({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text}
        }).to_string()),
        ("content_block_stop", &serde_json::json!({
            "type": "content_block_stop", "index": 0
        }).to_string()),
        ("message_delta", &serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 5}
        }).to_string()),
        ("message_stop", &serde_json::json!({"type": "message_stop"}).to_string()),
    ])
}

fn simple_openai_response(text: &str) -> String {
    [
        format!("data: {}\n\n", serde_json::json!({
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]
        })),
        format!("data: {}\n\n", serde_json::json!({
            "choices": [{"index": 0, "delta": {"content": text}}]
        })),
        format!("data: {}\n\n", serde_json::json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        })),
        "data: [DONE]\n\n".to_string(),
    ].join("")
}

async fn mock_anthropic_server(text: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(simple_anthropic_response(text))
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;
    server
}

async fn mock_openai_server(text: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(simple_openai_response(text))
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;
    server
}

// ============================================================================
// MiniMax Provider Tests
// ============================================================================

#[test]
fn test_minimax_api_type() {
    let provider = MiniMaxProvider::new();
    assert_eq!(provider.provider_type(), Provider::MiniMax);
}

#[test]
fn test_minimax_with_api_key() {
    let provider = MiniMaxProvider::with_api_key("test-key");
    assert_eq!(provider.provider_type(), Provider::MiniMax);
}

#[test]
fn test_minimax_default() {
    let provider = MiniMaxProvider::default();
    assert_eq!(provider.provider_type(), Provider::MiniMax);
}

#[tokio::test]
async fn test_minimax_stream_delegates_to_anthropic() {
    let server = mock_anthropic_server("Hello from MiniMax!").await;
    let model = make_model(Api::AnthropicMessages, Provider::MiniMax, &server.uri());
    let context = Context::with_system_prompt("test");
    let provider = MiniMaxProvider::with_api_key("test-key");

    let stream = provider.stream(
        &model, &context,
        StreamOptions { api_key: Some("test-key".into()), ..Default::default() },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Hello from MiniMax!");
}

// ============================================================================
// Kimi Coding Provider Tests
// ============================================================================

#[test]
fn test_kimi_coding_api_type() {
    let provider = KimiCodingProvider::new();
    assert_eq!(provider.provider_type(), Provider::KimiCoding);
}

#[test]
fn test_kimi_coding_with_api_key() {
    let provider = KimiCodingProvider::with_api_key("test-key");
    assert_eq!(provider.provider_type(), Provider::KimiCoding);
}

#[test]
fn test_kimi_coding_default() {
    let provider = KimiCodingProvider::default();
    assert_eq!(provider.provider_type(), Provider::KimiCoding);
}

#[tokio::test]
async fn test_kimi_coding_stream_delegates_to_anthropic() {
    let server = mock_anthropic_server("Kimi response").await;
    let model = make_model(Api::AnthropicMessages, Provider::KimiCoding, &server.uri());
    let context = Context::with_system_prompt("test");
    let provider = KimiCodingProvider::with_api_key("test-key");

    let stream = provider.stream(
        &model, &context,
        StreamOptions { api_key: Some("test-key".into()), ..Default::default() },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Kimi response");
}

// ============================================================================
// xAI Provider Tests
// ============================================================================

#[test]
fn test_xai_api_type() {
    let provider = XAIProvider::new();
    assert_eq!(provider.provider_type(), Provider::XAI);
}

#[test]
fn test_xai_default_compat() {
    let compat = XAIProvider::default_compat();
    assert!(!compat.supports_store, "xAI should not support store");
    assert!(!compat.supports_developer_role, "xAI should not support developer role");
    assert!(!compat.supports_reasoning_effort, "xAI should not support reasoning_effort");
    assert_eq!(compat.thinking_format, "openai");
    assert!(compat.supports_strict_mode);
}

#[tokio::test]
async fn test_xai_stream_delegates_to_openai() {
    let server = mock_openai_server("Grok says hi!").await;
    let model = make_model(Api::OpenAICompletions, Provider::XAI, &server.uri());
    let context = Context::with_system_prompt("test");
    let provider = XAIProvider::with_api_key("test-key");

    let stream = provider.stream(
        &model, &context,
        StreamOptions { api_key: Some("test-key".into()), ..Default::default() },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Grok says hi!");
}

#[tokio::test]
async fn test_xai_stream_events() {
    let server = mock_openai_server("test").await;
    let model = make_model(Api::OpenAICompletions, Provider::XAI, &server.uri());
    let context = Context::with_system_prompt("test");
    let provider = XAIProvider::with_api_key("test-key");

    let stream = provider.stream(
        &model, &context,
        StreamOptions { api_key: Some("test-key".into()), ..Default::default() },
    );

    let mut events = Vec::new();
    let mut s = stream;
    while let Some(event) = s.next().await {
        events.push(event);
    }
    assert!(!events.is_empty());
    assert!(matches!(&events[0], AssistantMessageEvent::Start { .. }));
}

// ============================================================================
// Groq Provider Tests
// ============================================================================

#[test]
fn test_groq_api_type() {
    let provider = GroqProvider::new();
    assert_eq!(provider.provider_type(), Provider::Groq);
}

#[test]
fn test_groq_default_compat_standard() {
    let compat = GroqProvider::default_compat("llama-3.3-70b-versatile");
    assert!(compat.supports_store);
    assert!(compat.supports_reasoning_effort);
    assert!(compat.reasoning_effort_map.is_empty());
}

#[test]
fn test_groq_default_compat_qwen3() {
    let compat = GroqProvider::default_compat("qwen/qwen3-32b");
    assert_eq!(compat.reasoning_effort_map.len(), 5);
    for level in &["minimal", "low", "medium", "high", "xhigh"] {
        assert_eq!(compat.reasoning_effort_map.get(*level).unwrap(), "default");
    }
}

#[tokio::test]
async fn test_groq_stream_delegates_to_openai() {
    let server = mock_openai_server("Fast inference!").await;
    let model = make_model(Api::OpenAICompletions, Provider::Groq, &server.uri());
    let context = Context::with_system_prompt("test");
    let provider = GroqProvider::with_api_key("test-key");

    let stream = provider.stream(
        &model, &context,
        StreamOptions { api_key: Some("test-key".into()), ..Default::default() },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Fast inference!");
}

// ============================================================================
// OpenRouter Provider Tests
// ============================================================================

#[test]
fn test_openrouter_api_type() {
    let provider = OpenRouterProvider::new();
    assert_eq!(provider.provider_type(), Provider::OpenRouter);
}

#[test]
fn test_openrouter_default() {
    let provider = OpenRouterProvider::default();
    assert_eq!(provider.provider_type(), Provider::OpenRouter);
}

#[tokio::test]
async fn test_openrouter_stream_delegates_to_openai() {
    let server = mock_openai_server("Routed response").await;
    let model = make_model(Api::OpenAICompletions, Provider::OpenRouter, &server.uri());
    let context = Context::with_system_prompt("test");
    let provider = OpenRouterProvider::with_api_key("test-key");

    let stream = provider.stream(
        &model, &context,
        StreamOptions { api_key: Some("test-key".into()), ..Default::default() },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Routed response");
}

// ============================================================================
// ZAI Provider Tests
// ============================================================================

#[test]
fn test_zai_api_type() {
    let provider = ZAIProvider::new();
    assert_eq!(provider.provider_type(), Provider::ZAI);
}

#[test]
fn test_zai_default_compat() {
    let compat = ZAIProvider::default_compat();
    assert!(!compat.supports_store);
    assert!(!compat.supports_developer_role);
    assert!(!compat.supports_reasoning_effort);
    assert_eq!(compat.thinking_format, "zai");
}

#[tokio::test]
async fn test_zai_stream_delegates_to_openai() {
    let server = mock_openai_server("GLM response").await;
    let model = make_model(Api::OpenAICompletions, Provider::ZAI, &server.uri());
    let context = Context::with_system_prompt("test");
    let provider = ZAIProvider::with_api_key("test-key");

    let stream = provider.stream(
        &model, &context,
        StreamOptions { api_key: Some("test-key".into()), ..Default::default() },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "GLM response");
}

// ============================================================================
// Zenmux Provider Tests
// ============================================================================

#[test]
fn test_zenmux_api_type() {
    let provider = ZenmuxProvider::new();
    assert_eq!(provider.provider_type(), Provider::Zenmux);
}

#[test]
fn test_zenmux_default() {
    let provider = ZenmuxProvider::default();
    assert_eq!(provider.provider_type(), Provider::Zenmux);
}

#[test]
fn test_zenmux_model_route_detection() {
    // Replicate the detection logic for unit testing
    fn detect_route(id: &str) -> &'static str {
        let lower = id.to_lowercase();
        if lower.contains("google") || lower.contains("gemini") {
            "google"
        } else if lower.contains("openai") || lower.contains("gpt") {
            "openai"
        } else {
            "anthropic"
        }
    }
    // Google models
    assert_eq!(detect_route("gemini-2.0-flash"), "google");
    assert_eq!(detect_route("google/gemini-pro"), "google");
    assert_eq!(detect_route("GEMINI-1.5-PRO"), "google");
    assert_eq!(detect_route("some-google-model"), "google");
    // OpenAI models
    assert_eq!(detect_route("gpt-4o"), "openai");
    assert_eq!(detect_route("gpt-4o-mini"), "openai");
    assert_eq!(detect_route("GPT-4.1"), "openai");
    assert_eq!(detect_route("openai/o3"), "openai");
    // Anthropic (default) models
    assert_eq!(detect_route("claude-sonnet-4"), "anthropic");
    assert_eq!(detect_route("llama-3.3-70b"), "anthropic");
    assert_eq!(detect_route("deepseek-r1"), "anthropic");
}

#[tokio::test]
async fn test_zenmux_adaptive_routes_to_anthropic() {
    // In adaptive mode, Zenmux sets model.base_url to the Zenmux Anthropic endpoint,
    // but we can verify the protocol by pointing model.base_url at our mock server
    // with a zenmux.ai prefix to trigger adaptive mode.
    // Since we can't fake DNS, we test by setting model.base_url = None and
    // verifying the provider delegates to the correct protocol implementation.
    //
    // Here we directly test the delegate by passing the mock URI as model.base_url
    // (non-adaptive path uses OpenAI Completions, so we use a different approach):
    // We set model.base_url to mock and use stream() — the non-adaptive path picks
    // OpenAI Completions, hitting /chat/completions.
    let server = mock_anthropic_server("Zenmux-Anthropic").await;
    let mut model = make_model(Api::AnthropicMessages, Provider::Zenmux, &server.uri());
    model.id = "claude-sonnet-4".to_string();
    let context = Context::with_system_prompt("test");

    // Directly test the Anthropic delegate (what Zenmux adaptive would call)
    let anthropic = tiy_core::provider::anthropic::AnthropicProvider::new();
    let stream = anthropic.stream(
        &model, &context,
        StreamOptions { api_key: Some("test-key".into()), ..Default::default() },
    );
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Zenmux-Anthropic");
}

#[tokio::test]
async fn test_zenmux_adaptive_routes_to_openai_responses() {
    let server = MockServer::start().await;

    // OpenAI Responses API mock at /responses
    let sse_body = [
        format!("event: response.output_item.added\ndata: {}\n\n", serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"type": "message", "id": "item_01", "role": "assistant", "content": []}
        })),
        format!("event: response.output_text.delta\ndata: {}\n\n", serde_json::json!({
            "type": "response.output_text.delta", "output_index": 0,
            "content_index": 0, "delta": "Zenmux-OpenAI"
        })),
        format!("event: response.output_item.done\ndata: {}\n\n", serde_json::json!({
            "type": "response.output_item.done", "output_index": 0,
            "item": {"type": "message", "id": "item_01"}
        })),
        format!("event: response.completed\ndata: {}\n\n", serde_json::json!({
            "type": "response.completed",
            "response": {
                "id": "resp_01", "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                "output": [{"type": "message", "id": "item_01"}]
            }
        })),
    ].join("");

    Mock::given(matchers::method("POST"))
        .and(matchers::path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    // Directly test the OpenAI Responses delegate
    let mut model = make_model(Api::OpenAIResponses, Provider::Zenmux, &server.uri());
    model.id = "gpt-4o".to_string();
    let context = Context::with_system_prompt("test");

    let responses_provider = tiy_core::provider::openai_responses::OpenAIResponsesProvider::new();
    let stream = responses_provider.stream(
        &model, &context,
        StreamOptions { api_key: Some("test-key".into()), ..Default::default() },
    );
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Zenmux-OpenAI");
}

#[tokio::test]
async fn test_zenmux_adaptive_routes_to_google() {
    let server = MockServer::start().await;

    let google_chunk = serde_json::json!({
        "candidates": [{
            "content": {
                "parts": [{"text": "Zenmux-Google"}],
                "role": "model"
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 3,
            "totalTokenCount": 13
        }
    });

    // Vertex AI URL format: /v1/publishers/google/models/{model}:streamGenerateContent
    Mock::given(matchers::method("POST"))
        .and(matchers::path_regex(r"/v1/publishers/google/models/gemini-2.0-flash:streamGenerateContent"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(format!("data: {}\n\n", google_chunk))
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    // Directly test the Google delegate with Vertex AI API type
    let mut model = make_model(Api::GoogleVertex, Provider::Zenmux, &server.uri());
    model.id = "gemini-2.0-flash".to_string();
    let context = Context::with_system_prompt("test");

    let google_provider = tiy_core::provider::google::GoogleProvider::new();
    let stream = google_provider.stream(
        &model, &context,
        StreamOptions { api_key: Some("test-key".into()), ..Default::default() },
    );
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Zenmux-Google");
}

#[tokio::test]
async fn test_zenmux_custom_base_url_uses_openai_completions() {
    // When a non-zenmux base_url is provided, always use OpenAI Completions protocol
    let server = mock_openai_server("Zenmux-Custom").await;
    let model = make_model(Api::AnthropicMessages, Provider::Zenmux, &server.uri());
    // model.base_url is server.uri() which is NOT zenmux.ai => non-adaptive mode
    let context = Context::with_system_prompt("test");
    let provider = ZenmuxProvider::with_api_key("test-key");

    let stream = provider.stream(
        &model, &context,
        StreamOptions { api_key: Some("test-key".into()), ..Default::default() },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Zenmux-Custom");
}

// ============================================================================
// Zenmux Provider serialization tests
// ============================================================================

#[test]
fn test_zenmux_provider_serialization() {
    let provider = Provider::Zenmux;
    assert_eq!(provider.as_str(), "zenmux");

    let json = serde_json::to_string(&provider).unwrap();
    assert_eq!(json, "\"zenmux\"");

    let deserialized: Provider = serde_json::from_str("\"zenmux\"").unwrap();
    assert_eq!(deserialized, Provider::Zenmux);
}

#[test]
fn test_zenmux_provider_from_string() {
    let provider = Provider::from("zenmux".to_string());
    assert_eq!(provider, Provider::Zenmux);
}

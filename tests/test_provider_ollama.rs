//! Tests for Ollama provider.

use tiycore::provider::ollama::OllamaProvider;
use tiycore::provider::LLMProtocol;
use tiycore::types::*;
use wiremock::matchers;
use wiremock::{Mock, MockServer, ResponseTemplate};

fn simple_openai_response(text: &str) -> String {
    [
        format!(
            "data: {}\n\n",
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]
            })
        ),
        format!(
            "data: {}\n\n",
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"content": text}}]
            })
        ),
        format!(
            "data: {}\n\n",
            serde_json::json!({
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            })
        ),
        "data: [DONE]\n\n".to_string(),
    ]
    .join("")
}

fn make_model(base_url: &str) -> Model {
    Model::builder()
        .id("llama3")
        .name("Llama 3")
        .api(Api::OpenAICompletions)
        .provider(Provider::Ollama)
        .base_url(base_url)
        .context_window(128000)
        .max_tokens(8192)
        .build()
        .unwrap()
}

// ============================================================================
// Constructor tests
// ============================================================================

#[test]
fn test_ollama_new() {
    let provider = OllamaProvider::new();
    assert_eq!(provider.provider_type(), Provider::Ollama);
}

#[test]
fn test_ollama_default() {
    let provider = OllamaProvider::default();
    assert_eq!(provider.provider_type(), Provider::Ollama);
}

#[test]
fn test_ollama_with_base_url() {
    let provider = OllamaProvider::with_base_url("http://custom:8080/v1");
    assert_eq!(provider.provider_type(), Provider::Ollama);
}

// ============================================================================
// Streaming tests (via wiremock)
// ============================================================================

#[tokio::test]
async fn test_ollama_stream_delegates_to_openai_completions() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(simple_openai_response("Ollama says hello"))
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OllamaProvider::with_base_url(server.uri());
    let model = make_model(&server.uri());
    let context = Context::with_system_prompt("test");
    let stream = provider.stream(
        &model,
        &context,
        StreamOptions {
            api_key: Some("unused".into()),
            ..Default::default()
        },
    );
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Ollama says hello");
}

#[tokio::test]
async fn test_ollama_stream_simple_delegates_to_openai_completions() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(simple_openai_response("simple response"))
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OllamaProvider::with_base_url(server.uri());
    let model = make_model(&server.uri());
    let context = Context::with_system_prompt("test");
    let stream = provider.stream_simple(
        &model,
        &context,
        SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("unused".into()),
                ..Default::default()
            },
            reasoning: None,
            thinking_budget_tokens: None,
        },
    );
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "simple response");
}

//! Google Generative AI provider.
//!
//! Implements streaming via Google's SSE protocol with JSON chunks containing
//! response candidates with parts-based content format.

/// Default base URL for Google Generative AI API.
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

use crate::provider::LLMProvider;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Google Generative AI provider.
pub struct GoogleProvider {
    client: Client,
    default_api_key: Option<String>,
}

impl GoogleProvider {
    /// Create a new Google provider.
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            default_api_key: None,
        }
    }

    /// Create a provider with a default API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            default_api_key: Some(api_key.into()),
        }
    }

    /// Resolve API key from options, default, or environment.
    fn resolve_api_key(&self, options: &StreamOptions) -> String {
        if let Some(ref key) = options.api_key {
            return key.clone();
        }
        if let Some(ref key) = self.default_api_key {
            return key.clone();
        }
        std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .unwrap_or_default()
    }
}

impl Default for GoogleProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for GoogleProvider {
    fn provider_type(&self) -> Provider {
        Provider::Google
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let stream = AssistantMessageEventStream::new_assistant_stream();
        let stream_clone = stream.clone();

        let model = model.clone();
        let context = context.clone();
        let client = self.client.clone();
        let api_key = self.resolve_api_key(&options);

        tokio::spawn(async move {
            if let Err(e) = run_stream(client, &model, &context, options, api_key, stream_clone).await {
                tracing::error!("Google stream error: {}", e);
            }
        });

        stream
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.stream(model, context, options.base)
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Google Generative AI request.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleRequest {
    contents: Vec<GoogleContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GoogleSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GoogleGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GoogleTool>>,
}

#[derive(Debug, Serialize)]
struct GoogleSystemInstruction {
    parts: Vec<GooglePart>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GoogleContent {
    role: String,
    parts: Vec<GooglePart>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GooglePart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<GoogleFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<GoogleFunctionResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<GoogleInlineData>,
}

impl GooglePart {
    fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            thought: None,
            thought_signature: None,
            function_call: None,
            function_response: None,
            inline_data: None,
        }
    }

    fn thinking(text: impl Into<String>, signature: Option<String>) -> Self {
        Self {
            text: Some(text.into()),
            thought: Some(true),
            thought_signature: signature,
            function_call: None,
            function_response: None,
            inline_data: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GoogleFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GoogleFunctionResponse {
    name: String,
    response: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GoogleInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleTool {
    function_declarations: Vec<GoogleFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleFunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

// ============================================================================
// SSE Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleStreamChunk {
    candidates: Option<Vec<GoogleCandidate>>,
    usage_metadata: Option<GoogleUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleCandidate {
    content: Option<GoogleContent>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleUsageMetadata {
    #[serde(default)]
    prompt_token_count: u64,
    #[serde(default)]
    candidates_token_count: u64,
    #[serde(default)]
    #[allow(dead_code)]
    thoughts_token_count: u64,
}

// ============================================================================
// Tool call ID generator
// ============================================================================

static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

fn generate_tool_call_id(name: &str) -> String {
    let counter = TOOL_CALL_COUNTER.fetch_add(1, AtomicOrdering::SeqCst);
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("{}_{}", name, timestamp + counter as u128)
}

// ============================================================================
// Message Conversion
// ============================================================================

fn convert_messages(context: &Context) -> Vec<GoogleContent> {
    let mut contents = Vec::new();

    for msg in &context.messages {
        match msg {
            Message::User(user_msg) => {
                let parts = match &user_msg.content {
                    UserContent::Text(text) => vec![GooglePart::text(text)],
                    UserContent::Blocks(blocks) => {
                        blocks.iter().filter_map(|b| match b {
                            ContentBlock::Text(t) => Some(GooglePart::text(&t.text)),
                            ContentBlock::Image(img) => Some(GooglePart {
                                text: None,
                                thought: None,
                                thought_signature: None,
                                function_call: None,
                                function_response: None,
                                inline_data: Some(GoogleInlineData {
                                    mime_type: img.mime_type.clone(),
                                    data: img.data.clone(),
                                }),
                            }),
                            _ => None,
                        }).collect()
                    }
                };
                contents.push(GoogleContent {
                    role: "user".to_string(),
                    parts,
                });
            }
            Message::Assistant(assistant_msg) => {
                if assistant_msg.stop_reason == StopReason::Error
                    || assistant_msg.stop_reason == StopReason::Aborted
                {
                    continue;
                }

                let mut parts = Vec::new();
                for block in &assistant_msg.content {
                    match block {
                        ContentBlock::Text(t) if !t.text.trim().is_empty() => {
                            parts.push(GooglePart::text(&t.text));
                        }
                        ContentBlock::Thinking(t) if !t.thinking.trim().is_empty() => {
                            parts.push(GooglePart::thinking(
                                &t.thinking,
                                t.thinking_signature.clone(),
                            ));
                        }
                        ContentBlock::ToolCall(tc) => {
                            parts.push(GooglePart {
                                text: None,
                                thought: None,
                                thought_signature: tc.thought_signature.clone(),
                                function_call: Some(GoogleFunctionCall {
                                    name: tc.name.clone(),
                                    args: tc.arguments.clone(),
                                }),
                                function_response: None,
                                inline_data: None,
                            });
                        }
                        _ => {}
                    }
                }

                if !parts.is_empty() {
                    contents.push(GoogleContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
            }
            Message::ToolResult(tool_result) => {
                let text: String = tool_result
                    .content
                    .iter()
                    .filter_map(|b| b.as_text())
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");

                let response_value = if tool_result.is_error {
                    serde_json::json!({ "error": text })
                } else {
                    serde_json::json!({ "result": text })
                };

                let part = GooglePart {
                    text: None,
                    thought: None,
                    thought_signature: None,
                    function_call: None,
                    function_response: Some(GoogleFunctionResponse {
                        name: tool_result.tool_name.clone(),
                        response: response_value,
                    }),
                    inline_data: None,
                };

                // Merge with last user/function message if possible
                if let Some(last) = contents.last_mut() {
                    if last.role == "user" {
                        last.parts.push(part);
                        continue;
                    }
                }

                contents.push(GoogleContent {
                    role: "user".to_string(),
                    parts: vec![part],
                });
            }
        }
    }

    contents
}

fn convert_tools(tools: &[Tool]) -> Vec<GoogleTool> {
    let declarations: Vec<GoogleFunctionDeclaration> = tools
        .iter()
        .map(|t| GoogleFunctionDeclaration {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.parameters.clone(),
        })
        .collect();

    if declarations.is_empty() {
        Vec::new()
    } else {
        vec![GoogleTool {
            function_declarations: declarations,
        }]
    }
}

// ============================================================================
// Streaming Implementation
// ============================================================================

async fn run_stream(
    client: Client,
    model: &Model,
    context: &Context,
    options: StreamOptions,
    api_key: String,
    stream: AssistantMessageEventStream,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut output = AssistantMessage::builder()
        .api(Api::GoogleGenerativeAi)
        .provider(model.provider.clone())
        .model(model.id.clone())
        .stop_reason(StopReason::Stop)
        .usage(Usage::default())
        .build()?;

    let contents = convert_messages(context);
    let tools = context.tools.as_ref().map(|t| convert_tools(t));

    let system_instruction = context.system_prompt.as_ref().map(|prompt| {
        GoogleSystemInstruction {
            parts: vec![GooglePart::text(prompt)],
        }
    });

    let request = GoogleRequest {
        contents,
        system_instruction,
        generation_config: Some(GoogleGenerationConfig {
            temperature: options.temperature,
            max_output_tokens: options.max_tokens.or(Some(model.max_tokens)),
        }),
        tools,
    };

    // Google API URL: {base_url}/models/{model_id}:streamGenerateContent?alt=sse
    let base = options.base_url.as_deref()
        .or(model.base_url.as_deref())
        .unwrap_or(DEFAULT_BASE_URL);
    let url = format!(
        "{}/models/{}:streamGenerateContent?alt=sse",
        base, model.id
    );

    tracing::info!(
        url = %url,
        model = %model.id,
        provider = %model.provider,
        content_count = request.contents.len(),
        has_tools = request.tools.is_some(),
        "Sending Google GenerativeAI request"
    );
    tracing::debug!(request_body = %serde_json::to_string(&request).unwrap_or_default(), "Request payload");

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse()?);
    headers.insert("x-goog-api-key", api_key.parse()?);

    // Add custom headers
    if let Some(ref custom_headers) = options.headers {
        for (key, value) in custom_headers {
            if let Ok(header_name) = reqwest::header::HeaderName::try_from(key.clone()) {
                if let Ok(header_value) = reqwest::header::HeaderValue::try_from(value.clone()) {
                    headers.insert(header_name, header_value);
                }
            }
        }
    }

    let response = client
        .post(&url)
        .headers(headers)
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        tracing::error!(
            model = %model.id,
            status = %status,
            response_body = %body,
            "Google GenerativeAI request failed"
        );
        output.stop_reason = StopReason::Error;
        output.error_message = Some(format!("HTTP {}: {}", status, body));
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: output,
        });
        stream.end(None);
        return Ok(());
    }

    // Send start event
    stream.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });

    let mut current_text_index: Option<usize> = None;
    let mut current_thinking_index: Option<usize> = None;
    let mut line_buffer = String::new();

    let mut byte_stream = response.bytes_stream();
    while let Some(chunk_result) = byte_stream.next().await {
        let chunk = chunk_result?;
        let text = String::from_utf8_lossy(&chunk);
        line_buffer.push_str(&text);

        while let Some(newline_pos) = line_buffer.find('\n') {
            let line = line_buffer[..newline_pos].trim_end_matches('\r').to_string();
            line_buffer = line_buffer[newline_pos + 1..].to_string();

            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];
            if data.is_empty() || data == "[DONE]" {
                continue;
            }

            let parsed: Result<GoogleStreamChunk, _> = serde_json::from_str(data);
            if let Ok(chunk_data) = parsed {
                // Handle usage metadata
                if let Some(ref usage) = chunk_data.usage_metadata {
                    output.usage.input = usage.prompt_token_count;
                    output.usage.output = usage.candidates_token_count;
                    output.usage.total_tokens = output.usage.input + output.usage.output;
                }

                if let Some(candidates) = chunk_data.candidates {
                    for candidate in &candidates {
                        // Handle finish reason
                        if let Some(ref reason) = candidate.finish_reason {
                            output.stop_reason = match reason.as_str() {
                                "STOP" => StopReason::Stop,
                                "MAX_TOKENS" => StopReason::Length,
                                "SAFETY" | "RECITATION" | "BLOCKLIST" => StopReason::Error,
                                _ => StopReason::Stop,
                            };
                        }

                        if let Some(ref content) = candidate.content {
                            for part in &content.parts {
                                // Handle thinking content
                                if part.thought == Some(true) {
                                    if let Some(ref thinking_text) = part.text {
                                        if !thinking_text.is_empty() {
                                            if current_thinking_index.is_none() {
                                                let idx = output.content.len();
                                                output.content.push(ContentBlock::Thinking(
                                                    ThinkingContent::new(""),
                                                ));
                                                current_thinking_index = Some(idx);
                                                stream.push(AssistantMessageEvent::ThinkingStart {
                                                    content_index: idx,
                                                    partial: output.clone(),
                                                });
                                            }

                                            let idx = current_thinking_index.unwrap();
                                            if let Some(ContentBlock::Thinking(ref mut t)) =
                                                output.content.get_mut(idx)
                                            {
                                                t.thinking.push_str(thinking_text);
                                                // Store thought signature if present
                                                if let Some(ref sig) = part.thought_signature {
                                                    t.thinking_signature = Some(sig.clone());
                                                }
                                            }
                                            stream.push(AssistantMessageEvent::ThinkingDelta {
                                                content_index: idx,
                                                delta: thinking_text.clone(),
                                                partial: output.clone(),
                                            });
                                        }
                                    }
                                    continue;
                                }

                                // Handle function call (arrives complete, not streamed)
                                if let Some(ref fc) = part.function_call {
                                    // End current thinking block if active
                                    if let Some(idx) = current_thinking_index.take() {
                                        let content = output.content.get(idx)
                                            .and_then(|b| b.as_thinking())
                                            .map(|t| t.thinking.clone())
                                            .unwrap_or_default();
                                        stream.push(AssistantMessageEvent::ThinkingEnd {
                                            content_index: idx,
                                            content,
                                            partial: output.clone(),
                                        });
                                    }
                                    // End current text block if active
                                    if let Some(idx) = current_text_index.take() {
                                        let content = output.content.get(idx)
                                            .and_then(|b| b.as_text())
                                            .map(|t| t.text.clone())
                                            .unwrap_or_default();
                                        stream.push(AssistantMessageEvent::TextEnd {
                                            content_index: idx,
                                            content,
                                            partial: output.clone(),
                                        });
                                    }

                                    let tool_call_id = generate_tool_call_id(&fc.name);
                                    let mut tool_call = ToolCall::new(
                                        &tool_call_id,
                                        &fc.name,
                                        fc.args.clone(),
                                    );
                                    tool_call.thought_signature = part.thought_signature.clone();

                                    let idx = output.content.len();
                                    output.content.push(ContentBlock::ToolCall(tool_call.clone()));
                                    output.stop_reason = StopReason::ToolUse;

                                    stream.push(AssistantMessageEvent::ToolCallStart {
                                        content_index: idx,
                                        partial: output.clone(),
                                    });
                                    stream.push(AssistantMessageEvent::ToolCallEnd {
                                        content_index: idx,
                                        tool_call,
                                        partial: output.clone(),
                                    });
                                    continue;
                                }

                                // Handle text content
                                if let Some(ref text_content) = part.text {
                                    if !text_content.is_empty() {
                                        // End thinking block if transitioning to text
                                        if let Some(idx) = current_thinking_index.take() {
                                            let content = output.content.get(idx)
                                                .and_then(|b| b.as_thinking())
                                                .map(|t| t.thinking.clone())
                                                .unwrap_or_default();
                                            stream.push(AssistantMessageEvent::ThinkingEnd {
                                                content_index: idx,
                                                content,
                                                partial: output.clone(),
                                            });
                                        }

                                        if current_text_index.is_none() {
                                            let idx = output.content.len();
                                            output.content.push(ContentBlock::Text(
                                                TextContent::new(""),
                                            ));
                                            current_text_index = Some(idx);
                                            stream.push(AssistantMessageEvent::TextStart {
                                                content_index: idx,
                                                partial: output.clone(),
                                            });
                                        }

                                        let idx = current_text_index.unwrap();
                                        if let Some(ContentBlock::Text(ref mut t)) =
                                            output.content.get_mut(idx)
                                        {
                                            t.text.push_str(text_content);
                                        }
                                        stream.push(AssistantMessageEvent::TextDelta {
                                            content_index: idx,
                                            delta: text_content.clone(),
                                            partial: output.clone(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // End any active blocks
    if let Some(idx) = current_thinking_index {
        let content = output.content.get(idx)
            .and_then(|b| b.as_thinking())
            .map(|t| t.thinking.clone())
            .unwrap_or_default();
        stream.push(AssistantMessageEvent::ThinkingEnd {
            content_index: idx,
            content,
            partial: output.clone(),
        });
    }
    if let Some(idx) = current_text_index {
        let content = output.content.get(idx)
            .and_then(|b| b.as_text())
            .map(|t| t.text.clone())
            .unwrap_or_default();
        stream.push(AssistantMessageEvent::TextEnd {
            content_index: idx,
            content,
            partial: output.clone(),
        });
    }

    stream.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output,
    });
    stream.end(None);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type() {
        let provider = GoogleProvider::new();
        assert_eq!(provider.provider_type(), Provider::Google);
    }

    #[test]
    fn test_convert_messages_basic() {
        let mut context = Context::new();
        context.add_message(Message::User(UserMessage::text("Hello")));

        let contents = convert_messages(&context);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, "user");
        assert_eq!(contents[0].parts.len(), 1);
    }

    #[test]
    fn test_generate_tool_call_id() {
        let id1 = generate_tool_call_id("test_tool");
        let id2 = generate_tool_call_id("test_tool");
        assert_ne!(id1, id2);
        assert!(id1.starts_with("test_tool_"));
    }
}

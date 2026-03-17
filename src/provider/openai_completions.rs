//! OpenAI Chat Completions API provider.

use crate::provider::LLMProvider;
use crate::types::{StreamOptions, SimpleStreamOptions};
use crate::stream::{AssistantMessageEventStream, parse_streaming_json};
use crate::thinking::OpenAIThinkingOptions;
use crate::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OpenAI Completions API provider.
pub struct OpenAICompletionsProvider {
    client: Client,
    default_api_key: Option<String>,
}

impl OpenAICompletionsProvider {
    /// Create a new OpenAI Completions provider.
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

    /// Get API key from options or environment.
    fn resolve_api_key(&self, options: &StreamOptions, provider: &Provider) -> String {
        // Priority: options.api_key > self.default_api_key > environment variable
        if let Some(ref key) = options.api_key {
            return key.clone();
        }
        if let Some(ref key) = self.default_api_key {
            return key.clone();
        }

        // Try environment variable based on provider
        let env_key = match provider {
            Provider::OpenAI => std::env::var("OPENAI_API_KEY").ok(),
            Provider::Groq => std::env::var("GROQ_API_KEY").ok(),
            Provider::XAI => std::env::var("XAI_API_KEY").ok(),
            Provider::Cerebras => std::env::var("CEREBRAS_API_KEY").ok(),
            Provider::OpenRouter => std::env::var("OPENROUTER_API_KEY").ok(),
            Provider::VercelAiGateway => std::env::var("AI_GATEWAY_API_KEY").ok(),
            Provider::Mistral => std::env::var("MISTRAL_API_KEY").ok(),
            Provider::ZAI => std::env::var("ZAI_API_KEY").ok(),
            Provider::Ollama => return String::new(), // Ollama doesn't need API key
            _ => None,
        };

        env_key.unwrap_or_default()
    }
}

impl Default for OpenAICompletionsProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for OpenAICompletionsProvider {
    fn api_type(&self) -> Api {
        Api::OpenAICompletions
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
        let api_key = self.resolve_api_key(&options, &model.provider);

        tokio::spawn(async move {
            if let Err(e) = run_stream(client, &model, &context, options, api_key, stream_clone).await {
                tracing::error!("Stream error: {}", e);
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
        let thinking_options = options.reasoning.map(OpenAIThinkingOptions::from_level);

        let stream_options = StreamOptions {
            temperature: options.base.temperature,
            max_tokens: options.base.max_tokens,
            api_key: options.base.api_key,
            base_url: options.base.base_url,
            headers: options.base.headers,
            session_id: options.base.session_id,
        };

        let stream = AssistantMessageEventStream::new_assistant_stream();
        let stream_clone = stream.clone();

        let model = model.clone();
        let context = context.clone();
        let client = self.client.clone();
        let api_key = self.resolve_api_key(&stream_options, &model.provider);

        tokio::spawn(async move {
            if let Err(e) = run_stream_with_thinking(
                client,
                &model,
                &context,
                stream_options,
                api_key,
                thinking_options,
                stream_clone,
            ).await {
                tracing::error!("Stream error: {}", e);
            }
        });

        stream
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// OpenAI Chat Completions request.
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptionsConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

#[derive(Debug, Serialize)]
struct StreamOptionsConfig {
    include_usage: bool,
}

/// OpenAI message format.
#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OpenAIContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum OpenAIContent {
    Text(String),
    Parts(Vec<OpenAIContentPart>),
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIContentPart {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<ImageUrl>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ImageUrl {
    url: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunctionDef,
}

#[derive(Debug, Serialize)]
struct OpenAIFunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// Streaming response chunk.
#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    usage: Option<ChunkUsage>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    #[allow(dead_code)]
    index: u32,
    delta: Option<ChunkDelta>,
    finish_reason: Option<String>,
    #[serde(default)]
    usage: Option<ChunkUsage>,
}

#[derive(Debug, Deserialize, Default)]
struct ChunkDelta {
    #[allow(dead_code)]
    role: Option<String>,
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_text: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ChunkToolCall>,
}

#[derive(Debug, Deserialize)]
struct ChunkToolCall {
    index: Option<u32>,
    id: Option<String>,
    function: Option<ChunkFunction>,
}

#[derive(Debug, Deserialize)]
struct ChunkFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChunkUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    #[allow(dead_code)]
    prompt_tokens_details: Option<PromptTokensDetails>,
    #[allow(dead_code)]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    #[allow(dead_code)]
    cached_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct CompletionTokensDetails {
    #[allow(dead_code)]
    reasoning_tokens: Option<u64>,
}

// ============================================================================
// Message Conversion
// ============================================================================

/// Convert context to OpenAI messages.
fn convert_messages(context: &Context, model: &Model) -> Vec<OpenAIMessage> {
    let mut messages = Vec::new();

    // Add system prompt
    if let Some(ref prompt) = context.system_prompt {
        let use_developer = model.reasoning
            && model.compat.as_ref().map_or(false, |c| c.supports_developer_role);
        let role = if use_developer { "developer" } else { "system" };

        messages.push(OpenAIMessage {
            role: role.to_string(),
            content: Some(OpenAIContent::Text(sanitize_surrogates(prompt))),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    // Convert messages
    for msg in &context.messages {
        match msg {
            Message::User(user_msg) => {
                let openai_msg = convert_user_message(user_msg, model);
                messages.push(openai_msg);
            }
            Message::Assistant(assistant_msg) => {
                let openai_msg = convert_assistant_message(assistant_msg, model);
                if let Some(msg) = openai_msg {
                    messages.push(msg);
                }
            }
            Message::ToolResult(tool_result) => {
                let openai_msg = convert_tool_result(tool_result, model);
                messages.push(openai_msg);
            }
        }
    }

    messages
}

fn convert_user_message(user_msg: &UserMessage, model: &Model) -> OpenAIMessage {
    match &user_msg.content {
        UserContent::Text(text) => OpenAIMessage {
            role: "user".to_string(),
            content: Some(OpenAIContent::Text(sanitize_surrogates(text))),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        UserContent::Blocks(blocks) => {
            let parts: Vec<OpenAIContentPart> = blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text(t) => Some(OpenAIContentPart {
                        content_type: "text".to_string(),
                        text: Some(sanitize_surrogates(&t.text)),
                        image_url: None,
                    }),
                    ContentBlock::Image(img) => {
                        if model.supports_image() {
                            Some(OpenAIContentPart {
                                content_type: "image_url".to_string(),
                                text: None,
                                image_url: Some(ImageUrl {
                                    url: format!("data:{};base64,{}", img.mime_type, img.data),
                                }),
                            })
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .collect();

            if parts.is_empty() {
                OpenAIMessage {
                    role: "user".to_string(),
                    content: Some(OpenAIContent::Text(String::new())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                }
            } else {
                OpenAIMessage {
                    role: "user".to_string(),
                    content: Some(OpenAIContent::Parts(parts)),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                }
            }
        }
    }
}

fn convert_assistant_message(assistant_msg: &AssistantMessage, _model: &Model) -> Option<OpenAIMessage> {
    // Skip error/aborted messages
    if assistant_msg.stop_reason == StopReason::Error || assistant_msg.stop_reason == StopReason::Aborted {
        return None;
    }

    // Get text content
    let text_content: String = assistant_msg
        .content
        .iter()
        .filter_map(|b| b.as_text())
        .filter(|t| !t.text.trim().is_empty())
        .map(|t| sanitize_surrogates(&t.text))
        .collect::<Vec<_>>()
        .join("");

    // Get tool calls
    let tool_calls: Vec<OpenAIToolCall> = assistant_msg
        .content
        .iter()
        .filter_map(|b| b.as_tool_call())
        .map(|tc| OpenAIToolCall {
            id: tc.id.clone(),
            call_type: "function".to_string(),
            function: OpenAIFunction {
                name: tc.name.clone(),
                arguments: serde_json::to_string(&tc.arguments).unwrap_or_default(),
            },
        })
        .collect();

    // Skip if no content and no tool calls
    if text_content.is_empty() && tool_calls.is_empty() {
        return None;
    }

    // Handle thinking blocks
    let thinking_blocks: Vec<_> = assistant_msg
        .content
        .iter()
        .filter_map(|b| b.as_thinking())
        .filter(|t| !t.thinking.trim().is_empty())
        .collect();

    let content = if text_content.is_empty() {
        None
    } else {
        Some(OpenAIContent::Text(text_content))
    };

    // Add thinking as a separate field if the model supports it
    let msg = OpenAIMessage {
        role: "assistant".to_string(),
        content,
        tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
        tool_call_id: None,
        name: None,
    };

    // For models that support reasoning_content field
    if !thinking_blocks.is_empty() {
        let thinking_text = thinking_blocks
            .iter()
            .map(|t| t.thinking.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        // This would need custom serialization to add the reasoning_content field
        // For now, we'll skip adding thinking blocks
        let _ = thinking_text;
    }

    Some(msg)
}

fn convert_tool_result(tool_result: &ToolResultMessage, model: &Model) -> OpenAIMessage {
    let text: String = tool_result
        .content
        .iter()
        .filter_map(|b| b.as_text())
        .map(|t| sanitize_surrogates(&t.text))
        .collect::<Vec<_>>()
        .join("\n");

    let requires_name = model.compat.as_ref().map_or(false, |c| c.requires_tool_result_name);

    OpenAIMessage {
        role: "tool".to_string(),
        content: Some(OpenAIContent::Text(if text.is_empty() {
            "(no output)".to_string()
        } else {
            text
        })),
        tool_calls: None,
        tool_call_id: Some(tool_result.tool_call_id.clone()),
        name: if requires_name { Some(tool_result.tool_name.clone()) } else { None },
    }
}

/// Convert tools to OpenAI format.
fn convert_tools(tools: &[Tool]) -> Vec<OpenAITool> {
    tools
        .iter()
        .map(|t| OpenAITool {
            tool_type: "function".to_string(),
            function: OpenAIFunctionDef {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            },
        })
        .collect()
}

/// Sanitize Unicode surrogates.
fn sanitize_surrogates(text: &str) -> String {
    text.replace(
        |c: char| {
            let cp = c as u32;
            (0xD800..=0xDFFF).contains(&cp)
        },
        "",
    )
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
        .api(model.api.clone())
        .provider(model.provider.clone())
        .model(model.id.clone())
        .stop_reason(StopReason::Stop)
        .usage(Usage::default())
        .build()?;

    let messages = convert_messages(context, model);
    let tools = context.tools.as_ref().map(|t| convert_tools(&t));

    // Determine which max tokens field to use
    let max_tokens_field = model.compat.as_ref().and_then(|c| c.max_tokens_field.as_deref());
    let (max_tokens, max_completion_tokens) = match max_tokens_field {
        Some("max_tokens") => (options.max_tokens, None),
        _ => (None, options.max_tokens),
    };

    let request = ChatCompletionRequest {
        model: model.id.clone(),
        messages,
        stream: true,
        temperature: options.temperature,
        max_tokens,
        max_completion_tokens,
        tools,
        stream_options: Some(StreamOptionsConfig { include_usage: true }),
        reasoning_effort: None,
    };

    let base = options.base_url.as_deref().unwrap_or(&model.base_url);
    let url = format!("{}/chat/completions", base);

    tracing::info!(
        url = %url,
        model = %model.id,
        provider = %model.provider,
        message_count = request.messages.len(),
        has_tools = request.tools.is_some(),
        "Sending OpenAI Completions request"
    );
    tracing::debug!(request_body = %serde_json::to_string(&request).unwrap_or_default(), "Request payload");

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::AUTHORIZATION,
        format!("Bearer {}", api_key).parse()?,
    );
    headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse()?);

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
            url = %url,
            model = %model.id,
            status = %status,
            response_body = %body,
            "OpenAI Completions request failed"
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

    let mut current_block: Option<ContentBlock> = None;
    let mut partial_tool_args: HashMap<u32, String> = HashMap::new();
    let mut current_tool_index: Option<u32> = None;
    let mut line_buffer = String::new(); // Buffer for incomplete SSE lines

    let mut byte_stream = response.bytes_stream();
    while let Some(chunk_result) = byte_stream.next().await {
        let chunk = chunk_result?;
        let text = String::from_utf8_lossy(&chunk);
        line_buffer.push_str(&text);

        // Process only complete lines (ending with \n), keep partial line in buffer
        while let Some(newline_pos) = line_buffer.find('\n') {
            let line = line_buffer[..newline_pos].trim_end_matches('\r').to_string();
            line_buffer = line_buffer[newline_pos + 1..].to_string();

            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];
            if data == "[DONE]" {
                continue;
            }

            let parsed: Result<ChatCompletionChunk, _> = serde_json::from_str(data);
            if let Ok(chunk_data) = parsed {
                // Handle usage
                if let Some(usage) = &chunk_data.usage {
                    output.usage.input = usage.prompt_tokens.unwrap_or(0);
                    output.usage.output = usage.completion_tokens.unwrap_or(0);
                    output.usage.total_tokens = output.usage.input + output.usage.output;
                }

                for choice in &chunk_data.choices {
                    // Handle finish reason
                    if let Some(ref reason) = choice.finish_reason {
                        output.stop_reason = match reason.as_str() {
                            "stop" | "end" => StopReason::Stop,
                            "length" => StopReason::Length,
                            "tool_calls" | "function_call" => StopReason::ToolUse,
                            "content_filter" => StopReason::Error,
                            _ => StopReason::Stop,
                        };
                    }

                    // Handle usage in choice (fallback for some providers)
                    if let Some(usage) = &choice.usage {
                        output.usage.input = usage.prompt_tokens.unwrap_or(0);
                        output.usage.output = usage.completion_tokens.unwrap_or(0);
                        output.usage.total_tokens = output.usage.input + output.usage.output;
                    }

                    if let Some(ref delta) = choice.delta {
                        // Handle text content
                        if let Some(ref content) = delta.content {
                            if !content.is_empty() {
                                if current_block.as_ref().map_or(true, |b| !b.is_text()) {
                                    if let Some(block) = current_block.take() {
                                        output.content.push(block);
                                    }
                                    current_block = Some(ContentBlock::Text(TextContent::new("")));
                                    stream.push(AssistantMessageEvent::TextStart {
                                        content_index: output.content.len(),
                                        partial: output.clone(),
                                    });
                                }

                                if let Some(ContentBlock::Text(ref mut text_block)) = current_block {
                                    text_block.text.push_str(content);
                                    stream.push(AssistantMessageEvent::TextDelta {
                                        content_index: output.content.len(),
                                        delta: content.clone(),
                                        partial: output.clone(),
                                    });
                                }
                            }
                        }

                        // Handle reasoning/thinking
                        let reasoning = delta.reasoning_content.as_ref()
                            .or(delta.reasoning.as_ref())
                            .or(delta.reasoning_text.as_ref());

                        if let Some(content) = reasoning {
                            if !content.is_empty() {
                                if current_block.as_ref().map_or(true, |b| !b.is_thinking()) {
                                    if let Some(block) = current_block.take() {
                                        output.content.push(block);
                                    }
                                    current_block = Some(ContentBlock::Thinking(ThinkingContent::new("")));
                                    stream.push(AssistantMessageEvent::ThinkingStart {
                                        content_index: output.content.len(),
                                        partial: output.clone(),
                                    });
                                }

                                if let Some(ContentBlock::Thinking(ref mut thinking_block)) = current_block {
                                    thinking_block.thinking.push_str(content);
                                    stream.push(AssistantMessageEvent::ThinkingDelta {
                                        content_index: output.content.len(),
                                        delta: content.clone(),
                                        partial: output.clone(),
                                    });
                                }
                            }
                        }

                        // Handle tool calls
                        for tc in &delta.tool_calls {
                            let index = tc.index.unwrap_or(0);

                            let is_new = current_tool_index.map_or(true, |i| i != index)
                                || current_block.as_ref().map_or(true, |b| !b.is_tool_call());

                            if is_new {
                                // Finish previous block
                                if let Some(block) = current_block.take() {
                                    output.content.push(block);
                                }

                                let id = tc.id.clone().unwrap_or_default();
                                let name = tc.function.as_ref().and_then(|f| f.name.clone()).unwrap_or_default();

                                current_block = Some(ContentBlock::ToolCall(ToolCall::new(
                                    id,
                                    name,
                                    serde_json::Value::Object(serde_json::Map::new()),
                                )));
                                current_tool_index = Some(index);

                                partial_tool_args.insert(index, String::new());

                                stream.push(AssistantMessageEvent::ToolCallStart {
                                    content_index: output.content.len(),
                                    partial: output.clone(),
                                });
                            }

                            if let Some(ContentBlock::ToolCall(ref mut tool_call)) = current_block {
                                if let Some(ref id) = tc.id {
                                    tool_call.id = id.clone();
                                }
                                if let Some(ref func) = tc.function {
                                    if let Some(ref name) = func.name {
                                        tool_call.name = name.clone();
                                    }
                                    if let Some(ref args) = func.arguments {
                                        let partial = partial_tool_args.entry(index).or_insert_with(String::new);
                                        partial.push_str(args);
                                        tool_call.arguments = parse_streaming_json(partial);

                                        stream.push(AssistantMessageEvent::ToolCallDelta {
                                            content_index: output.content.len(),
                                            delta: args.clone(),
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

    // Finish current block
    if let Some(block) = current_block.take() {
        output.content.push(block);
    }

    stream.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output,
    });
    stream.end(None);

    Ok(())
}

async fn run_stream_with_thinking(
    client: Client,
    model: &Model,
    context: &Context,
    options: StreamOptions,
    api_key: String,
    thinking_options: Option<OpenAIThinkingOptions>,
    stream: AssistantMessageEventStream,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // For now, just run without thinking options
    // TODO: Add reasoning_effort to request when model supports it
    let _ = thinking_options;
    run_stream(client, model, context, options, api_key, stream).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_messages_basic() {
        let mut context = Context::with_system_prompt("You are helpful.");
        context.add_message(Message::User(UserMessage::text("Hello")));

        let model = Model::builder()
            .id("gpt-4o-mini")
            .name("GPT-4o Mini")
            .api(Api::OpenAICompletions)
            .provider(Provider::OpenAI)
            .base_url("https://api.openai.com/v1")
            .context_window(128000)
            .max_tokens(16384)
            .build()
            .unwrap();

        let messages = convert_messages(&context, &model);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
    }
}

//! OpenAI Responses API provider (new API for o1, o3, gpt-5 models).
//!
//! Implements streaming via typed SSE events:
//! response.output_item.added → response.output_text.delta / response.function_call_arguments.delta
//! → response.output_item.done → response.completed

use crate::provider::LLMProvider;
use crate::stream::{AssistantMessageEventStream, parse_streaming_json};
use crate::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OpenAI Responses API provider.
pub struct OpenAIResponsesProvider {
    client: Client,
    default_api_key: Option<String>,
}

impl OpenAIResponsesProvider {
    /// Create a new OpenAI Responses provider.
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
        std::env::var("OPENAI_API_KEY").unwrap_or_default()
    }
}

impl Default for OpenAIResponsesProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for OpenAIResponsesProvider {
    fn provider_type(&self) -> Provider {
        Provider::OpenAI
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
                tracing::error!("OpenAI Responses stream error: {}", e);
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
// Request Types
// ============================================================================

/// OpenAI Responses API request.
#[derive(Debug, Serialize)]
struct ResponsesRequest {
    model: String,
    input: Vec<ResponsesInputItem>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ResponsesTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ResponsesReasoning>,
}

/// Flat typed input item for Responses API.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum ResponsesInputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: ResponsesContent,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum ResponsesContent {
    Text(String),
    Parts(Vec<ResponsesContentPart>),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum ResponsesContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_image")]
    InputImage {
        image_url: String,
    },
    #[serde(rename = "output_text")]
    OutputText { text: String },
}

#[derive(Debug, Serialize)]
struct ResponsesTool {
    #[serde(rename = "type")]
    tool_type: String,
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct ResponsesReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
}

// ============================================================================
// SSE Event Types
// ============================================================================

/// Parsed SSE event data from OpenAI Responses API.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ResponseEvent {
    #[serde(flatten)]
    data: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OutputItem {
    #[serde(rename = "type")]
    item_type: Option<String>,
    id: Option<String>,
    // For function_call items
    call_id: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseCompleted {
    response: Option<ResponseData>,
}

#[derive(Debug, Deserialize)]
struct ResponseData {
    status: Option<String>,
    usage: Option<ResponseUsage>,
    #[allow(dead_code)]
    output: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct ResponseUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    input_tokens_details: Option<InputTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct InputTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

// ============================================================================
// Message Conversion
// ============================================================================

fn convert_messages(context: &Context) -> Vec<ResponsesInputItem> {
    let mut items = Vec::new();

    for msg in &context.messages {
        match msg {
            Message::User(user_msg) => {
                let content = match &user_msg.content {
                    UserContent::Text(text) => ResponsesContent::Text(text.clone()),
                    UserContent::Blocks(blocks) => {
                        let parts: Vec<ResponsesContentPart> = blocks
                            .iter()
                            .filter_map(|b| match b {
                                ContentBlock::Text(t) => Some(ResponsesContentPart::InputText {
                                    text: t.text.clone(),
                                }),
                                ContentBlock::Image(img) => Some(ResponsesContentPart::InputImage {
                                    image_url: format!("data:{};base64,{}", img.mime_type, img.data),
                                }),
                                _ => None,
                            })
                            .collect();
                        ResponsesContent::Parts(parts)
                    }
                };
                items.push(ResponsesInputItem::Message {
                    role: "user".to_string(),
                    content,
                });
            }
            Message::Assistant(assistant_msg) => {
                if assistant_msg.stop_reason == StopReason::Error
                    || assistant_msg.stop_reason == StopReason::Aborted
                {
                    continue;
                }

                // Collect text content
                let text_content: String = assistant_msg
                    .content
                    .iter()
                    .filter_map(|b| b.as_text())
                    .filter(|t| !t.text.trim().is_empty())
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join("");

                if !text_content.is_empty() {
                    items.push(ResponsesInputItem::Message {
                        role: "assistant".to_string(),
                        content: ResponsesContent::Parts(vec![
                            ResponsesContentPart::OutputText {
                                text: text_content,
                            },
                        ]),
                    });
                }

                // Add function calls as separate items
                for block in &assistant_msg.content {
                    if let ContentBlock::ToolCall(tc) = block {
                        // Parse composite ID if present: "{call_id}|{item_id}"
                        let (call_id, item_id) = if tc.id.contains('|') {
                            let parts: Vec<&str> = tc.id.splitn(2, '|').collect();
                            (parts[0].to_string(), parts.get(1).unwrap_or(&"").to_string())
                        } else {
                            (tc.id.clone(), format!("fc_{}", tc.id))
                        };

                        items.push(ResponsesInputItem::FunctionCall {
                            id: item_id,
                            call_id,
                            name: tc.name.clone(),
                            arguments: serde_json::to_string(&tc.arguments).unwrap_or_default(),
                        });
                    }
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

                // Parse composite ID
                let call_id = if tool_result.tool_call_id.contains('|') {
                    tool_result.tool_call_id.splitn(2, '|').next().unwrap_or(&tool_result.tool_call_id).to_string()
                } else {
                    tool_result.tool_call_id.clone()
                };

                items.push(ResponsesInputItem::FunctionCallOutput {
                    call_id,
                    output: if text.is_empty() { "(no output)".to_string() } else { text },
                });
            }
        }
    }

    items
}

fn convert_tools(tools: &[Tool]) -> Vec<ResponsesTool> {
    tools
        .iter()
        .map(|t| ResponsesTool {
            tool_type: "function".to_string(),
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.parameters.clone(),
        })
        .collect()
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
        .api(Api::OpenAIResponses)
        .provider(model.provider.clone())
        .model(model.id.clone())
        .stop_reason(StopReason::Stop)
        .usage(Usage::default())
        .build()?;

    let input = convert_messages(context);
    let tools = context.tools.as_ref().map(|t| convert_tools(t));

    let request = ResponsesRequest {
        model: model.id.clone(),
        input,
        stream: true,
        instructions: context.system_prompt.clone(),
        temperature: options.temperature,
        max_output_tokens: options.max_tokens,
        tools,
        reasoning: None,
    };

    let base = options.base_url.as_deref().unwrap_or(&model.base_url);
    let url = format!("{}/responses", base);

    tracing::info!(
        url = %url,
        model = %model.id,
        provider = %model.provider,
        input_count = request.input.len(),
        has_tools = request.tools.is_some(),
        "Sending OpenAI Responses request"
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
            "OpenAI Responses request failed"
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

    // Track output items by their index
    let mut item_content_map: HashMap<usize, ItemInfo> = HashMap::new();
    let mut partial_tool_args: HashMap<usize, String> = HashMap::new();
    let mut line_buffer = String::new();
    let mut current_event_type = String::new();
    let mut item_counter: usize = 0;

    let mut byte_stream = response.bytes_stream();
    while let Some(chunk_result) = byte_stream.next().await {
        let chunk = chunk_result?;
        let text = String::from_utf8_lossy(&chunk);
        line_buffer.push_str(&text);

        while let Some(newline_pos) = line_buffer.find('\n') {
            let line = line_buffer[..newline_pos].trim_end_matches('\r').to_string();
            line_buffer = line_buffer[newline_pos + 1..].to_string();

            if line.starts_with("event: ") {
                current_event_type = line[7..].to_string();
                continue;
            }

            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];
            if data == "[DONE]" {
                continue;
            }

            match current_event_type.as_str() {
                "response.output_item.added" => {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                        let item = val.get("item");
                        let item_type = item
                            .and_then(|i| i.get("type"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        let item_id = item
                            .and_then(|i| i.get("id"))
                            .and_then(|i| i.as_str())
                            .unwrap_or("")
                            .to_string();

                        let output_index = val.get("output_index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(item_counter as u64) as usize;

                        match item_type {
                            "message" => {
                                let content_idx = output.content.len();
                                output.content.push(ContentBlock::Text(TextContent::new("")));
                                item_content_map.insert(output_index, ItemInfo {
                                    content_idx,
                                    item_type: ItemType::Message,
                                    item_id,
                                    call_id: None,
                                    name: None,
                                });
                                stream.push(AssistantMessageEvent::TextStart {
                                    content_index: content_idx,
                                    partial: output.clone(),
                                });
                            }
                            "function_call" => {
                                let call_id = item
                                    .and_then(|i| i.get("call_id"))
                                    .and_then(|c| c.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let name = item
                                    .and_then(|i| i.get("name"))
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                // Composite ID: "{call_id}|{item_id}"
                                let composite_id = format!("{}|{}", call_id, item_id);
                                let content_idx = output.content.len();
                                output.content.push(ContentBlock::ToolCall(ToolCall::new(
                                    &composite_id,
                                    &name,
                                    serde_json::Value::Object(serde_json::Map::new()),
                                )));
                                partial_tool_args.insert(output_index, String::new());
                                item_content_map.insert(output_index, ItemInfo {
                                    content_idx,
                                    item_type: ItemType::FunctionCall,
                                    item_id,
                                    call_id: Some(call_id),
                                    name: Some(name),
                                });
                                stream.push(AssistantMessageEvent::ToolCallStart {
                                    content_index: content_idx,
                                    partial: output.clone(),
                                });
                            }
                            "reasoning" => {
                                let content_idx = output.content.len();
                                output.content.push(ContentBlock::Thinking(ThinkingContent::new("")));
                                item_content_map.insert(output_index, ItemInfo {
                                    content_idx,
                                    item_type: ItemType::Reasoning,
                                    item_id,
                                    call_id: None,
                                    name: None,
                                });
                                stream.push(AssistantMessageEvent::ThinkingStart {
                                    content_index: content_idx,
                                    partial: output.clone(),
                                });
                            }
                            _ => {}
                        }
                        item_counter += 1;
                    }
                }

                "response.output_text.delta" => {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                        let output_index = val.get("output_index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(0) as usize;
                        let delta = val.get("delta")
                            .and_then(|d| d.as_str())
                            .unwrap_or("");

                        if let Some(info) = item_content_map.get(&output_index) {
                            let idx = info.content_idx;
                            if let Some(ContentBlock::Text(ref mut t)) = output.content.get_mut(idx) {
                                t.text.push_str(delta);
                            }
                            stream.push(AssistantMessageEvent::TextDelta {
                                content_index: idx,
                                delta: delta.to_string(),
                                partial: output.clone(),
                            });
                        }
                    }
                }

                "response.function_call_arguments.delta" => {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                        let output_index = val.get("output_index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(0) as usize;
                        let delta = val.get("delta")
                            .and_then(|d| d.as_str())
                            .unwrap_or("");

                        if let Some(info) = item_content_map.get(&output_index) {
                            let idx = info.content_idx;
                            if let Some(ref mut args_str) = partial_tool_args.get_mut(&output_index) {
                                args_str.push_str(delta);
                                let parsed = parse_streaming_json(args_str);
                                if let Some(ContentBlock::ToolCall(ref mut tc)) = output.content.get_mut(idx) {
                                    tc.arguments = parsed;
                                }
                            }
                            stream.push(AssistantMessageEvent::ToolCallDelta {
                                content_index: idx,
                                delta: delta.to_string(),
                                partial: output.clone(),
                            });
                        }
                    }
                }

                "response.reasoning_summary_text.delta" => {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                        let output_index = val.get("output_index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(0) as usize;
                        let delta = val.get("delta")
                            .and_then(|d| d.as_str())
                            .unwrap_or("");

                        if let Some(info) = item_content_map.get(&output_index) {
                            if info.item_type == ItemType::Reasoning {
                                let idx = info.content_idx;
                                if let Some(ContentBlock::Thinking(ref mut t)) = output.content.get_mut(idx) {
                                    t.thinking.push_str(delta);
                                }
                                stream.push(AssistantMessageEvent::ThinkingDelta {
                                    content_index: idx,
                                    delta: delta.to_string(),
                                    partial: output.clone(),
                                });
                            }
                        }
                    }
                }

                "response.output_item.done" => {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                        let output_index = val.get("output_index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(0) as usize;

                        if let Some(info) = item_content_map.get(&output_index) {
                            let idx = info.content_idx;
                            match info.item_type {
                                ItemType::Message => {
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
                                ItemType::FunctionCall => {
                                    // Finalize tool call args
                                    if let Some(args_str) = partial_tool_args.get(&output_index) {
                                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(args_str) {
                                            if let Some(ContentBlock::ToolCall(ref mut tc)) = output.content.get_mut(idx) {
                                                tc.arguments = parsed;
                                            }
                                        }
                                    }
                                    let tool_call = output.content.get(idx)
                                        .and_then(|b| b.as_tool_call())
                                        .cloned()
                                        .unwrap_or_else(|| ToolCall::new("", "", serde_json::Value::Null));
                                    stream.push(AssistantMessageEvent::ToolCallEnd {
                                        content_index: idx,
                                        tool_call,
                                        partial: output.clone(),
                                    });
                                }
                                ItemType::Reasoning => {
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
                            }
                        }
                    }
                }

                "response.completed" => {
                    if let Ok(completed) = serde_json::from_str::<ResponseCompleted>(data) {
                        if let Some(ref resp) = completed.response {
                            // Update usage
                            if let Some(ref usage) = resp.usage {
                                output.usage.input = usage.input_tokens;
                                output.usage.output = usage.output_tokens;
                                if let Some(ref details) = usage.input_tokens_details {
                                    output.usage.cache_read = details.cached_tokens;
                                }
                                output.usage.total_tokens = output.usage.input + output.usage.output;
                            }

                            // Update stop reason from status
                            if let Some(ref status) = resp.status {
                                output.stop_reason = match status.as_str() {
                                    "completed" => {
                                        if output.has_tool_calls() {
                                            StopReason::ToolUse
                                        } else {
                                            StopReason::Stop
                                        }
                                    }
                                    "incomplete" => StopReason::Length,
                                    "failed" | "cancelled" => StopReason::Error,
                                    _ => StopReason::Stop,
                                };
                            }
                        }
                    }
                }

                "error" => {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                        let error_msg = val.get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                            .or_else(|| val.get("message").and_then(|m| m.as_str()))
                            .unwrap_or("Unknown OpenAI error");
                        output.stop_reason = StopReason::Error;
                        output.error_message = Some(error_msg.to_string());
                        stream.push(AssistantMessageEvent::Error {
                            reason: StopReason::Error,
                            error: output,
                        });
                        stream.end(None);
                        return Ok(());
                    }
                }

                _ => {
                    // Ignore other events like response.created, etc.
                }
            }
        }
    }

    stream.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output,
    });
    stream.end(None);

    Ok(())
}

/// Track information about output items.
#[derive(Debug, Clone)]
struct ItemInfo {
    content_idx: usize,
    item_type: ItemType,
    #[allow(dead_code)]
    item_id: String,
    #[allow(dead_code)]
    call_id: Option<String>,
    #[allow(dead_code)]
    name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ItemType {
    Message,
    FunctionCall,
    Reasoning,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type() {
        let provider = OpenAIResponsesProvider::new();
        assert_eq!(provider.provider_type(), Provider::OpenAI);
    }

    #[test]
    fn test_convert_messages_basic() {
        let mut context = Context::with_system_prompt("You are helpful.");
        context.add_message(Message::User(UserMessage::text("Hello")));

        let items = convert_messages(&context);
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn test_convert_tool_call_composite_id() {
        let mut context = Context::new();
        context.add_message(Message::User(UserMessage::text("Hello")));

        // Create an assistant message with a tool call using composite ID
        let msg = AssistantMessage::builder()
            .api(Api::OpenAIResponses)
            .provider(Provider::OpenAI)
            .model("gpt-4o")
            .content(vec![ContentBlock::ToolCall(ToolCall::new(
                "call_abc|item_123",
                "get_weather",
                serde_json::json!({"city": "Tokyo"}),
            ))])
            .stop_reason(StopReason::ToolUse)
            .build()
            .unwrap();
        context.add_message(Message::Assistant(msg));

        // Add tool result
        context.add_message(Message::ToolResult(ToolResultMessage::text(
            "call_abc|item_123", "get_weather", "Sunny 25°C", false,
        )));

        let items = convert_messages(&context);
        assert_eq!(items.len(), 3); // user + function_call + function_call_output
    }
}

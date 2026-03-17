//! Anthropic Messages API provider.
//!
//! Implements streaming via Anthropic's SSE protocol with events:
//! message_start → content_block_start → content_block_delta → content_block_stop → message_delta → message_stop

/// Default base URL for Anthropic Messages API.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

use crate::provider::LLMProvider;
use crate::stream::{AssistantMessageEventStream, parse_streaming_json};
use crate::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Anthropic Messages API provider.
pub struct AnthropicProvider {
    client: Client,
    default_api_key: Option<String>,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider.
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
        std::env::var("ANTHROPIC_API_KEY").unwrap_or_default()
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    fn provider_type(&self) -> Provider {
        Provider::Anthropic
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
                tracing::error!("Anthropic stream error: {}", e);
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

/// Anthropic Messages request.
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinkingParam>,
}

/// Anthropic thinking parameter for the request.
#[derive(Debug, Serialize)]
#[serde(untagged)]
#[allow(dead_code)]
enum AnthropicThinkingParam {
    Budget {
        #[serde(rename = "type")]
        param_type: String,
        budget_tokens: u32,
    },
}

/// Anthropic message format.
#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

/// Anthropic content can be text or an array of content blocks.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

/// Anthropic content block in a message.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        source: AnthropicImageSource,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        data: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

/// Anthropic tool definition.
#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

// ============================================================================
// SSE Event Types
// ============================================================================

/// SSE event from Anthropic's streaming API.
#[derive(Debug, Deserialize)]
struct MessageStartData {
    message: MessageStartMessage,
}

#[derive(Debug, Deserialize)]
struct MessageStartMessage {
    #[allow(dead_code)]
    id: String,
    model: String,
    usage: Option<MessageUsage>,
}

#[derive(Debug, Deserialize)]
struct MessageUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct ContentBlockStartData {
    index: usize,
    content_block: ContentBlockInfo,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentBlockInfo {
    #[serde(rename = "text")]
    Text {
        #[allow(dead_code)]
        text: String,
    },
    #[serde(rename = "thinking")]
    Thinking {
        #[allow(dead_code)]
        thinking: String,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {},
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
    },
}

#[derive(Debug, Deserialize)]
struct ContentBlockDeltaData {
    index: usize,
    delta: DeltaInfo,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum DeltaInfo {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
}

#[derive(Debug, Deserialize)]
struct ContentBlockStopData {
    index: usize,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaData {
    delta: MessageDelta,
    usage: Option<MessageDeltaUsage>,
}

#[derive(Debug, Deserialize)]
struct MessageDelta {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaUsage {
    #[serde(default)]
    output_tokens: u64,
}

// ============================================================================
// Message Conversion
// ============================================================================

/// Convert context to Anthropic messages.
fn convert_messages(context: &Context) -> Vec<AnthropicMessage> {
    let mut messages = Vec::new();

    for msg in &context.messages {
        match msg {
            Message::User(user_msg) => {
                let content = match &user_msg.content {
                    UserContent::Text(text) => AnthropicContent::Text(text.clone()),
                    UserContent::Blocks(blocks) => {
                        let parts: Vec<AnthropicContentBlock> = blocks
                            .iter()
                            .filter_map(|b| match b {
                                ContentBlock::Text(t) => Some(AnthropicContentBlock::Text {
                                    text: t.text.clone(),
                                }),
                                ContentBlock::Image(img) => Some(AnthropicContentBlock::Image {
                                    source: AnthropicImageSource {
                                        source_type: "base64".to_string(),
                                        media_type: img.mime_type.clone(),
                                        data: img.data.clone(),
                                    },
                                }),
                                _ => None,
                            })
                            .collect();
                        AnthropicContent::Blocks(parts)
                    }
                };
                messages.push(AnthropicMessage {
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

                let mut blocks = Vec::new();

                for block in &assistant_msg.content {
                    match block {
                        ContentBlock::Text(t) => {
                            if !t.text.trim().is_empty() {
                                blocks.push(AnthropicContentBlock::Text {
                                    text: t.text.clone(),
                                });
                            }
                        }
                        ContentBlock::Thinking(t) => {
                            if t.redacted {
                                blocks.push(AnthropicContentBlock::RedactedThinking {
                                    data: t.thinking.clone(),
                                });
                            } else if !t.thinking.trim().is_empty() {
                                blocks.push(AnthropicContentBlock::Thinking {
                                    thinking: t.thinking.clone(),
                                    signature: t.thinking_signature.clone(),
                                });
                            }
                        }
                        ContentBlock::ToolCall(tc) => {
                            blocks.push(AnthropicContentBlock::ToolUse {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                                input: tc.arguments.clone(),
                            });
                        }
                        _ => {}
                    }
                }

                if !blocks.is_empty() {
                    messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: AnthropicContent::Blocks(blocks),
                    });
                }
            }
            Message::ToolResult(tool_result) => {
                // Anthropic: tool results go in a user message with tool_result blocks
                let text: String = tool_result
                    .content
                    .iter()
                    .filter_map(|b| b.as_text())
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");

                let block = AnthropicContentBlock::ToolResult {
                    tool_use_id: tool_result.tool_call_id.clone(),
                    content: if text.is_empty() { None } else { Some(text) },
                    is_error: if tool_result.is_error { Some(true) } else { None },
                };

                // Check if the last message is a user message; merge if so
                if let Some(last) = messages.last_mut() {
                    if last.role == "user" {
                        match &mut last.content {
                            AnthropicContent::Blocks(ref mut blocks) => {
                                blocks.push(block);
                                continue;
                            }
                            AnthropicContent::Text(text) => {
                                let text_block = AnthropicContentBlock::Text {
                                    text: text.clone(),
                                };
                                last.content = AnthropicContent::Blocks(vec![text_block, block]);
                                continue;
                            }
                        }
                    }
                }

                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: AnthropicContent::Blocks(vec![block]),
                });
            }
        }
    }

    messages
}

/// Convert tools to Anthropic format.
fn convert_tools(tools: &[Tool]) -> Vec<AnthropicTool> {
    tools
        .iter()
        .map(|t| AnthropicTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.parameters.clone(),
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
        .api(Api::AnthropicMessages)
        .provider(model.provider.clone())
        .model(model.id.clone())
        .stop_reason(StopReason::Stop)
        .usage(Usage::default())
        .build()?;

    let messages = convert_messages(context);
    let tools = context.tools.as_ref().map(|t| convert_tools(t));

    let request = AnthropicRequest {
        model: model.id.clone(),
        messages,
        system: context.system_prompt.clone(),
        max_tokens: options.max_tokens.unwrap_or(model.max_tokens),
        stream: true,
        temperature: options.temperature,
        tools,
        thinking: None, // Thinking params can be set via SimpleStreamOptions
    };

    let base = options.base_url.as_deref()
        .or(model.base_url.as_deref())
        .unwrap_or(DEFAULT_BASE_URL);
    let url = format!("{}/messages", base);

    tracing::info!(
        url = %url,
        model = %model.id,
        provider = %model.provider,
        message_count = request.messages.len(),
        has_tools = request.tools.is_some(),
        "Sending Anthropic Messages request"
    );
    tracing::debug!(request_body = %serde_json::to_string(&request).unwrap_or_default(), "Request payload");

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("x-api-key", api_key.parse()?);
    headers.insert("anthropic-version", "2023-06-01".parse()?);
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
            "Anthropic Messages request failed"
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

    // Track content blocks by index
    let mut block_types: Vec<BlockType> = Vec::new();
    let mut partial_tool_args: std::collections::HashMap<usize, String> = std::collections::HashMap::new();
    let mut line_buffer = String::new();
    let mut current_event_type = String::new();

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
                "message_start" => {
                    if let Ok(msg_start) = serde_json::from_str::<MessageStartData>(data) {
                        output.model = msg_start.message.model;
                        if let Some(usage) = msg_start.message.usage {
                            output.usage.input = usage.input_tokens;
                            output.usage.output = usage.output_tokens;
                            output.usage.cache_read = usage.cache_read_input_tokens;
                            output.usage.cache_write = usage.cache_creation_input_tokens;
                            output.usage.total_tokens = output.usage.input + output.usage.output
                                + output.usage.cache_read + output.usage.cache_write;
                        }
                    }
                }
                "content_block_start" => {
                    if let Ok(block_start) = serde_json::from_str::<ContentBlockStartData>(data) {
                        let idx = block_start.index;
                        match block_start.content_block {
                            ContentBlockInfo::Text { .. } => {
                                while block_types.len() <= idx {
                                    block_types.push(BlockType::Unknown);
                                }
                                block_types[idx] = BlockType::Text;
                                output.content.push(ContentBlock::Text(TextContent::new("")));
                                stream.push(AssistantMessageEvent::TextStart {
                                    content_index: idx,
                                    partial: output.clone(),
                                });
                            }
                            ContentBlockInfo::Thinking { .. } => {
                                while block_types.len() <= idx {
                                    block_types.push(BlockType::Unknown);
                                }
                                block_types[idx] = BlockType::Thinking;
                                output.content.push(ContentBlock::Thinking(ThinkingContent::new("")));
                                stream.push(AssistantMessageEvent::ThinkingStart {
                                    content_index: idx,
                                    partial: output.clone(),
                                });
                            }
                            ContentBlockInfo::RedactedThinking { .. } => {
                                while block_types.len() <= idx {
                                    block_types.push(BlockType::Unknown);
                                }
                                block_types[idx] = BlockType::RedactedThinking;
                                let mut thinking = ThinkingContent::new("");
                                thinking.redacted = true;
                                output.content.push(ContentBlock::Thinking(thinking));
                            }
                            ContentBlockInfo::ToolUse { id, name } => {
                                while block_types.len() <= idx {
                                    block_types.push(BlockType::Unknown);
                                }
                                block_types[idx] = BlockType::ToolUse;
                                partial_tool_args.insert(idx, String::new());
                                output.content.push(ContentBlock::ToolCall(ToolCall::new(
                                    id,
                                    name,
                                    serde_json::Value::Object(serde_json::Map::new()),
                                )));
                                stream.push(AssistantMessageEvent::ToolCallStart {
                                    content_index: idx,
                                    partial: output.clone(),
                                });
                            }
                        }
                    }
                }
                "content_block_delta" => {
                    if let Ok(delta_data) = serde_json::from_str::<ContentBlockDeltaData>(data) {
                        let idx = delta_data.index;
                        match delta_data.delta {
                            DeltaInfo::TextDelta { text } => {
                                if let Some(ContentBlock::Text(ref mut t)) = output.content.get_mut(idx) {
                                    t.text.push_str(&text);
                                }
                                stream.push(AssistantMessageEvent::TextDelta {
                                    content_index: idx,
                                    delta: text,
                                    partial: output.clone(),
                                });
                            }
                            DeltaInfo::ThinkingDelta { thinking } => {
                                if let Some(ContentBlock::Thinking(ref mut t)) = output.content.get_mut(idx) {
                                    t.thinking.push_str(&thinking);
                                }
                                stream.push(AssistantMessageEvent::ThinkingDelta {
                                    content_index: idx,
                                    delta: thinking,
                                    partial: output.clone(),
                                });
                            }
                            DeltaInfo::InputJsonDelta { partial_json } => {
                                if let Some(ref mut args_str) = partial_tool_args.get_mut(&idx) {
                                    args_str.push_str(&partial_json);
                                    let parsed = parse_streaming_json(args_str);
                                    if let Some(ContentBlock::ToolCall(ref mut tc)) = output.content.get_mut(idx) {
                                        tc.arguments = parsed;
                                    }
                                }
                                stream.push(AssistantMessageEvent::ToolCallDelta {
                                    content_index: idx,
                                    delta: partial_json,
                                    partial: output.clone(),
                                });
                            }
                            DeltaInfo::SignatureDelta { signature } => {
                                // Store signature on thinking blocks
                                if let Some(ContentBlock::Thinking(ref mut t)) = output.content.get_mut(idx) {
                                    let existing = t.thinking_signature.get_or_insert_with(String::new);
                                    existing.push_str(&signature);
                                }
                            }
                        }
                    }
                }
                "content_block_stop" => {
                    if let Ok(stop_data) = serde_json::from_str::<ContentBlockStopData>(data) {
                        let idx = stop_data.index;
                        if let Some(block_type) = block_types.get(idx) {
                            match block_type {
                                BlockType::Text => {
                                    let text = output.content.get(idx)
                                        .and_then(|b| b.as_text())
                                        .map(|t| t.text.clone())
                                        .unwrap_or_default();
                                    stream.push(AssistantMessageEvent::TextEnd {
                                        content_index: idx,
                                        content: text,
                                        partial: output.clone(),
                                    });
                                }
                                BlockType::Thinking => {
                                    let text = output.content.get(idx)
                                        .and_then(|b| b.as_thinking())
                                        .map(|t| t.thinking.clone())
                                        .unwrap_or_default();
                                    stream.push(AssistantMessageEvent::ThinkingEnd {
                                        content_index: idx,
                                        content: text,
                                        partial: output.clone(),
                                    });
                                }
                                BlockType::ToolUse => {
                                    // Finalize tool call args from the accumulated partial JSON
                                    if let Some(args_str) = partial_tool_args.get(&idx) {
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
                                _ => {}
                            }
                        }
                    }
                }
                "message_delta" => {
                    if let Ok(delta_data) = serde_json::from_str::<MessageDeltaData>(data) {
                        if let Some(ref reason) = delta_data.delta.stop_reason {
                            output.stop_reason = match reason.as_str() {
                                "end_turn" => StopReason::Stop,
                                "max_tokens" => StopReason::Length,
                                "tool_use" => StopReason::ToolUse,
                                "stop_sequence" => StopReason::Stop,
                                _ => StopReason::Stop,
                            };
                        }
                        if let Some(usage) = delta_data.usage {
                            output.usage.output = usage.output_tokens;
                            output.usage.total_tokens = output.usage.input + output.usage.output
                                + output.usage.cache_read + output.usage.cache_write;
                        }
                    }
                }
                "message_stop" => {
                    // Stream complete, handled below
                }
                "error" => {
                    // Parse error event
                    if let Ok(error_val) = serde_json::from_str::<serde_json::Value>(data) {
                        let error_msg = error_val
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                            .unwrap_or("Unknown Anthropic error");
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
                "ping" => {
                    // Heartbeat, ignore
                }
                _ => {}
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

/// Track block types by index for content_block_stop handling.
#[derive(Debug, Clone, Copy)]
enum BlockType {
    Unknown,
    Text,
    Thinking,
    RedactedThinking,
    ToolUse,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type() {
        let provider = AnthropicProvider::new();
        assert_eq!(provider.provider_type(), Provider::Anthropic);
    }

    #[test]
    fn test_convert_messages_basic() {
        let mut context = Context::with_system_prompt("You are helpful.");
        context.add_message(Message::User(UserMessage::text("Hello")));

        let messages = convert_messages(&context);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }

    #[test]
    fn test_convert_tool_results_merged() {
        let mut context = Context::new();
        context.add_message(Message::ToolResult(ToolResultMessage::text(
            "call_1", "tool_a", "result1", false,
        )));
        context.add_message(Message::ToolResult(ToolResultMessage::text(
            "call_2", "tool_b", "result2", false,
        )));

        let messages = convert_messages(&context);
        // Tool results should be merged into a single user message
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        match &messages[0].content {
            AnthropicContent::Blocks(blocks) => assert_eq!(blocks.len(), 2),
            _ => panic!("Expected blocks"),
        }
    }
}

//! Message transformation for cross-provider compatibility.

use crate::types::*;

/// Transform messages for cross-provider compatibility.
///
/// This handles:
/// - Thinking block conversion between providers
/// - ToolCall ID normalization
/// - Orphan tool call handling
pub fn transform_messages(
    messages: &[Message],
    target_model: &Model,
    normalize_tool_call_id: Option<fn(&str) -> String>,
) -> Vec<Message> {
    let mut result = Vec::new();

    for msg in messages {
        match msg {
            Message::User(user_msg) => {
                result.push(Message::User(user_msg.clone()));
            }
            Message::Assistant(assistant_msg) => {
                // Skip error/aborted messages
                if assistant_msg.stop_reason == StopReason::Error
                    || assistant_msg.stop_reason == StopReason::Aborted
                {
                    continue;
                }

                let transformed = transform_assistant_message(
                    assistant_msg,
                    target_model,
                    normalize_tool_call_id,
                );
                result.push(Message::Assistant(transformed));
            }
            Message::ToolResult(tool_result) => {
                let mut result_msg = tool_result.clone();
                if let Some(normalize) = normalize_tool_call_id {
                    result_msg.tool_call_id = normalize(&result_msg.tool_call_id);
                }
                result.push(Message::ToolResult(result_msg));
            }
        }
    }

    // Handle orphan tool calls
    handle_orphan_tool_calls(&mut result);

    result
}

fn transform_assistant_message(
    msg: &AssistantMessage,
    target_model: &Model,
    normalize_tool_call_id: Option<fn(&str) -> String>,
) -> AssistantMessage {
    let mut new_msg = msg.clone();

    // Transform content blocks
    new_msg.content = msg
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Thinking(thinking) => {
                // Convert thinking blocks based on target provider
                transform_thinking_block(thinking, &msg.provider, &msg.model, target_model)
            }
            ContentBlock::ToolCall(tc) => {
                let mut new_tc = tc.clone();
                if let Some(normalize) = normalize_tool_call_id {
                    new_tc.id = normalize(&new_tc.id);
                }
                ContentBlock::ToolCall(new_tc)
            }
            _ => block.clone(),
        })
        .collect();

    if let Some(api) = &target_model.api {
        new_msg.api = api.clone();
    }
    new_msg.provider = target_model.provider.clone();
    new_msg.model = target_model.id.clone();

    new_msg
}

fn transform_thinking_block(
    thinking: &ThinkingContent,
    source_provider: &Provider,
    source_model: &str,
    target_model: &Model,
) -> ContentBlock {
    // If same provider and model, keep thinking block
    if *source_provider == target_model.provider && source_model == target_model.id {
        return ContentBlock::Thinking(thinking.clone());
    }

    // For different providers, convert to text
    // This avoids having the model mimic thinking tags
    if thinking.thinking.trim().is_empty() {
        ContentBlock::Text(TextContent::new(""))
    } else {
        ContentBlock::Text(TextContent::new(format!(
            "[Reasoning]\n{}\n[/Reasoning]",
            thinking.thinking
        )))
    }
}

fn handle_orphan_tool_calls(messages: &mut Vec<Message>) {
    // Find tool calls without corresponding results
    let mut tool_call_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut result_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    for msg in messages.iter() {
        match msg {
            Message::Assistant(assistant) => {
                for block in &assistant.content {
                    if let ContentBlock::ToolCall(tc) = block {
                        tool_call_ids.insert(tc.id.clone());
                    }
                }
            }
            Message::ToolResult(result) => {
                result_ids.insert(result.tool_call_id.clone());
            }
            _ => {}
        }
    }

    // Find orphan IDs
    let orphan_ids: std::collections::HashSet<String> =
        tool_call_ids.difference(&result_ids).cloned().collect();

    if orphan_ids.is_empty() {
        return;
    }

    // Insert synthetic error results for orphan tool calls
    let mut new_messages = Vec::new();
    for msg in messages.iter() {
        new_messages.push(msg.clone());
        if let Message::Assistant(assistant) = msg {
            let orphan_calls: Vec<_> = assistant
                .content
                .iter()
                .filter_map(|b| b.as_tool_call())
                .filter(|tc| orphan_ids.contains(&tc.id))
                .collect();

            for tc in orphan_calls {
                let error_result = ToolResultMessage::error(
                    tc.id.clone(),
                    tc.name.clone(),
                    "Tool call was not executed (orphaned)",
                );
                new_messages.push(Message::ToolResult(error_result));
            }
        }
    }

    *messages = new_messages;
}

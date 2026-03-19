//! Shared infrastructure for protocol providers.
//!
//! Eliminates duplication across `openai_completions`, `anthropic`, `google`,
//! and `openai_responses` by extracting common patterns:
//! - Base URL resolution
//! - on_payload hook application
//! - URL validation with error event emission
//! - Debug preview truncation
//! - Custom header injection (H2)
//! - HTTP error response handling
//! - SSE line buffer limit checking

use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use reqwest::header::HeaderMap;
use serde::Serialize;
use std::collections::HashMap;

/// Resolve the effective base URL using 3-level fallback:
/// `options.base_url` > `model.base_url` > `default`.
pub fn resolve_base_url<'a>(
    options_base_url: Option<&'a str>,
    model_base_url: Option<&'a str>,
    default: &'a str,
) -> &'a str {
    options_base_url.or(model_base_url).unwrap_or(default)
}

/// Apply the `on_payload` hook (if set) and serialize the request body.
///
/// When a hook is provided, the request is first serialized to `serde_json::Value`,
/// passed to the hook, and the (possibly modified) result is serialized to a JSON string.
/// Without a hook, the request is serialized directly to a JSON string.
pub async fn apply_on_payload<T: Serialize>(
    request: &T,
    hook: &Option<OnPayloadFn>,
    model: &Model,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(ref hook) = hook {
        let request_json = serde_json::to_value(request)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;
        match hook(request_json.clone(), model.clone()).await {
            Some(modified) => serde_json::to_string(&modified)
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) }),
            None => serde_json::to_string(&request_json)
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) }),
        }
    } else {
        serde_json::to_string(request)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })
    }
}

/// Validate the base URL against the security policy (H1).
///
/// On failure, pushes an `Error` event to the stream and returns `false`.
/// On success, returns `true`.
pub fn validate_url_or_error(
    base: &str,
    limits: &SecurityConfig,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) -> bool {
    if let Err(e) = limits.url.validate(base) {
        tracing::error!(url = %base, error = %e, "Base URL validation failed");
        output.stop_reason = StopReason::Error;
        output.error_message = Some(format!("URL validation error: {}", e));
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: output.clone(),
        });
        stream.end(None);
        false
    } else {
        true
    }
}

/// Return a truncated preview of the body string for debug logging.
pub fn debug_preview(body: &str, max_len: usize) -> &str {
    if body.len() > max_len {
        &body[..max_len]
    } else {
        body
    }
}

/// OpenAI-style APIs reject very small output-token limits.
///
/// Clamp any explicit token limit below 16 up to 16 before serializing the
/// request payload. `None` is preserved as-is so providers can apply their own
/// defaults.
pub fn clamp_openai_max_tokens(max_tokens: Option<u32>) -> Option<u32> {
    max_tokens.map(|value| value.max(16))
}

/// Inject custom headers, skipping protected headers per security policy (H2).
pub fn apply_custom_headers(
    headers: &mut HeaderMap,
    custom: &Option<HashMap<String, String>>,
    policy: &HeaderPolicy,
) {
    if let Some(ref custom_headers) = custom {
        for (key, value) in custom_headers {
            if policy.is_protected(key) {
                tracing::warn!(header = %key, "Skipping protected header override");
                continue;
            }
            if let Ok(header_name) = reqwest::header::HeaderName::try_from(key.clone()) {
                if let Ok(header_value) = reqwest::header::HeaderValue::try_from(value.clone()) {
                    headers.insert(header_name, header_value);
                }
            }
        }
    }
}

/// Handle an HTTP error response: read the body (bounded), log it,
/// push an `Error` event to the stream.
///
/// Returns `true` to indicate that an error was handled (caller should return early).
pub async fn handle_error_response(
    response: reqwest::Response,
    url: &str,
    model: &Model,
    limits: &SecurityConfig,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
    provider_name: &str,
) {
    let status = response.status();
    let body = crate::types::read_error_body(response, limits.http.max_error_body_bytes).await;
    tracing::error!(
        url = %url,
        model = %model.id,
        status = %status,
        response_body = %body,
        "{} request failed", provider_name
    );
    output.stop_reason = StopReason::Error;
    output.error_message = Some(crate::types::truncate_error_message(
        &format!("HTTP {}: {}", status, body),
        limits.http.max_error_message_chars,
    ));
    stream.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: output.clone(),
    });
    stream.end(None);
}

/// Check the SSE line buffer against the configured limit (C2).
///
/// On exceeding the limit, pushes an `Error` event to the stream and returns `true`
/// (indicating the stream should be aborted). Returns `false` if within limits.
pub fn check_sse_buffer_overflow(
    buffer_len: usize,
    max_bytes: usize,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) -> bool {
    if buffer_len > max_bytes {
        tracing::error!(
            buffer_size = buffer_len,
            limit = max_bytes,
            "SSE line buffer exceeded limit, aborting stream"
        );
        output.stop_reason = StopReason::Error;
        output.error_message = Some("SSE line buffer exceeded maximum size".to_string());
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: output.clone(),
        });
        stream.end(None);
        true
    } else {
        false
    }
}

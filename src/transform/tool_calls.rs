//! ToolCall ID normalization utilities.

/// Normalize a tool call ID to be compatible with different providers.
///
/// Different providers have different requirements:
/// - OpenAI: accepts most IDs
/// - Anthropic: requires `^[a-zA-Z0-9_-]+$`, max 64 chars
/// - Google: similar to OpenAI
pub fn normalize_tool_call_id(id: &str, target_provider: &crate::types::Provider) -> String {
    match target_provider {
        crate::types::Provider::Anthropic => {
            // Handle pipe-separated IDs from OpenAI Responses API
            // Format: {call_id}|{id} where {id} can be 400+ chars
            let id = if id.contains('|') {
                id.split('|').next().unwrap_or(id)
            } else {
                id
            };

            // Sanitize to allowed chars and truncate to 64 chars
            let sanitized: String = id
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .take(64)
                .collect();

            sanitized
        }
        crate::types::Provider::OpenAI | crate::types::Provider::Groq => {
            // OpenAI limits ID to 40 chars
            if id.len() > 40 {
                id[..40].to_string()
            } else {
                id.to_string()
            }
        }
        _ => id.to_string(),
    }
}

/// Create a mapping for tool call IDs between providers.
pub struct ToolCallIdMapper {
    /// Map from original ID to normalized ID
    to_normalized: std::collections::HashMap<String, String>,
    /// Map from normalized ID to original ID
    from_normalized: std::collections::HashMap<String, String>,
    /// Target provider
    target_provider: crate::types::Provider,
}

impl ToolCallIdMapper {
    /// Create a new mapper for a target provider.
    pub fn new(target_provider: crate::types::Provider) -> Self {
        Self {
            to_normalized: std::collections::HashMap::new(),
            from_normalized: std::collections::HashMap::new(),
            target_provider,
        }
    }

    /// Normalize an ID, caching the mapping.
    pub fn normalize(&mut self, id: &str) -> String {
        if let Some(normalized) = self.to_normalized.get(id) {
            return normalized.clone();
        }

        let normalized = normalize_tool_call_id(id, &self.target_provider);

        // Handle collisions
        let mut final_normalized = normalized.clone();
        let mut counter = 1;
        while self.from_normalized.contains_key(&final_normalized) {
            final_normalized = format!("{}_{}", normalized, counter);
            counter += 1;
        }

        self.to_normalized.insert(id.to_string(), final_normalized.clone());
        self.from_normalized.insert(final_normalized.clone(), id.to_string());

        final_normalized
    }

    /// Get the original ID from a normalized one.
    pub fn denormalize(&self, normalized: &str) -> Option<&String> {
        self.from_normalized.get(normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_for_anthropic() {
        let id = "call_abc123+def/ghi=";
        let normalized = normalize_tool_call_id(id, &crate::types::Provider::Anthropic);
        assert!(normalized.len() <= 64);
        assert!(normalized.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'));
    }

    #[test]
    fn test_normalize_pipe_separated() {
        let id = "call_123|very_long_suffix_here";
        let normalized = normalize_tool_call_id(id, &crate::types::Provider::Anthropic);
        assert!(!normalized.contains('|'));
        assert_eq!(normalized, "call_123");
    }

    #[test]
    fn test_normalize_for_openai() {
        let id = "a".repeat(50);
        let normalized = normalize_tool_call_id(&id, &crate::types::Provider::OpenAI);
        assert_eq!(normalized.len(), 40);
    }
}

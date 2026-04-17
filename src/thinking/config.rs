//! Thinking level and configuration.

use serde::{Deserialize, Serialize};

/// Thinking/Reasoning level for models that support it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    /// No thinking.
    Off,
    /// Minimal thinking.
    Minimal,
    /// Low thinking.
    Low,
    /// Medium thinking.
    Medium,
    /// High thinking.
    High,
    /// Extra high thinking (OpenAI GPT-5, Anthropic Opus 4.7+).
    XHigh,
}

impl Default for ThinkingLevel {
    fn default() -> Self {
        ThinkingLevel::Off
    }
}

impl std::fmt::Display for ThinkingLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThinkingLevel::Off => write!(f, "off"),
            ThinkingLevel::Minimal => write!(f, "minimal"),
            ThinkingLevel::Low => write!(f, "low"),
            ThinkingLevel::Medium => write!(f, "medium"),
            ThinkingLevel::High => write!(f, "high"),
            ThinkingLevel::XHigh => write!(f, "xhigh"),
        }
    }
}

impl From<&str> for ThinkingLevel {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "off" => ThinkingLevel::Off,
            "minimal" => ThinkingLevel::Minimal,
            "low" => ThinkingLevel::Low,
            "medium" => ThinkingLevel::Medium,
            "high" => ThinkingLevel::High,
            "xhigh" => ThinkingLevel::XHigh,
            _ => ThinkingLevel::Off,
        }
    }
}

/// Thinking content display mode for Anthropic models that support it (Opus 4.7+).
///
/// Controls whether thinking/reasoning content is included in the API response.
/// When a model requires explicit display opt-in (e.g., Opus 4.7 defaults to omitting
/// thinking content), this enum specifies the desired behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingDisplay {
    /// Thinking content is summarized and visible in the response.
    Summarized,
    /// Thinking content is omitted from the response.
    Omitted,
}

impl Default for ThinkingDisplay {
    fn default() -> Self {
        ThinkingDisplay::Summarized
    }
}

impl std::fmt::Display for ThinkingDisplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThinkingDisplay::Summarized => write!(f, "summarized"),
            ThinkingDisplay::Omitted => write!(f, "omitted"),
        }
    }
}

impl From<&str> for ThinkingDisplay {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "omitted" => ThinkingDisplay::Omitted,
            _ => ThinkingDisplay::Summarized,
        }
    }
}

/// Thinking configuration for models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Thinking level.
    pub level: ThinkingLevel,
    /// Budget tokens for thinking (provider-specific).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            level: ThinkingLevel::Off,
            budget_tokens: None,
        }
    }
}

impl ThinkingConfig {
    /// Create a new thinking config with the given level.
    pub fn new(level: ThinkingLevel) -> Self {
        Self {
            level,
            budget_tokens: None,
        }
    }

    /// Create a thinking config with budget tokens.
    pub fn with_budget(level: ThinkingLevel, budget_tokens: u32) -> Self {
        Self {
            level,
            budget_tokens: Some(budget_tokens),
        }
    }

    /// Get the default budget tokens for a level.
    pub fn default_budget(level: ThinkingLevel) -> u32 {
        match level {
            ThinkingLevel::Off => 0,
            ThinkingLevel::Minimal => 128,
            ThinkingLevel::Low => 512,
            ThinkingLevel::Medium => 1024,
            ThinkingLevel::High => 2048,
            ThinkingLevel::XHigh => 4096,
        }
    }
}

/// OpenAI-specific thinking options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAIThinkingOptions {
    /// Reasoning effort level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Reasoning summary mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_summary: Option<String>,
}

impl OpenAIThinkingOptions {
    /// Convert thinking level to OpenAI reasoning effort.
    pub fn from_level(level: ThinkingLevel) -> Self {
        let effort = match level {
            ThinkingLevel::Off => None,
            ThinkingLevel::Minimal => Some("minimal".to_string()),
            ThinkingLevel::Low => Some("low".to_string()),
            ThinkingLevel::Medium => Some("medium".to_string()),
            ThinkingLevel::High => Some("high".to_string()),
            ThinkingLevel::XHigh => Some("xhigh".to_string()),
        };

        Self {
            reasoning_effort: effort,
            reasoning_summary: None,
        }
    }
}

/// Anthropic-specific thinking options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicThinkingOptions {
    /// Whether thinking is enabled.
    pub thinking_enabled: bool,
    /// Budget tokens for thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    /// Whether to use adaptive thinking (Opus 4.6+ / Sonnet 4.6+).
    #[serde(default)]
    pub adaptive: bool,
}

impl AnthropicThinkingOptions {
    /// Convert thinking config to Anthropic options.
    pub fn from_config(config: &ThinkingConfig) -> Self {
        Self {
            thinking_enabled: config.level != ThinkingLevel::Off,
            budget_tokens: config.budget_tokens.or_else(|| {
                if config.level == ThinkingLevel::Off {
                    None
                } else {
                    Some(ThinkingConfig::default_budget(config.level))
                }
            }),
            adaptive: false,
        }
    }

    /// Create adaptive thinking options.
    pub fn adaptive(budget_tokens: Option<u32>) -> Self {
        Self {
            thinking_enabled: true,
            budget_tokens,
            adaptive: true,
        }
    }
}

/// Google-specific thinking options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoogleThinkingOptions {
    /// Whether thinking is enabled.
    pub enabled: bool,
    /// Budget tokens for thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

impl GoogleThinkingOptions {
    /// Convert thinking config to Google options.
    pub fn from_config(config: &ThinkingConfig) -> Self {
        Self {
            enabled: config.level != ThinkingLevel::Off,
            budget_tokens: config.budget_tokens,
        }
    }
}

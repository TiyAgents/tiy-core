//! Tests for thinking/config module.

use tiy_core::thinking::*;

// ============================================================================
// ThinkingLevel tests
// ============================================================================

#[test]
fn test_thinking_level_default() {
    assert_eq!(ThinkingLevel::default(), ThinkingLevel::Off);
}

#[test]
fn test_thinking_level_display() {
    assert_eq!(format!("{}", ThinkingLevel::Off), "off");
    assert_eq!(format!("{}", ThinkingLevel::Minimal), "minimal");
    assert_eq!(format!("{}", ThinkingLevel::Low), "low");
    assert_eq!(format!("{}", ThinkingLevel::Medium), "medium");
    assert_eq!(format!("{}", ThinkingLevel::High), "high");
    assert_eq!(format!("{}", ThinkingLevel::XHigh), "xhigh");
}

#[test]
fn test_thinking_level_from_str() {
    assert_eq!(ThinkingLevel::from("off"), ThinkingLevel::Off);
    assert_eq!(ThinkingLevel::from("minimal"), ThinkingLevel::Minimal);
    assert_eq!(ThinkingLevel::from("low"), ThinkingLevel::Low);
    assert_eq!(ThinkingLevel::from("medium"), ThinkingLevel::Medium);
    assert_eq!(ThinkingLevel::from("high"), ThinkingLevel::High);
    assert_eq!(ThinkingLevel::from("xhigh"), ThinkingLevel::XHigh);
}

#[test]
fn test_thinking_level_from_str_case_insensitive() {
    assert_eq!(ThinkingLevel::from("OFF"), ThinkingLevel::Off);
    assert_eq!(ThinkingLevel::from("High"), ThinkingLevel::High);
    assert_eq!(ThinkingLevel::from("MEDIUM"), ThinkingLevel::Medium);
}

#[test]
fn test_thinking_level_from_str_unknown_defaults_off() {
    assert_eq!(ThinkingLevel::from("unknown"), ThinkingLevel::Off);
    assert_eq!(ThinkingLevel::from(""), ThinkingLevel::Off);
    assert_eq!(ThinkingLevel::from("extreme"), ThinkingLevel::Off);
}

#[test]
fn test_thinking_level_serde_roundtrip() {
    let levels = vec![
        ThinkingLevel::Off,
        ThinkingLevel::Minimal,
        ThinkingLevel::Low,
        ThinkingLevel::Medium,
        ThinkingLevel::High,
        ThinkingLevel::XHigh,
    ];

    for level in levels {
        let json = serde_json::to_string(&level).unwrap();
        let back: ThinkingLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(back, level, "Failed roundtrip for {:?}", level);
    }
}

// ============================================================================
// ThinkingConfig tests
// ============================================================================

#[test]
fn test_thinking_config_default() {
    let config = ThinkingConfig::default();
    assert_eq!(config.level, ThinkingLevel::Off);
    assert!(config.budget_tokens.is_none());
}

#[test]
fn test_thinking_config_new() {
    let config = ThinkingConfig::new(ThinkingLevel::High);
    assert_eq!(config.level, ThinkingLevel::High);
    assert!(config.budget_tokens.is_none());
}

#[test]
fn test_thinking_config_with_budget() {
    let config = ThinkingConfig::with_budget(ThinkingLevel::Medium, 8192);
    assert_eq!(config.level, ThinkingLevel::Medium);
    assert_eq!(config.budget_tokens, Some(8192));
}

#[test]
fn test_thinking_config_default_budgets() {
    assert_eq!(ThinkingConfig::default_budget(ThinkingLevel::Off), 0);
    assert_eq!(ThinkingConfig::default_budget(ThinkingLevel::Minimal), 128);
    assert_eq!(ThinkingConfig::default_budget(ThinkingLevel::Low), 512);
    assert_eq!(ThinkingConfig::default_budget(ThinkingLevel::Medium), 1024);
    assert_eq!(ThinkingConfig::default_budget(ThinkingLevel::High), 2048);
    assert_eq!(ThinkingConfig::default_budget(ThinkingLevel::XHigh), 4096);
}

// ============================================================================
// OpenAIThinkingOptions tests
// ============================================================================

#[test]
fn test_openai_thinking_off() {
    let opts = OpenAIThinkingOptions::from_level(ThinkingLevel::Off);
    assert!(opts.reasoning_effort.is_none());
    assert!(opts.reasoning_summary.is_none());
}

#[test]
fn test_openai_thinking_levels() {
    let test_cases = vec![
        (ThinkingLevel::Minimal, "minimal"),
        (ThinkingLevel::Low, "low"),
        (ThinkingLevel::Medium, "medium"),
        (ThinkingLevel::High, "high"),
        (ThinkingLevel::XHigh, "xhigh"),
    ];

    for (level, expected) in test_cases {
        let opts = OpenAIThinkingOptions::from_level(level);
        assert_eq!(
            opts.reasoning_effort.as_deref(),
            Some(expected),
            "Failed for {:?}",
            level
        );
    }
}

// ============================================================================
// AnthropicThinkingOptions tests
// ============================================================================

#[test]
fn test_anthropic_thinking_off() {
    let config = ThinkingConfig::new(ThinkingLevel::Off);
    let opts = AnthropicThinkingOptions::from_config(&config);
    assert!(!opts.thinking_enabled);
    assert!(opts.budget_tokens.is_none());
    assert!(!opts.adaptive);
}

#[test]
fn test_anthropic_thinking_enabled_with_default_budget() {
    let config = ThinkingConfig::new(ThinkingLevel::Medium);
    let opts = AnthropicThinkingOptions::from_config(&config);
    assert!(opts.thinking_enabled);
    assert_eq!(opts.budget_tokens, Some(1024)); // Default for Medium
    assert!(!opts.adaptive);
}

#[test]
fn test_anthropic_thinking_custom_budget() {
    let config = ThinkingConfig::with_budget(ThinkingLevel::High, 32768);
    let opts = AnthropicThinkingOptions::from_config(&config);
    assert!(opts.thinking_enabled);
    assert_eq!(opts.budget_tokens, Some(32768)); // Custom budget overrides default
}

#[test]
fn test_anthropic_adaptive_thinking() {
    let opts = AnthropicThinkingOptions::adaptive(Some(16384));
    assert!(opts.thinking_enabled);
    assert!(opts.adaptive);
    assert_eq!(opts.budget_tokens, Some(16384));
}

#[test]
fn test_anthropic_adaptive_no_budget() {
    let opts = AnthropicThinkingOptions::adaptive(None);
    assert!(opts.thinking_enabled);
    assert!(opts.adaptive);
    assert!(opts.budget_tokens.is_none());
}

// ============================================================================
// GoogleThinkingOptions tests
// ============================================================================

#[test]
fn test_google_thinking_off() {
    let config = ThinkingConfig::new(ThinkingLevel::Off);
    let opts = GoogleThinkingOptions::from_config(&config);
    assert!(!opts.enabled);
    assert!(opts.budget_tokens.is_none());
}

#[test]
fn test_google_thinking_enabled() {
    let config = ThinkingConfig::new(ThinkingLevel::High);
    let opts = GoogleThinkingOptions::from_config(&config);
    assert!(opts.enabled);
    // No automatic budget for Google - only from config
    assert!(opts.budget_tokens.is_none());
}

#[test]
fn test_google_thinking_with_budget() {
    let config = ThinkingConfig::with_budget(ThinkingLevel::High, 24576);
    let opts = GoogleThinkingOptions::from_config(&config);
    assert!(opts.enabled);
    assert_eq!(opts.budget_tokens, Some(24576));
}

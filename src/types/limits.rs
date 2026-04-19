//! Security and resource limits configuration.
//!
//! Provides a centralized [`SecurityConfig`] struct for managing all security-related
//! limits across the library. Supports JSON/TOML serialization for file-based management.
//!
//! # Example
//!
//! ```rust
//! use tiycore::types::SecurityConfig;
//!
//! // Use defaults
//! let config = SecurityConfig::default();
//!
//! // Or deserialize from JSON
//! let json = r#"{ "http": { "connect_timeout_secs": 10 }, "agent": { "max_messages": 500 } }"#;
//! let config: SecurityConfig = serde_json::from_str(json).unwrap();
//! ```

use serde::{Deserialize, Serialize};
use std::time::Duration;

// ============================================================================
// Top-level SecurityConfig
// ============================================================================

/// Top-level security and resource limits configuration.
///
/// Designed to be loaded from config files (JSON/TOML) or constructed programmatically.
/// All sub-configs implement `Default` with conservative, safe values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// HTTP client and SSE stream limits (used by providers).
    #[serde(default)]
    pub http: HttpLimits,

    /// Agent-level limits (tool execution, message history, event queues).
    #[serde(default)]
    pub agent: AgentLimits,

    /// Stream/EventStream limits.
    #[serde(default)]
    pub stream: StreamLimits,

    /// Header security policy.
    #[serde(default)]
    pub headers: HeaderPolicy,

    /// Base URL validation policy.
    #[serde(default)]
    pub url: UrlPolicy,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            http: HttpLimits::default(),
            agent: AgentLimits::default(),
            stream: StreamLimits::default(),
            headers: HeaderPolicy::default(),
            url: UrlPolicy::default(),
        }
    }
}

impl SecurityConfig {
    /// Create a new SecurityConfig with all defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style setter for HTTP limits.
    pub fn with_http(mut self, http: HttpLimits) -> Self {
        self.http = http;
        self
    }

    /// Builder-style setter for agent limits.
    pub fn with_agent(mut self, agent: AgentLimits) -> Self {
        self.agent = agent;
        self
    }

    /// Builder-style setter for stream limits.
    pub fn with_stream(mut self, stream: StreamLimits) -> Self {
        self.stream = stream;
        self
    }

    /// Builder-style setter for header policy.
    pub fn with_headers(mut self, headers: HeaderPolicy) -> Self {
        self.headers = headers;
        self
    }

    /// Builder-style setter for URL policy.
    pub fn with_url(mut self, url: UrlPolicy) -> Self {
        self.url = url;
        self
    }
}

// ============================================================================
// HTTP / SSE Limits (Provider-level)
// ============================================================================

/// Limits applied to HTTP requests and SSE stream parsing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HttpLimits {
    /// TCP connect timeout in seconds. Default: 30.
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,

    /// Total request timeout (including streaming body) in seconds. Default: 1800 (30 min).
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,

    /// Maximum size of the SSE line buffer in bytes.
    /// Protects against OOM from a malicious server sending infinite data without newlines.
    /// Default: 2 MiB (2_097_152 bytes).
    #[serde(default = "default_max_sse_line_buffer_bytes")]
    pub max_sse_line_buffer_bytes: usize,

    /// Maximum size of an error response body to read, in bytes.
    /// Prevents OOM from unbounded error responses.
    /// Default: 64 KiB (65_536 bytes).
    #[serde(default = "default_max_error_body_bytes")]
    pub max_error_body_bytes: usize,

    /// Maximum character length of error messages stored in `AssistantMessage.error_message`.
    /// Truncated with "...[truncated]" suffix. Default: 4096 chars.
    #[serde(default = "default_max_error_message_chars")]
    pub max_error_message_chars: usize,
}

fn default_connect_timeout_secs() -> u64 {
    30
}
fn default_request_timeout_secs() -> u64 {
    1800
}
fn default_max_sse_line_buffer_bytes() -> usize {
    2 * 1024 * 1024
}
fn default_max_error_body_bytes() -> usize {
    64 * 1024
}
fn default_max_error_message_chars() -> usize {
    4096
}

impl Default for HttpLimits {
    fn default() -> Self {
        Self {
            connect_timeout_secs: default_connect_timeout_secs(),
            request_timeout_secs: default_request_timeout_secs(),
            max_sse_line_buffer_bytes: default_max_sse_line_buffer_bytes(),
            max_error_body_bytes: default_max_error_body_bytes(),
            max_error_message_chars: default_max_error_message_chars(),
        }
    }
}

impl HttpLimits {
    /// Get connect timeout as `Duration`.
    pub fn connect_timeout(&self) -> Duration {
        Duration::from_secs(self.connect_timeout_secs)
    }

    /// Get request timeout as `Duration`.
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_secs)
    }
}

// ============================================================================
// Agent Limits
// ============================================================================

/// Limits applied to the Agent runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentLimits {
    /// Maximum number of messages in agent conversation history.
    /// When exceeded, oldest messages are dropped (FIFO).
    /// 0 = unlimited. Default: 1000.
    #[serde(default = "default_max_messages")]
    pub max_messages: usize,

    /// Maximum number of parallel tool executions.
    /// Excess tool calls are queued and executed in batches.
    /// Default: 16.
    #[serde(default = "default_max_parallel_tool_calls")]
    pub max_parallel_tool_calls: usize,

    /// Timeout for a single tool execution in seconds. Default: 120.
    #[serde(default = "default_tool_execution_timeout_secs")]
    pub tool_execution_timeout_secs: u64,

    /// Whether to validate tool call arguments against JSON Schema before execution.
    /// Default: true.
    #[serde(default = "default_validate_tool_calls")]
    pub validate_tool_calls: bool,

    /// Maximum subscriber slots before compaction is triggered.
    /// Prevents tombstone memory leak. Default: 128.
    #[serde(default = "default_max_subscriber_slots")]
    pub max_subscriber_slots: usize,
}

fn default_max_messages() -> usize {
    1000
}
fn default_max_parallel_tool_calls() -> usize {
    16
}
fn default_tool_execution_timeout_secs() -> u64 {
    120
}
fn default_validate_tool_calls() -> bool {
    true
}
fn default_max_subscriber_slots() -> usize {
    128
}

impl Default for AgentLimits {
    fn default() -> Self {
        Self {
            max_messages: default_max_messages(),
            max_parallel_tool_calls: default_max_parallel_tool_calls(),
            tool_execution_timeout_secs: default_tool_execution_timeout_secs(),
            validate_tool_calls: default_validate_tool_calls(),
            max_subscriber_slots: default_max_subscriber_slots(),
        }
    }
}

impl AgentLimits {
    /// Get tool execution timeout as `Duration`.
    pub fn tool_execution_timeout(&self) -> Duration {
        Duration::from_secs(self.tool_execution_timeout_secs)
    }
}

// ============================================================================
// Stream Limits
// ============================================================================

/// Limits for the EventStream infrastructure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamLimits {
    /// Maximum number of events buffered in the EventStream queue.
    /// When exceeded, the oldest non-consumed events are dropped.
    /// 0 = unlimited. Default: 10_000.
    #[serde(default = "default_max_event_queue_size")]
    pub max_event_queue_size: usize,

    /// Timeout for `EventStream::result()` in seconds.
    /// Prevents infinite blocking. Default: 600 (10 min).
    #[serde(default = "default_result_timeout_secs")]
    pub result_timeout_secs: u64,
}

fn default_max_event_queue_size() -> usize {
    10_000
}
fn default_result_timeout_secs() -> u64 {
    600
}

impl Default for StreamLimits {
    fn default() -> Self {
        Self {
            max_event_queue_size: default_max_event_queue_size(),
            result_timeout_secs: default_result_timeout_secs(),
        }
    }
}

impl StreamLimits {
    /// Get result timeout as `Duration`.
    pub fn result_timeout(&self) -> Duration {
        Duration::from_secs(self.result_timeout_secs)
    }
}

// ============================================================================
// Header Policy
// ============================================================================

/// Policy for custom header handling to prevent security-sensitive overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeaderPolicy {
    /// Headers that cannot be overridden by user-supplied custom headers.
    /// Comparison is case-insensitive.
    /// Default: `["authorization", "x-api-key", "x-goog-api-key", "anthropic-version"]`
    #[serde(default = "default_protected_headers")]
    pub protected_headers: Vec<String>,
}

fn default_protected_headers() -> Vec<String> {
    vec![
        "authorization".to_string(),
        "x-api-key".to_string(),
        "x-goog-api-key".to_string(),
        "anthropic-version".to_string(),
        "anthropic-beta".to_string(),
    ]
}

impl Default for HeaderPolicy {
    fn default() -> Self {
        Self {
            protected_headers: default_protected_headers(),
        }
    }
}

impl HeaderPolicy {
    /// Check if a header name is protected and should not be overridden.
    /// Comparison is case-insensitive.
    pub fn is_protected(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        self.protected_headers
            .iter()
            .any(|h| h.to_lowercase() == lower)
    }
}

// ============================================================================
// URL Policy
// ============================================================================

/// Base URL validation policy for SSRF protection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UrlPolicy {
    /// Only allow HTTPS base URLs (except for known local providers like Ollama).
    /// Default: true.
    #[serde(default = "default_require_https")]
    pub require_https: bool,

    /// Disallow base URLs pointing to private/loopback IP ranges.
    /// Default: false (disabled by default to allow local development).
    #[serde(default)]
    pub block_private_ips: bool,

    /// Allowed URL schemes. Default: `["https", "http"]`.
    /// When `require_https` is true, HTTP URLs are rejected even if listed here
    /// (unless the host is localhost/127.0.0.1 for local development).
    #[serde(default = "default_allowed_schemes")]
    pub allowed_schemes: Vec<String>,

    /// Hostnames or domain suffixes exempt from the HTTPS requirement.
    ///
    /// Useful for enterprise intranet domains that serve HTTP only (e.g. `llm.oa.com`).
    ///
    /// - **Exact match**: `"llm.oa.com"` exempts only that host.
    /// - **Suffix match**: `".oa.com"` (leading dot) exempts any sub-domain such as
    ///   `llm.oa.com`, `api.llm.oa.com`, etc.
    ///
    /// Matching is case-insensitive. Default: empty (no exemptions).
    #[serde(default)]
    pub https_exempt_hosts: Vec<String>,
}

fn default_require_https() -> bool {
    true
}
fn default_allowed_schemes() -> Vec<String> {
    vec!["https".to_string(), "http".to_string()]
}

impl Default for UrlPolicy {
    fn default() -> Self {
        Self {
            require_https: default_require_https(),
            block_private_ips: false,
            allowed_schemes: default_allowed_schemes(),
            https_exempt_hosts: Vec::new(),
        }
    }
}

impl UrlPolicy {
    /// Builder-style setter for `https_exempt_hosts`.
    ///
    /// Each entry is either an exact hostname (`"llm.oa.com"`) or a domain suffix
    /// starting with a dot (`".oa.com"`) that matches all sub-domains.
    pub fn with_https_exempt_hosts(mut self, hosts: Vec<String>) -> Self {
        self.https_exempt_hosts = hosts;
        self
    }

    /// Check if a hostname is exempt from the HTTPS requirement.
    ///
    /// Supports exact match (`"llm.oa.com"`) and suffix match (`".oa.com"`).
    /// Matching is case-insensitive.
    fn is_https_exempt(&self, host: &str) -> bool {
        if self.https_exempt_hosts.is_empty() {
            return false;
        }
        let lower = host.to_lowercase();
        self.https_exempt_hosts.iter().any(|entry| {
            let entry_lower = entry.to_lowercase();
            if entry_lower.is_empty() || entry_lower == "." {
                return false;
            }
            if entry_lower.starts_with('.') {
                // Suffix match: ".oa.com" matches "llm.oa.com" and "api.llm.oa.com"
                lower.ends_with(&entry_lower)
            } else {
                // Exact match
                lower == entry_lower
            }
        })
    }

    /// Validate a base URL against the policy.
    /// Returns `Ok(())` if the URL passes validation, or an error description.
    ///
    /// Empty URLs are allowed (they will use provider defaults).
    pub fn validate(&self, url_str: &str) -> Result<(), String> {
        if url_str.is_empty() {
            return Ok(());
        }

        // Parse URL
        let parsed =
            url::Url::parse(url_str).map_err(|e| format!("Invalid URL '{}': {}", url_str, e))?;

        let scheme = parsed.scheme().to_lowercase();

        // Check allowed schemes
        if !self
            .allowed_schemes
            .iter()
            .any(|s| s.to_lowercase() == scheme)
        {
            return Err(format!(
                "URL scheme '{}' not allowed. Allowed: {:?}",
                scheme, self.allowed_schemes
            ));
        }

        // Check HTTPS requirement
        if self.require_https && scheme == "http" {
            // Allow HTTP for localhost/loopback (local development)
            let is_local = parsed.host_str().is_some_and(|h| is_local_host(h));
            // Allow HTTP for explicitly exempted hosts (enterprise intranet)
            let is_exempt = parsed
                .host_str()
                .is_some_and(|h| self.is_https_exempt(h));
            if !is_local && !is_exempt {
                return Err(format!(
                    "HTTPS is required for non-local URLs, got HTTP: '{}'",
                    url_str
                ));
            }
        }

        // Check private IP blocking
        if self.block_private_ips {
            if let Some(host) = parsed.host_str() {
                if is_private_host(host) {
                    return Err(format!(
                        "URL host '{}' resolves to a private/loopback address",
                        host
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Check if a hostname is a localhost/loopback address.
fn is_local_host(host: &str) -> bool {
    let lower = host.to_lowercase();
    lower == "localhost" || lower == "127.0.0.1" || lower == "::1" || lower == "[::1]"
}

/// Check if a hostname refers to a private/loopback address.
fn is_private_host(host: &str) -> bool {
    if is_local_host(host) {
        return true;
    }

    // Check IP address ranges
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        return match ip {
            std::net::IpAddr::V4(v4) => {
                v4.is_loopback() || v4.is_private() || v4.is_link_local() || v4.is_unspecified()
            }
            std::net::IpAddr::V6(v6) => {
                v6.is_loopback()
                    || v6.is_unspecified()
                    // fe80::/10 (link-local)
                    || (v6.segments()[0] & 0xffc0) == 0xfe80
            }
        };
    }

    false
}

// ============================================================================
// Shared helper functions (used by providers)
// ============================================================================

/// Read an error response body with a size limit.
///
/// Reads at most `max_bytes` from the response stream, preventing OOM
/// from unbounded error responses.
pub async fn read_error_body(response: reqwest::Response, max_bytes: usize) -> String {
    use futures::StreamExt;

    let mut body = Vec::with_capacity(max_bytes.min(4096));
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                body.extend_from_slice(&bytes);
                if body.len() >= max_bytes {
                    body.truncate(max_bytes);
                    break;
                }
            }
            Err(_) => break,
        }
    }

    let mut text = String::from_utf8_lossy(&body).to_string();
    if body.len() >= max_bytes {
        text.truncate(max_bytes.saturating_sub(15));
        text.push_str("...[truncated]");
    }
    text
}

/// Truncate an error message to a maximum character length.
pub fn truncate_error_message(msg: &str, max_chars: usize) -> String {
    if msg.len() <= max_chars {
        msg.to_string()
    } else {
        let truncated_len = max_chars.saturating_sub(15);
        format!("{}...[truncated]", &msg[..truncated_len])
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_config_default() {
        let config = SecurityConfig::default();
        assert_eq!(config.http.connect_timeout_secs, 30);
        assert_eq!(config.http.request_timeout_secs, 1800);
        assert_eq!(config.http.max_sse_line_buffer_bytes, 2 * 1024 * 1024);
        assert_eq!(config.http.max_error_body_bytes, 64 * 1024);
        assert_eq!(config.http.max_error_message_chars, 4096);
        assert_eq!(config.agent.max_messages, 1000);
        assert_eq!(config.agent.max_parallel_tool_calls, 16);
        assert_eq!(config.agent.tool_execution_timeout_secs, 120);
        assert!(config.agent.validate_tool_calls);
        assert_eq!(config.stream.max_event_queue_size, 10_000);
        assert_eq!(config.stream.result_timeout_secs, 600);
        assert!(config.headers.is_protected("Authorization"));
        assert!(config.headers.is_protected("x-api-key"));
        assert!(!config.headers.is_protected("content-type"));
    }

    #[test]
    fn test_security_config_serde_roundtrip() {
        let config = SecurityConfig::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        let deserialized: SecurityConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.http.max_sse_line_buffer_bytes,
            config.http.max_sse_line_buffer_bytes
        );
        assert_eq!(deserialized.agent.max_messages, config.agent.max_messages);
        assert_eq!(
            deserialized.stream.result_timeout_secs,
            config.stream.result_timeout_secs
        );
    }

    #[test]
    fn test_security_config_partial_json() {
        let json = r#"{
            "http": { "max_sse_line_buffer_bytes": 1048576, "connect_timeout_secs": 10 },
            "agent": { "max_messages": 500, "validate_tool_calls": false }
        }"#;
        let config: SecurityConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.http.max_sse_line_buffer_bytes, 1_048_576);
        assert_eq!(config.http.connect_timeout_secs, 10);
        // Unspecified fields use defaults
        assert_eq!(config.http.request_timeout_secs, 1800);
        assert_eq!(config.agent.max_messages, 500);
        assert!(!config.agent.validate_tool_calls);
        // Completely unspecified sections use defaults
        assert_eq!(config.stream.result_timeout_secs, 600);
    }

    #[test]
    fn test_security_config_empty_json() {
        let config: SecurityConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config.http.connect_timeout_secs, 30);
        assert_eq!(config.agent.max_messages, 1000);
    }

    #[test]
    fn test_header_policy_case_insensitive() {
        let policy = HeaderPolicy::default();
        assert!(policy.is_protected("Authorization"));
        assert!(policy.is_protected("AUTHORIZATION"));
        assert!(policy.is_protected("authorization"));
        assert!(policy.is_protected("X-Api-Key"));
        assert!(!policy.is_protected("X-Custom-Header"));
    }

    #[test]
    fn test_url_policy_valid_https() {
        let policy = UrlPolicy::default();
        assert!(policy.validate("https://api.openai.com/v1").is_ok());
        assert!(policy.validate("https://api.anthropic.com").is_ok());
        assert!(policy.validate("").is_ok()); // Empty is allowed
    }

    #[test]
    fn test_url_policy_http_local_allowed() {
        let policy = UrlPolicy::default();
        // HTTP to localhost is allowed even when require_https = true
        assert!(policy.validate("http://localhost:11434/v1").is_ok());
        assert!(policy.validate("http://127.0.0.1:11434/v1").is_ok());
    }

    #[test]
    fn test_url_policy_http_remote_blocked() {
        let policy = UrlPolicy::default();
        // HTTP to remote hosts is blocked when require_https = true
        assert!(policy.validate("http://api.openai.com/v1").is_err());
    }

    #[test]
    fn test_url_policy_private_ip_blocking() {
        let policy = UrlPolicy {
            block_private_ips: true,
            ..Default::default()
        };
        assert!(policy.validate("https://192.168.1.1:8080").is_err());
        assert!(policy.validate("https://10.0.0.1:8080").is_err());
        assert!(policy.validate("https://api.openai.com/v1").is_ok());
    }

    #[test]
    fn test_url_policy_invalid_url() {
        let policy = UrlPolicy::default();
        assert!(policy.validate("not a url").is_err());
    }

    #[test]
    fn test_url_policy_disabled() {
        let policy = UrlPolicy {
            require_https: false,
            block_private_ips: false,
            ..Default::default()
        };
        assert!(policy.validate("http://api.openai.com/v1").is_ok());
        assert!(policy.validate("http://192.168.1.1:8080").is_ok());
    }

    #[test]
    fn test_truncate_error_message() {
        assert_eq!(truncate_error_message("short", 100), "short");
        let long = "a".repeat(5000);
        let truncated = truncate_error_message(&long, 100);
        assert!(truncated.len() <= 100);
        assert!(truncated.ends_with("...[truncated]"));
    }

    #[test]
    fn test_http_limits_duration_helpers() {
        let limits = HttpLimits::default();
        assert_eq!(limits.connect_timeout(), Duration::from_secs(30));
        assert_eq!(limits.request_timeout(), Duration::from_secs(1800));
    }

    #[test]
    fn test_agent_limits_duration_helpers() {
        let limits = AgentLimits::default();
        assert_eq!(limits.tool_execution_timeout(), Duration::from_secs(120));
    }

    #[test]
    fn test_stream_limits_duration_helpers() {
        let limits = StreamLimits::default();
        assert_eq!(limits.result_timeout(), Duration::from_secs(600));
    }

    #[test]
    fn test_builder_pattern() {
        let config = SecurityConfig::new()
            .with_http(HttpLimits {
                connect_timeout_secs: 10,
                ..Default::default()
            })
            .with_agent(AgentLimits {
                max_messages: 500,
                ..Default::default()
            });
        assert_eq!(config.http.connect_timeout_secs, 10);
        assert_eq!(config.agent.max_messages, 500);
        // Other sections still default
        assert_eq!(config.stream.result_timeout_secs, 600);
    }

    // ---- HTTPS exempt hosts tests ----

    #[test]
    fn test_url_policy_https_exempt_exact_match() {
        let policy = UrlPolicy::default().with_https_exempt_hosts(vec!["llm.oa.com".to_string()]);
        // Exact match allows HTTP
        assert!(policy.validate("http://llm.oa.com/v1").is_ok());
        // Different host still blocked
        assert!(policy.validate("http://other.example.com/v1").is_err());
    }

    #[test]
    fn test_url_policy_https_exempt_suffix_match() {
        let policy = UrlPolicy::default().with_https_exempt_hosts(vec![".oa.com".to_string()]);
        // Sub-domains match the suffix
        assert!(policy.validate("http://llm.oa.com/v1").is_ok());
        assert!(policy.validate("http://api.llm.oa.com/v1").is_ok());
        // Bare "oa.com" does NOT match ".oa.com" (no leading dot in host)
        assert!(policy.validate("http://oa.com/v1").is_err());
        // Unrelated domain still blocked
        assert!(policy.validate("http://api.example.com/v1").is_err());
    }

    #[test]
    fn test_url_policy_https_exempt_case_insensitive() {
        let policy = UrlPolicy::default().with_https_exempt_hosts(vec!["LLM.OA.COM".to_string()]);
        assert!(policy.validate("http://llm.oa.com/v1").is_ok());
        assert!(policy.validate("http://LLM.OA.COM/v1").is_ok());

        let policy2 =
            UrlPolicy::default().with_https_exempt_hosts(vec![".OA.COM".to_string()]);
        assert!(policy2.validate("http://api.oa.com/v1").is_ok());
    }

    #[test]
    fn test_url_policy_https_exempt_does_not_affect_https() {
        // HTTPS URLs always pass regardless of exempt list
        let policy = UrlPolicy::default().with_https_exempt_hosts(vec!["llm.oa.com".to_string()]);
        assert!(policy.validate("https://llm.oa.com/v1").is_ok());
        assert!(policy.validate("https://other.example.com/v1").is_ok());
    }

    #[test]
    fn test_url_policy_https_exempt_empty_by_default() {
        let policy = UrlPolicy::default();
        assert!(policy.https_exempt_hosts.is_empty());
        // Without exemptions, remote HTTP is blocked as usual
        assert!(policy.validate("http://llm.oa.com/v1").is_err());
    }

    #[test]
    fn test_url_policy_https_exempt_serde_roundtrip() {
        let json = r#"{
            "require_https": true,
            "https_exempt_hosts": ["llm.oa.com", ".internal.corp"]
        }"#;
        let policy: UrlPolicy = serde_json::from_str(json).unwrap();
        assert_eq!(policy.https_exempt_hosts.len(), 2);
        assert!(policy.validate("http://llm.oa.com/v1").is_ok());
        assert!(policy.validate("http://api.internal.corp/v1").is_ok());
        assert!(policy.validate("http://external.com/v1").is_err());
    }
}

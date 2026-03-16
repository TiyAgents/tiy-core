//! Streaming JSON parser for handling incomplete JSON during tool call streaming.

use serde_json::Value;

/// Parse potentially incomplete JSON string.
///
/// This function handles the case where JSON is being streamed character by character
/// and may be incomplete. It attempts to parse valid JSON first, then falls back
/// to partial parsing.
pub fn parse_streaming_json(json: &str) -> Value {
    if json.trim().is_empty() {
        return Value::Object(serde_json::Map::new());
    }

    // Try normal parsing first
    if let Ok(value) = serde_json::from_str::<Value>(json) {
        return value;
    }

    // Try to fix and parse incomplete JSON
    let fixed = fix_incomplete_json(json);
    if let Ok(value) = serde_json::from_str::<Value>(&fixed) {
        return value;
    }

    // Return empty object as fallback
    Value::Object(serde_json::Map::new())
}

/// Fix potentially incomplete JSON by adding missing closing characters.
fn fix_incomplete_json(json: &str) -> String {
    let mut result = json.to_string();
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escape_next = false;

    for ch in json.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }

        match ch {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '{' | '[' if !in_string => stack.push(ch),
            '}' if !in_string => {
                if let Some('{') = stack.last() {
                    stack.pop();
                }
            }
            ']' if !in_string => {
                if let Some('[') = stack.last() {
                    stack.pop();
                }
            }
            _ => {}
        }
    }

    // Close any unclosed strings
    if in_string {
        result.push('"');
    }

    // Close any unclosed structures in reverse order
    while let Some(ch) = stack.pop() {
        match ch {
            '{' => result.push('}'),
            '[' => result.push(']'),
            _ => {}
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_json() {
        let result = parse_streaming_json("");
        assert!(result.is_object());
        assert!(result.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_complete_json() {
        let result = parse_streaming_json(r#"{"name": "test", "value": 123}"#);
        assert_eq!(result["name"], "test");
        assert_eq!(result["value"], 123);
    }

    #[test]
    fn test_incomplete_json_object() {
        // Missing closing brace
        let result = parse_streaming_json(r#"{"name": "test""#);
        assert_eq!(result["name"], "test");
    }

    #[test]
    fn test_incomplete_json_nested() {
        // Nested incomplete object
        let result = parse_streaming_json(r#"{"outer": {"inner": "value""#);
        assert_eq!(result["outer"]["inner"], "value");
    }

    #[test]
    fn test_incomplete_json_array() {
        // Incomplete array
        let result = parse_streaming_json(r#"{"items": [1, 2, 3"#);
        let items = result["items"].as_array().unwrap();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_incomplete_json_string() {
        // Unclosed string
        let result = parse_streaming_json(r#"{"text": "hello"#);
        assert_eq!(result["text"], "hello");
    }

    #[test]
    fn test_escaped_quotes() {
        let json = r#"{"text": "hello \"world\""}"#;
        let result = parse_streaming_json(json);
        assert_eq!(result["text"], r#"hello "world""#);
    }
}

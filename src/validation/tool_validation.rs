//! Tool parameter validation using JSON Schema.

use crate::types::Tool;
use crate::types::ToolCall;
use jsonschema::{Draft, JSONSchema, ValidationError};
use serde_json::Value;

/// Validate tool call arguments against tool schema.
pub fn validate_tool_call(tools: &[Tool], tool_call: &ToolCall) -> Result<Value, ToolValidationError> {
    let tool = tools
        .iter()
        .find(|t| t.name == tool_call.name)
        .ok_or_else(|| ToolValidationError::ToolNotFound(tool_call.name.clone()))?;

    validate_tool_arguments(tool, tool_call)
}

/// Validate arguments against a tool's schema.
pub fn validate_tool_arguments(tool: &Tool, tool_call: &ToolCall) -> Result<Value, ToolValidationError> {
    let compiled = JSONSchema::options()
        .with_draft(Draft::Draft7)
        .compile(&tool.parameters)
        .map_err(|e| ToolValidationError::SchemaError(e.to_string()))?;

    let tool_name = tool.name.clone();
    let args = tool_call.arguments.clone();

    // Validate first
    if let Err(errors) = compiled.validate(&args) {
        let error_messages: Vec<String> = errors
            .map(|e| format!("  - {}: {}", e.instance_path, e))
            .collect();

        return Err(ToolValidationError::ValidationFailed(tool_name, error_messages));
    }

    // Return a new validated args (we already have a clone)
    Ok(args)
}

/// Error type for tool validation.
#[derive(Debug, thiserror::Error)]
pub enum ToolValidationError {
    #[error("Tool '{0}' not found")]
    ToolNotFound(String),

    #[error("Invalid JSON schema: {0}")]
    SchemaError(String),

    #[error("Validation failed for tool '{0}': {1:?}")]
    ValidationFailed(String, Vec<String>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_tool() -> Tool {
        Tool::builder()
            .name("get_weather")
            .description("Get weather for a location")
            .parameters(json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            }))
            .build()
            .unwrap()
    }

    #[test]
    fn test_valid_tool_call() {
        let tool = create_test_tool();
        let tool_call = ToolCall::new(
            "call_1",
            "get_weather",
            json!({"location": "Tokyo", "units": "celsius"}),
        );

        let result = validate_tool_call(&[tool], &tool_call);
        assert!(result.is_ok());
    }

    #[test]
    fn test_missing_required_field() {
        let tool = create_test_tool();
        let tool_call = ToolCall::new(
            "call_1",
            "get_weather",
            json!({"units": "celsius"}),
        );

        let result = validate_tool_call(&[tool], &tool_call);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_enum_value() {
        let tool = create_test_tool();
        let tool_call = ToolCall::new(
            "call_1",
            "get_weather",
            json!({"location": "Tokyo", "units": "kelvin"}),
        );

        let result = validate_tool_call(&[tool], &tool_call);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_not_found() {
        let tool = create_test_tool();
        let tool_call = ToolCall::new(
            "call_1",
            "unknown_tool",
            json!({}),
        );

        let result = validate_tool_call(&[tool], &tool_call);
        assert!(matches!(result, Err(ToolValidationError::ToolNotFound(_))));
    }
}

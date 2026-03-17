//! Tests for validation module: tool_validation.

use serde_json::json;
use tiy_core::types::*;
use tiy_core::validation::{validate_tool_call, ToolValidationError};

fn weather_tool() -> Tool {
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

fn calculator_tool() -> Tool {
    Tool::builder()
        .name("calculator")
        .description("Basic calculator")
        .parameters(json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string"
                },
                "precision": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10
                }
            },
            "required": ["expression"]
        }))
        .build()
        .unwrap()
}

// ============================================================================
// validate_tool_call tests
// ============================================================================

#[test]
fn test_validate_valid_call_all_fields() {
    let tool = weather_tool();
    let call = ToolCall::new(
        "call_1",
        "get_weather",
        json!({"location": "Tokyo", "units": "celsius"}),
    );
    let result = validate_tool_call(&[tool], &call);
    assert!(result.is_ok());
    let args = result.unwrap();
    assert_eq!(args["location"], "Tokyo");
    assert_eq!(args["units"], "celsius");
}

#[test]
fn test_validate_valid_call_required_only() {
    let tool = weather_tool();
    let call = ToolCall::new("call_1", "get_weather", json!({"location": "Paris"}));
    let result = validate_tool_call(&[tool], &call);
    assert!(result.is_ok());
}

#[test]
fn test_validate_missing_required_field() {
    let tool = weather_tool();
    let call = ToolCall::new("call_1", "get_weather", json!({"units": "celsius"}));
    let result = validate_tool_call(&[tool], &call);
    assert!(result.is_err());
    match result.unwrap_err() {
        ToolValidationError::ValidationFailed(name, errors) => {
            assert_eq!(name, "get_weather");
            assert!(!errors.is_empty());
        }
        _ => panic!("Expected ValidationFailed"),
    }
}

#[test]
fn test_validate_invalid_enum_value() {
    let tool = weather_tool();
    let call = ToolCall::new(
        "call_1",
        "get_weather",
        json!({"location": "Tokyo", "units": "kelvin"}),
    );
    let result = validate_tool_call(&[tool], &call);
    assert!(result.is_err());
}

#[test]
fn test_validate_wrong_type() {
    let tool = weather_tool();
    let call = ToolCall::new("call_1", "get_weather", json!({"location": 42}));
    let result = validate_tool_call(&[tool], &call);
    assert!(result.is_err());
}

#[test]
fn test_validate_tool_not_found() {
    let tool = weather_tool();
    let call = ToolCall::new("call_1", "nonexistent_tool", json!({}));
    let result = validate_tool_call(&[tool], &call);
    assert!(
        matches!(result, Err(ToolValidationError::ToolNotFound(name)) if name == "nonexistent_tool")
    );
}

#[test]
fn test_validate_empty_tool_list() {
    let call = ToolCall::new("call_1", "any_tool", json!({}));
    let result = validate_tool_call(&[], &call);
    assert!(matches!(result, Err(ToolValidationError::ToolNotFound(_))));
}

#[test]
fn test_validate_multiple_tools_finds_correct() {
    let tools = vec![weather_tool(), calculator_tool()];
    let call = ToolCall::new("call_1", "calculator", json!({"expression": "2+2"}));
    let result = validate_tool_call(&tools, &call);
    assert!(result.is_ok());
}

#[test]
fn test_validate_integer_bounds() {
    let tool = calculator_tool();
    let call = ToolCall::new(
        "call_1",
        "calculator",
        json!({"expression": "1+1", "precision": 15}),
    );
    let result = validate_tool_call(&[tool], &call);
    assert!(result.is_err()); // precision > 10
}

#[test]
fn test_validate_extra_properties_allowed_by_default() {
    // JSON Schema draft7 allows additional properties unless additionalProperties: false
    let tool = weather_tool();
    let call = ToolCall::new(
        "call_1",
        "get_weather",
        json!({"location": "Tokyo", "extra": "field"}),
    );
    let result = validate_tool_call(&[tool], &call);
    assert!(result.is_ok());
}

#[test]
fn test_validate_empty_object_missing_required() {
    let tool = weather_tool();
    let call = ToolCall::new("call_1", "get_weather", json!({}));
    let result = validate_tool_call(&[tool], &call);
    assert!(result.is_err());
}

#[test]
fn test_validate_nested_object_schema() {
    let tool = Tool::new(
        "nested_tool",
        "Tool with nested schema",
        json!({
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "number"}
                    },
                    "required": ["key"]
                }
            },
            "required": ["config"]
        }),
    );

    // Valid nested
    let call = ToolCall::new(
        "c1",
        "nested_tool",
        json!({"config": {"key": "foo", "value": 42}}),
    );
    assert!(validate_tool_call(&[tool.clone()], &call).is_ok());

    // Missing nested required
    let call = ToolCall::new("c2", "nested_tool", json!({"config": {"value": 42}}));
    assert!(validate_tool_call(&[tool.clone()], &call).is_err());

    // Missing top-level required
    let call = ToolCall::new("c3", "nested_tool", json!({}));
    assert!(validate_tool_call(&[tool], &call).is_err());
}

#[test]
fn test_validate_array_schema() {
    let tool = Tool::new(
        "list_tool",
        "Tool with array",
        json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                }
            },
            "required": ["items"]
        }),
    );

    let call = ToolCall::new("c1", "list_tool", json!({"items": ["a", "b", "c"]}));
    assert!(validate_tool_call(&[tool.clone()], &call).is_ok());

    let call = ToolCall::new("c2", "list_tool", json!({"items": []}));
    assert!(validate_tool_call(&[tool.clone()], &call).is_err()); // minItems: 1

    let call = ToolCall::new("c3", "list_tool", json!({"items": [1, 2]}));
    assert!(validate_tool_call(&[tool], &call).is_err()); // wrong item type
}

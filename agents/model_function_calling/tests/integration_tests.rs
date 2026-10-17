use model_function_calling::{ChatRequest, FunctionCall, Message, ToolCall};
use serde_json::json;

#[test]
fn test_tool_calling_request_serialization() {
    // Tool schema definition
    let tools_schema = json!([
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Use this function to get the current price of a stock.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The ticker symbol for the stock"
                        }
                    },
                    "required": ["ticker"]
                }
            }
        }
    ]);

    let messages = vec![
        Message {
            role: "user".to_string(),
            content: "What is the current stock price for MSFT?".to_string(),
            images: None,
            tool_calls: None,
            name: None,
        }
    ];

    let request = ChatRequest {
        model: "granite4.2:3b".to_string(),
        messages,
        stream: false,
        tools: Some(tools_schema),
    };

    // Verify that the structure containing the tools serializes successfully to JSON
    let json_result = serde_json::to_string(&request);
    assert!(json_result.is_ok());

    let json_str = json_result.unwrap();
    assert!(json_str.contains("granite4.2:3b"));
    assert!(json_str.contains("tools"));
    assert!(json_str.contains("get_stock_price"));
}

#[test]
fn test_tool_response_message_serialization() {
    // Simulate a message with role 'tool' that returns the result of a local function back to the model
    let tool_message = Message {
        role: "tool".to_string(),
        content: r#"{"ticker": "MSFT", "current_price": 514.05}"#.to_string(),
        images: None,
        tool_calls: None,
        name: Some("get_stock_price".to_string()),
    };

    let json_result = serde_json::to_string(&tool_message);
    assert!(json_result.is_ok());

    let json_str = json_result.unwrap();
    assert!(json_str.contains(r#""role":"tool""#));
    assert!(json_str.contains(r#""name":"get_stock_price""#));
    assert!(json_str.contains("514.05"));
}
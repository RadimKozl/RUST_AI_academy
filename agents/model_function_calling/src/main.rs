use model_function_calling::{call_chat, ChatRequest, Message};
use serde_json::json;
use std::error::Error;

// Simulated function for retrieving stock prices (similar to yfinance from the tutorial)
fn get_stock_price(ticker: &str) -> serde_json::Value {
    // In a real-world application, there might be an HTTP call to an actual API here.
    json!({
        "ticker": ticker,
        "current_price": 514.05
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = reqwest::Client::new();
    let chat_url = "http://localhost:11434/api/chat";
    let model_name = "granite4.2:3b"; // Model trained for tool use

    // Definition of tools using a JSON schema
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
                            "description": "The ticker symbol for the stock, e.g. MSFT"
                        }
                    },
                    "required": ["ticker"]
                }
            }
        }
    ]);

    // Conversation initialization
    let mut messages = vec![
        Message {
            role: "user".to_string(),
            content: "What is the current stock price for MSFT?".to_string(),
            images: None,
            tool_calls: None,
            name: None,
        }
    ];

    let payload = ChatRequest {
        model: model_name.to_string(),
        messages: messages.clone(),
        stream: false,
        tools: Some(tools_schema.clone()),
    };

    println!("--- 1. Sending a request with defined tools ---");
    let response = call_chat(&client, chat_url, &payload).await?;

    // Check whether the model returned a request for a function call (tool call)
    if let Some(tool_calls) = &response.message.tool_calls {
        if let Some(tool_call) = tool_calls.first() {
            let function_name = &tool_call.function.name;
            println!("The model requires a function call: {}", function_name);

            if function_name == "get_stock_price" {
                // Extraction of the 'ticker' argument
                if let Some(ticker) = tool_call.function.arguments.get("ticker").and_then(|v| v.as_str()) {
                    println!("I am launching a local function for the ticker: {}", ticker);
                    
                    // Execution of the function
                    let func_result = get_stock_price(ticker);
                    println!("Function result: {}", func_result);

                    // Adding the assistant's response (containing tool_calls) to the history
                    messages.push(response.message.clone());

                    // Adding the function result with the 'tool' role back to the message history
                    messages.push(Message {
                        role: "tool".to_string(),
                        content: func_result.to_string(),
                        images: None,
                        tool_calls: None,
                        name: Some(function_name.clone()),
                    });

                    // 2. Round: Sending the context, including the function result, back to the model
                    let final_payload = ChatRequest {
                        model: model_name.to_string(),
                        messages,
                        stream: false,
                        tools: Some(tools_schema),
                    };

                    println!("\n--- 2. Sending the function result back to the model ---");
                    let final_response = call_chat(&client, chat_url, &final_payload).await?;
                    
                    println!("Model's final response:\n{}", final_response.message.content);
                }
            }
        }
    } else {
        println!("The model did not return any tool calls; response: {}", response.message.content);
    }

    Ok(())
}
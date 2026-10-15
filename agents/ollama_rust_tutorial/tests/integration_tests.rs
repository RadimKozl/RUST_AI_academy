use ollama_rust_tutorial::{ChatRequest, ChatResponse, Message};

#[test]
fn test_chat_request_serialization() {
    let request = ChatRequest {
        model: "gemma3n:e4b".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: "Test message".to_string(),
        }],
        stream: false,
    };

    // Verify that the structure is correctly converted to JSON
    let json_result = serde_json::to_string(&request);
    assert!(json_result.is_ok());

    let json_str = json_result.unwrap();
    assert!(json_str.contains("gemma3n:e4b"));
    assert!(json_str.contains("Test message"));
}

#[test]
fn test_chat_response_deserialization() {
    // Simulated JSON response from the Ollama API
    let json_data = r#"{
        "model": "gemma3n:e4b",
        "created_at": "2025-10-13T15:49:42.1953413Z",
        "message": {
            "role": "assistant",
            "content": "Why don't scientists trust atoms? Because they make up everything!"
        },
        "done": true
    }"#;

    // We verify that the JSON maps successfully to our ChatResponse structure
    let response_result: Result<ChatResponse, _> = serde_json::from_str(json_data);
    assert!(response_result.is_ok());

    let response = response_result.unwrap();
    assert_eq!(response.model, "gemma3n:e4b");
    assert_eq!(response.message.role, "assistant");
    assert!(response.message.content.contains("scientists trust atoms"));
    assert!(response.done);
}
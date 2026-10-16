use rust_multimedia_models::{ChatRequest, ChatResponse, Message};

#[test]
fn test_chat_request_with_image_serialization() {
    let request = ChatRequest {
        model: "qwen3-vl:2b".to_string(),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: "You are an AI assistant.".to_string(),
                images: None,
            },
            Message {
                role: "user".to_string(),
                content: "What is in this image?".to_string(),
                // Simulated Base64 image string
                images: Some(vec!["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==".to_string()]),
            },
        ],
        stream: false,
    };

    // Verify that the structure containing the image serializes correctly to JSON
    let json_result = serde_json::to_string(&request);
    assert!(json_result.is_ok());

    let json_str = json_result.unwrap();
    assert!(json_str.contains("qwen3-vl:2b"));
    assert!(json_str.contains("images"));
    assert!(json_str.contains("iVBORw0KGgoAAAANSUhEUg"));
}

#[test]
fn test_vlm_chat_response_deserialization() {
    // Simulated JSON response from the Ollama VLM API (e.g., analysis of a car from the tutorial)
    let json_data = r#"{
        "model": "qwen3-vl:2b",
        "created_at": "2025-10-13T18:24:21.9716653Z",
        "message": {
            "role": "assistant",
            "content": "The image shows a black car in front of a brick house.",
            "images": null
        },
        "done": true
    }"#;

    let response_result: Result<ChatResponse, _> = serde_json::from_str(json_data);
    assert!(response_result.is_ok());

    let response = response_result.unwrap();
    assert_eq!(response.model, "qwen3-vl:2b");
    assert_eq!(response.message.role, "assistant");
    assert!(response.message.content.contains("black car"));
    assert!(response.done);
}
use model_chat_history::{ChatRequest, Message};

#[test]
fn test_chat_history_serialization() {
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: "You are an AI assistant.".to_string(),
            images: None,
        },
        Message {
            role: "user".to_string(),
            content: "Tell me a joke.".to_string(),
            images: None,
        },
        Message {
            role: "assistant".to_string(),
            content: "Why don't scientists trust atoms? Because they make up everything!".to_string(),
            images: None,
        },
        Message {
            role: "user".to_string(),
            content: "Repeat me please the previous joke.".to_string(),
            images: None,
        },
    ];

    let request = ChatRequest {
        model: "gemma3n:e4b".to_string(),
        messages,
        stream: false,
    };

    let json_result = serde_json::to_string(&request);
    assert!(json_result.is_ok());

    let json_str = json_result.unwrap();
    assert!(json_str.contains("system"));
    assert!(json_str.contains("assistant"));
    assert!(json_str.contains("Repeat me please"));
}
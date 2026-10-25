use vlm_ocr_response::{ChatOptions, ChatRequest, Message, OcrResponse};

#[test]
fn test_ocr_request_serialization() {
    let request = ChatRequest {
        model: "qwen3-vl:8b".to_string(),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: "You are an OCR system. Return JSON.".to_string(),
                images: None,
            },
            Message {
                role: "user".to_string(),
                content: "Extract text.".to_string(),
                images: Some(vec!["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==".to_string()]),
            },
        ],
        stream: false,
        format: Some("json".to_string()),
        options: Some(ChatOptions { temperature: 0.2 }),
    };

    let json_result = serde_json::to_string(&request);
    assert!(json_result.is_ok());
    let json_str = json_result.unwrap();
    assert!(json_str.contains("qwen3-vl:8b"));
    assert!(json_str.contains(r#""format":"json""#));
}

#[test]
fn test_ocr_response_deserialization() {
    // Test JSON corresponding to the text from the new image
    let json_data = r#"{
        "extracted_text": "Take it Easy"
    }"#;

    let response_result: Result<OcrResponse, _> = serde_json::from_str(json_data);
    assert!(response_result.is_ok());

    let ocr = response_result.unwrap();
    assert_eq!(ocr.extracted_text, "Take it Easy");
}
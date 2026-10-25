use image_classification::{ChatOptions, ChatRequest, ImageAnalysisResponse, Message};

#[test]
fn test_image_analysis_request_serialization() {
    let request = ChatRequest {
        model: "qwen3-vl:8b".to_string(),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: "You are an image analyzer. Respond with JSON.".to_string(),
                images: None,
            },
            Message {
                role: "user".to_string(),
                content: "Classify this image.".to_string(),
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
    assert!(json_str.contains(r#""temperature":0.2"#));
    assert!(json_str.contains("images"));
}

#[test]
fn test_image_analysis_response_deserialization() {
    // Simulated JSON response from the model for image classification and description
    let json_data = r#"{
        "classification": "car",
        "description": "A black sports car parked on a gravel driveway in front of a rustic brick building."
    }"#;

    let response_result: Result<ImageAnalysisResponse, _> = serde_json::from_str(json_data);
    assert!(response_result.is_ok());

    let analysis = response_result.unwrap();
    assert_eq!(analysis.classification, "car");
    assert!(analysis.description.contains("black sports car"));
}
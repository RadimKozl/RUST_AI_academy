use object_detection::{ChatOptions, ChatRequest, DetectionResponse, Message};

#[test]
fn test_vlm_detection_request_serialization() {
    let request = ChatRequest {
        model: "qwen3-vl:8b".to_string(),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: "You are an object detector. Return JSON.".to_string(),
                images: None,
            },
            Message {
                role: "user".to_string(),
                content: "Detect cars in this image.".to_string(),
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
fn test_vlm_detection_response_deserialization() {
    // Simulated JSON response from the model, containing detected objects and image dimensions
    let json_data = r#"{
        "image_size": {
            "width": 800,
            "height": 600
        },
        "objects": [
            {
                "bbox": [100, 150, 300, 400]
            }
        ]
    }"#;

    let response_result: Result<DetectionResponse, _> = serde_json::from_str(json_data);
    assert!(response_result.is_ok());

    let detection = response_result.unwrap();
    assert_eq!(detection.image_size.width, 800);
    assert_eq!(detection.image_size.height, 600);
    assert_eq!(detection.objects.len(), 1);
    assert_eq!(detection.objects[0].bbox, vec![100, 150, 300, 400]);
}
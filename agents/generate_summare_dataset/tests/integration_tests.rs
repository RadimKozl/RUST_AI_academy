use generate_summare_dataset::{GenerateRequest, NewsItem};

#[test]
fn test_news_item_serialization() {
    let item = NewsItem {
        id: "123e4567-e89b-12d3-a456-426614174000".to_string(),
        message: "This is a generated news message about science and technology.".to_string(),
        summarization: "A short summary about tech news.".to_string(),
    };

    let json_result = serde_json::to_string(&item);
    assert!(json_result.is_ok());

    let json_str = json_result.unwrap();
    assert!(json_str.contains("123e4567-e89b-12d3-a456-426614174000"));
    assert!(json_str.contains("message"));
    assert!(json_str.contains("summarization"));
}

#[test]
fn test_generate_request_serialization() {
    let request = GenerateRequest {
        model: "granite4.2:3b".to_string(),
        prompt: "Generate a random news message in English about sports.".to_string(),
        stream: false,
    };

    let json_result = serde_json::to_string(&request);
    assert!(json_result.is_ok());

    let json_str = json_result.unwrap();
    assert!(json_str.contains("granite4.2:3b"));
    assert!(json_str.contains("sports"));
    assert!(json_str.contains(r#"stream":false"#));
}
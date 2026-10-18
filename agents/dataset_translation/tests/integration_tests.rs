use dataset_translation::TranslatedNewsItem;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct InputRecord {
    #[allow(dead_code)]
    id: Option<String>,
    message: String,
    #[allow(dead_code)]
    summarization: Option<String>,
}

#[test]
fn test_input_json_deserialization() {
    let json_data = r#"[
        {
            "id": "b165149a-eb9e-463a-b745-37c81b9ecfe9",
            "message": "Researchers at the National Institute of Health have announced a breakthrough...",
            "summarization": "A short summary of the medical breakthrough."
        }
    ]"#;

    let records: Result<Vec<InputRecord>, _> = serde_json::from_str(json_data);
    assert!(records.is_ok());

    let records = records.unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].id.as_deref(), Some("b165149a-eb9e-463a-b745-37c81b9ecfe9"));
    assert!(records[0].message.contains("National Institute of Health"));
}

#[test]
fn test_translated_news_item_serialization() {
    let item = TranslatedNewsItem {
        id: "123e4567-e89b-12d3-a456-426614174000".to_string(),
        origin_text: "Vědci oznámili průlom v léčbě...".to_string(),
        summary_text: "Stručné shrnutí průlomu v léčbě.".to_string(),
    };

    let json_result = serde_json::to_string(&item);
    assert!(json_result.is_ok());

    let json_str = json_result.unwrap();
    assert!(json_str.contains("123e4567-e89b-12d3-a456-426614174000"));
    assert!(json_str.contains("origin_text"));
    assert!(json_str.contains("summary_text"));
}
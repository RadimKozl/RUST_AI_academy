use rust_multimedia_models::{call_chat, ChatRequest, Message};
use std::error::Error;
use std::fs;
use base64::{Engine as _, engine::general_purpose};

// Helper function for converting an image to a Base64 string
fn encode_image(image_path: &str) -> Result<String, Box<dyn Error>> {
    let image_bytes = fs::read(image_path)?;
    let encoded = general_purpose::STANDARD.encode(&image_bytes);
    Ok(encoded)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = reqwest::Client::new();
    let chat_url = "http://localhost:11434/api/chat";

    // Path to the test image (e.g., in the images/ directory)
    let image_path = "images/car_test_image.png";
    
    println!("Loading and encoding the image: {}", image_path);
    let base64_image = encode_image(image_path)?;

    // Constructing a request for a multimodal model (e.g., qwen3-vl:2b)
    let chat_payload = ChatRequest {
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
                images: Some(vec![base64_image]), // Passing an image in Base64
            },
        ],
        stream: false,
    };

    println!("Sending a request to the VLM Ollama endpoint...");
    let chat_res = call_chat(&client, chat_url, &chat_payload).await?;

    println!("\n--- VLM Model Response ---");
    println!("Model: {}", chat_res.model);
    println!("Response:\n{}", chat_res.message.content);

    Ok(())
}
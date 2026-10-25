use vlm_ocr_response::{call_chat, ChatOptions, ChatRequest, Message, OcrResponse};
use base64::{engine::general_purpose, Engine as _};
use std::error::Error;
use std::fs;
use std::path::Path;

fn encode_image(image_path: &str) -> Result<String, Box<dyn Error>> {
    let image_bytes = fs::read(image_path)?;
    Ok(general_purpose::STANDARD.encode(&image_bytes))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = reqwest::Client::new();
    let chat_url = "http://localhost:11434/api/chat";
    
    let model = "qwen3-vl:8b";
    let image_path = "images/text_image.jpg"; // Example of an image containing text (e.g., a car license plate)

    // System prompt for OCR from a Python script
    let system_prompt = 
        "You are an OCR system specialized in extracting text from images. \
        Respond ONLY with valid JSON in the following schema: \
        {\"extracted_text\": \"text_content\"}. \
        - 'extracted_text' should contain all readable text found in the image as a single string. \
        - If no text is found or the image is unclear, return an empty string for 'extracted_text'. \
        - Do not output anything else besides the JSON.";

    println!("Loading and encoding image for OCR: {}", image_path);
    let base64_image = match encode_image(image_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load image file: {}", e);
            return Err(e.into());
        }
    };

    let chat_payload = ChatRequest {
        model: model.to_string(),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: system_prompt.to_string(),
                images: None,
            },
            Message {
                role: "user".to_string(),
                content: "Extract all readable text from this image.".to_string(),
                images: Some(vec![base64_image]),
            },
        ],
        stream: false,
        format: Some("json".to_string()), // Enforcing a JSON schema
        options: Some(ChatOptions { temperature: 0.2 }),
    };

    println!("Sending request to Ollama ({}) for OCR...", model);
    let chat_res = call_chat(&client, chat_url, &chat_payload).await?;

    let response_text = chat_res.message.content.trim();
    println!("\n--- Raw model response ---");
    println!("{}", response_text);

    // Parsing the JSON response with fallback protection
    let ocr_data: OcrResponse = match serde_json::from_str(response_text) {
        Ok(data) => data,
        Err(_) => {
            if let (Some(start), Some(end)) = (response_text.find('{'), response_text.rfind('}')) {
                let json_slice = &response_text[start..=end];
                serde_json::from_str(json_slice).unwrap_or(OcrResponse {
                    extracted_text: "".to_string(),
                })
            } else {
                OcrResponse {
                    extracted_text: "".to_string(),
                }
            }
        }
    };

    println!("\n--- Extracted Text Result ---");
    println!("Text: \"{}\"", ocr_data.extracted_text);

    // Saving the output to the file {nazev}_ocr.json
    let path_obj = Path::new(image_path);
    let file_stem = path_obj.file_stem().unwrap().to_str().unwrap();
    let parent_dir = path_obj.parent().unwrap_or(Path::new("images"));
    let output_json_path = parent_dir.join(format!("{}_ocr.json", file_stem));

    let json_output_string = serde_json::to_string_pretty(&ocr_data)?;
    fs::write(&output_json_path, json_output_string)?;

    println!("\n✔ Saved OCR JSON to: {:?}", output_json_path);

    Ok(())
}
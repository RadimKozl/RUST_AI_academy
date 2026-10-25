use image_classification::{call_chat, ChatOptions, ChatRequest, ImageAnalysisResponse, Message};
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
    
    // We will use qwen3-vl:8b, which excels in this role
    let model = "qwen3-vl:8b";
    let image_path = "images/car_test_image.png"; // Path to the test image

    // System prompt enforcing a precise JSON schema for classification
    let system_prompt = 
        "You are an image analyzer specialized in classifying and describing images. \
        Respond ONLY with valid JSON in the following schema: \
        {\"classification\": \"main_category\", \"description\": \"detailed_description\"}. \
        - 'classification' should be a single string representing the main category or type of the image (e.g., 'landscape', 'portrait', 'car'). \
        - 'description' should be a detailed textual description of the image content, including key elements, colors, and context. \
        - If the image is unclear or cannot be classified, use 'unknown' for classification and explain in description. \
        - Do not output anything else besides the JSON.";

    println!("Loading and encoding image: {}", image_path);
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
                content: "Classify the main category of this image and provide a detailed description of its content.".to_string(),
                images: Some(vec![base64_image]),
            },
        ],
        stream: false,
        format: Some("json".to_string()), // Enforcing JSON mode in Ollama
        options: Some(ChatOptions { temperature: 0.2 }),
    };

    println!("Sending request to Ollama ({}) for image analysis...", model);
    let chat_res = call_chat(&client, chat_url, &chat_payload).await?;

    let response_text = chat_res.message.content.trim();
    println!("\n--- Raw model response ---");
    println!("{}", response_text);

    // Parsing the JSON response into a structure with a fallback
    let analysis_data: ImageAnalysisResponse = match serde_json::from_str(response_text) {
        Ok(data) => data,
        Err(_) => {
            // Safety catch, in case the model includes some text around the JSON
            if let (Some(start), Some(end)) = (response_text.find('{'), response_text.rfind('}')) {
                let json_slice = &response_text[start..=end];
                serde_json::from_str(json_slice).unwrap_or(ImageAnalysisResponse {
                    classification: "unknown".to_string(),
                    description: "Unable to parse model response properly.".to_string(),
                })
            } else {
                ImageAnalysisResponse {
                    classification: "unknown".to_string(),
                    description: "Model did not return valid JSON.".to_string(),
                }
            }
        }
    };

    println!("\n--- Parsed Analysis ---");
    println!("Classification: {}", analysis_data.classification);
    println!("Description:    {}", analysis_data.description);

    // Saving the output to the file {nazev}_analysis.json
    let path_obj = Path::new(image_path);
    let file_stem = path_obj.file_stem().unwrap().to_str().unwrap();
    let parent_dir = path_obj.parent().unwrap_or(Path::new("images"));
    let output_json_path = parent_dir.join(format!("{}_analysis.json", file_stem));

    let json_output_string = serde_json::to_string_pretty(&analysis_data)?;
    fs::write(&output_json_path, json_output_string)?;

    println!("\n✔ Saved JSON analysis to: {:?}", output_json_path);

    Ok(())
}
use ollama_rust_tutorial::{
    call_chat, call_generate, ChatRequest, GenerateRequest, Message,
};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = reqwest::Client::new();

    // ----------------------------------------------------
    // 1. Basic call using a chat template (/api/chat)
    // ----------------------------------------------------
    let chat_url = "http://localhost:11434/api/chat";
    
    let chat_payload = ChatRequest {
        model: "gemma3n:e4b".to_string(), // Model used in the tutorial
        messages: vec![
            Message {
                role: "system".to_string(),
                content: "You are an AI assistant.".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "Tell me a joke.".to_string(),
            },
        ],
        stream: false,
    };

    println!("--- 1. Call to /api/chat (Local client) ---");
    let chat_res = call_chat(&client, chat_url, &chat_payload).await?;

    println!("Model: {}", chat_res.model);
    println!("--- Response text: ---\n{}\n", chat_res.message.content);

    // ----------------------------------------------------
    // 2. Calling using a simplified format (/api/generate)
    // ----------------------------------------------------
    let generate_url = "http://localhost:11434/api/generate";
    
    let generate_payload = GenerateRequest {
        model: "gemma3n:e4b".to_string(),
        prompt: "Tell me a joke.".to_string(),
        stream: false,
    };

    println!("--- 2. Calling /api/generate (Simplified format) ---");
    let generate_res = call_generate(&client, generate_url, &generate_payload).await?;

    println!("--- Response text: ---\n{}\n", generate_res.response);

    Ok(())
}
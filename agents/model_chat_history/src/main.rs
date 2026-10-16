use model_chat_history::{call_chat, ChatRequest, Message};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = reqwest::Client::new();
    let chat_url = "http://localhost:11434/api/chat";
    let model_name = "gemma3n:e4b"; // Model used in the tutorial

    // --------------------------------------------------------------
    // Initialization of chat history (analogous to a Python list)
    // --------------------------------------------------------------
    let mut messages: Vec<Message> = vec![
        Message {
            role: "system".to_string(),
            content: "You are an AI assistant.".to_string(),
            images: None,
        }
    ];

    // ----------------------------------------------------
    // 1. First user query
    // ----------------------------------------------------
    messages.push(Message {
        role: "user".to_string(),
        content: "Tell me a joke.".to_string(),
        images: None,
    });

    let payload1 = ChatRequest {
        model: model_name.to_string(),
        messages: messages.clone(),
        stream: false,
    };

    println!("--- 1. Sending the first inquiry ---");
    let response1 = call_chat(&client, chat_url, &payload1).await?;
    println!("Assistant's response:\n{}", response1.message.content);

    // Adding the assistant's response to the chat history
    messages.push(response1.message.clone());

    // ----------------------------------------------------
    // 2. Second user query (requiring context/history)
    // ----------------------------------------------------
    messages.push(Message {
        role: "user".to_string(),
        content: "Repeat me please the previous joke.".to_string(),
        images: None,
    });

    let payload2 = ChatRequest {
        model: model_name.to_string(),
        messages: messages.clone(),
        stream: false,
    };

    println!("\n--- 2. Sending the second query (with history) ---");
    let response2 = call_chat(&client, chat_url, &payload2).await?;
    println!("Assistant's response:\n{}", response2.message.content);

    // Adding the second response to the history
    messages.push(response2.message.clone());

    Ok(())
}

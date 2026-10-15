use serde::{Deserialize, Serialize};

// Data structures for the Chat API (/api/chat)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,
}

#[derive(Deserialize, Debug)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: Message,
    pub done: bool,
}

// Data structures for the Generate API (/api/generate)
#[derive(Serialize, Debug)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
}

#[derive(Deserialize, Debug)]
pub struct GenerateResponse {
    pub model: String,
    pub response: String,
    pub done: bool,
}

// Modular function for calling /api/chat
pub async fn call_chat(
    client: &reqwest::Client,
    url: &str,
    payload: &ChatRequest,
) -> Result<ChatResponse, reqwest::Error> {
    let res = client
        .post(url)
        .json(payload)
        .send()
        .await?
        .json::<ChatResponse>()
        .await?;
    Ok(res)
}

// Modular function for calling /api/generate
pub async fn call_generate(
    client: &reqwest::Client,
    url: &str,
    payload: &GenerateRequest,
) -> Result<GenerateResponse, reqwest::Error> {
    let res = client
        .post(url)
        .json(payload)
        .send()
        .await?
        .json::<GenerateResponse>()
        .await?;
    Ok(res)
}
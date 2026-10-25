use serde::{Deserialize, Serialize};

// --- Structure for OCR output ---
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OcrResponse {
    pub extracted_text: String,
}

// --- Common structures for the Ollama API ---
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

#[derive(Serialize, Debug)]
pub struct ChatOptions {
    pub temperature: f32,
}

#[derive(Serialize, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ChatOptions>,
}

#[derive(Deserialize, Debug)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: Message,
    pub done: bool,
}

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
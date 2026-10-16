use serde::{Deserialize, Serialize};

// Extended message structure with support for VLM (images)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
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

// Function for asynchronous chat calls (including multimodal data)
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
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    pub function: FunctionCall,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_json::Value>,
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
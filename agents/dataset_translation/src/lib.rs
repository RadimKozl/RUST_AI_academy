use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TranslatedNewsItem {
    pub id: String,
    pub origin_text: String,
    pub summary_text: String,
}

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
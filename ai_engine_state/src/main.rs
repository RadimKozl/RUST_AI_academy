use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

// ============================================================================
// 1. CHYBOVÉ STAVY (Error Handling)
// ============================================================================
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Error communicating with Ollama API: {0}")]
    OllamaNetworkError(#[from] reqwest::Error),

    #[error("Ollama returned an invalid HTTP status: {0}")]
    OllamaStatusError(StatusCode),

    #[error("Error while processing JSON: {0}")]
    JsonError(#[from] serde_json::Error),
}

// Convert to Axum HTTP Response
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, err_msg) = match self {
            ApiError::OllamaNetworkError(e) => (StatusCode::BAD_GATEWAY, e.to_string()),
            ApiError::OllamaStatusError(code) => (code, format!("Ollama API returned HTTP {}", code)),
            ApiError::JsonError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        };

        let body = Json(serde_json::json!({
            "success": false,
            "error": err_msg
        }));

        (status, body).into_response()
    }
}

// ============================================================================
// 2. STRUCTS FOR OLLAMA AND CLIENT DTO
// ============================================================================

// Input request from our REST API client
#[derive(Deserialize)]
pub struct ChatRequest {
    pub prompt: String,
    pub system_prompt: Option<String>,
}

// Response from our REST API
#[derive(Serialize)]
pub struct ChatResponse {
    pub model: String,
    pub response: String,
    pub done: bool,
}

// DTO for official Ollama /api/generate endpoint
#[derive(Serialize)]
struct OllamaApiPayload<'a> {
    model: &'a str,
    prompt: &'a str,
    system: Option<&'a str>,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaApiResponse {
    model: String,
    response: String,
    done: bool,
}

// ============================================================================
// 3. SHARED STATE AXIOM / AXUM
// ============================================================================
pub struct AppState {
    pub http_client: reqwest::Client,
    pub ollama_url: String,
    pub model_name: String,
}

// ============================================================================
// 4. HANDLERS
// ============================================================================
async fn health_check() -> &'static str {
    "OK - AI Engine ready"
}

async fn generate_chat(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, ApiError> {
    // Preparing data for Ollama API
    let ollama_payload = OllamaApiPayload {
        model: &state.model_name,
        prompt: &payload.prompt,
        system: payload.system_prompt.as_deref(),
        stream: false, // For simplicity, we do not use SSE streaming
    };

    let target_endpoint = format!("{}/api/generate", state.ollama_url);

    // Sending a request to Ollam
    let res = state
        .http_client
        .post(&target_endpoint)
        .json(&ollama_payload)
        .send()
        .await?;

    if !res.status().is_success() {
        return Err(ApiError::OllamaStatusError(res.status()));
    }

    let ollama_res: OllamaApiResponse = res.json().await?;

    Ok(Json(ChatResponse {
        model: ollama_res.model,
        response: ollama_res.response,
        done: ollama_res.done,
    }))
}

// ============================================================================
// 5. MAIN LOGIC
// ============================================================================
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Rust LLM Microservice...");

    // Vytvoření sdíleného HTTP klienta s connection-poolingem
    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120)) // Dělší timeout pro inference
        .build()?;

    let shared_state = Arc::new(AppState {
        http_client,
        ollama_url: "http://127.0.0.1:11434".to_string(),
        model_name: "gemma3n:e4b".to_string(),
    });

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/chat", post(generate_chat))
        .with_state(shared_state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    println!("🌐 REST API started at http://127.0.0.1:3000");
    println!("🤖 Connected to Ollama model: gemma3n:e4b");

    axum::serve(listener, app).await?;

    Ok(())
}
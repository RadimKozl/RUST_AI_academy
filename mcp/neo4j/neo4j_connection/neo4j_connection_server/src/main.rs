use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use neo4rs::{query, ConfigBuilder, Graph};
use neo4j_connection_server::{AppState, MoviesGraphServer};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

type SessionSenders = Arc<Mutex<HashMap<String, mpsc::Sender<String>>>>;

#[derive(Clone)]
struct HttpBridgeState {
    mcp_server: MoviesGraphServer,
    sessions: SessionSenders,
}

#[derive(Deserialize)]
struct MessageQuery {
    #[serde(rename = "sessionId")]
    session_id: Option<String>,
}

#[derive(Deserialize)]
struct JsonRpcRequest {
    jsonrpc: Option<String>,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

fn json_rpc_ok(id: Option<Value>, result: Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: Some(result),
        error: None,
    }
}

fn json_rpc_err(id: Option<Value>, code: i64, message: impl Into<String>) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(JsonRpcError {
            code,
            message: message.into(),
        }),
    }
}

async fn handle_json_rpc(server: &MoviesGraphServer, req: JsonRpcRequest) -> Option<JsonRpcResponse> {
    let id = req.id.clone();

    match req.method.as_str() {
        "initialize" => {
            let info = rmcp::handler::server::ServerHandler::get_info(server);
            match serde_json::to_value(info) {
                Ok(val) => Some(json_rpc_ok(id, val)),
                Err(err) => Some(json_rpc_err(id, -32603, format!("Serialize error: {err}"))),
            }
        }
        "notifications/initialized" => None,
        "tools/list" => {
            let stats_tool = serde_json::json!({
                "name": "graph_statistics",
                "description": "Count the number of nodes and relationships in the graph.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            });
            Some(json_rpc_ok(id, serde_json::json!({ "tools": [stats_tool] })))
        }
        "tools/call" => {
            let tool_name = req
                .params
                .as_ref()
                .and_then(|p| p.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("");

            if tool_name != "graph_statistics" {
                return Some(json_rpc_err(
                    id,
                    -32601,
                    format!("Unknown tool: {tool_name}"),
                ));
            }

            match server.query_graph_statistics().await {
                Ok(stats) => match serde_json::to_string(&stats) {
                    Ok(json_text) => Some(json_rpc_ok(
                        id,
                        serde_json::json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": json_text
                                }
                            ],
                            "isError": false
                        }),
                    )),
                    Err(err) => Some(json_rpc_err(id, -32603, format!("JSON error: {err}"))),
                },
                Err(err) => Some(json_rpc_err(
                    id,
                    -32603,
                    format!("Neo4j error: {err}"),
                )),
            }
        }
        other => {
            warn!(method = %other, "Unsupported MCP method over SSE bridge");
            id.map(|req_id| {
                json_rpc_err(Some(req_id), -32601, format!("Method not found: {other}"))
            })
        }
    }
}

async fn sse_handler(
    State(state): State<HttpBridgeState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let session_id = Uuid::new_v4().to_string();
    let (tx, rx) = mpsc::channel::<String>(32);

    {
        let mut guard = state.sessions.lock().await;
        guard.insert(session_id.clone(), tx.clone());
    }

    let endpoint_uri = format!("/messages?sessionId={session_id}");
    let _ = tx
        .send(
            serde_json::to_string(&serde_json::json!({
                "type": "endpoint",
                "endpoint": endpoint_uri
            }))
            .unwrap_or_default(),
        )
        .await;

    let initial_event = Event::default().event("endpoint").data(endpoint_uri);
    let stream = tokio_stream::once(Ok(initial_event)).chain(ReceiverStream::new(rx).map(|msg| {
        Ok(Event::default().event("message").data(msg))
    }));

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

async fn messages_handler(
    State(state): State<HttpBridgeState>,
    Query(query_params): Query<MessageQuery>,
    Json(payload): Json<JsonRpcRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let session_id = query_params
        .session_id
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "Missing sessionId query parameter".into()))?;

    let sender = {
        let guard = state.sessions.lock().await;
        guard.get(&session_id).cloned()
    }
    .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Unknown sessionId: {session_id}")))?;

    let version_ok = payload.jsonrpc.as_deref().unwrap_or("2.0") == "2.0";
    if !version_ok {
        return Err((StatusCode::BAD_REQUEST, "Invalid jsonrpc version".into()));
    }

    if let Some(response) = handle_json_rpc(&state.mcp_server, payload).await {
        let serialized = serde_json::to_string(&response).map_err(|err| {
            error!(error = %err, "Failed to serialize JSON-RPC response");
            (StatusCode::INTERNAL_SERVER_ERROR, "Serialization failed".into())
        })?;

        sender.send(serialized).await.map_err(|_| {
            (
                StatusCode::GONE,
                "SSE client disconnected before receiving response".into(),
            )
        })?;
    }

    Ok(StatusCode::ACCEPTED)
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".into());
    let username = std::env::var("NEO4J_USERNAME").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".into());
    let database = std::env::var("NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".into());

    info!(uri = %uri, username = %username, database = %database, "Connecting to Neo4j");

    let config = ConfigBuilder::new()
        .uri(&uri)
        .user(&username)
        .password(&password)
        .db(database.as_str())
        .build()
        .context("Invalid Neo4j driver configuration")?;

    let graph = Graph::connect(config)
        .await
        .context("Failed to connect to Neo4j")?;

    graph
        .run(query("RETURN 1"))
        .await
        .context("Neo4j connectivity check failed")?;

    let state = Arc::new(AppState {
        graph: Arc::new(graph),
        database,
    });

    let bridge_state = HttpBridgeState {
        mcp_server: MoviesGraphServer::new(state),
        sessions: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/sse", get(sse_handler))
        .route("/messages", post(messages_handler))
        .with_state(bridge_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8000));
    info!(%addr, "Starting Movies GraphRAG SSE MCP server");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("Failed to bind TCP listener on 0.0.0.0:8000")?;

    axum::serve(listener, app)
        .await
        .context("HTTP server exited with error")?;

    Ok(())
}
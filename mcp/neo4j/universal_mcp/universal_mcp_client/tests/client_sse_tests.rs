use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex};
use tokio::time::timeout;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;
use universal_mcp_client::{
    origin_from_url, parse_sse_event_block, resolve_endpoint_url, run_graphrag_client_checks,
};
use uuid::Uuid;

#[test]
fn test_parse_sse_event_block() {
    let raw = ": keepalive comment\r\nevent: endpoint\r\ndata: /messages?sessionId=sess-1\r\n";
    assert_eq!(
        parse_sse_event_block(raw),
        Some((
            "endpoint".to_string(),
            "/messages?sessionId=sess-1".to_string()
        ))
    );

    let multiline = "data: {\"line\":1}\ndata: {\"line\":2}";
    assert_eq!(
        parse_sse_event_block(multiline),
        Some((
            "message".to_string(),
            "{\"line\":1}\n{\"line\":2}".to_string()
        ))
    );

    assert_eq!(parse_sse_event_block(": comment only\n\n"), None);
}

#[test]
fn test_origin_and_endpoint_resolution() {
    assert_eq!(
        origin_from_url("http://127.0.0.1:54321/sse"),
        "http://127.0.0.1:54321"
    );
    assert_eq!(
        origin_from_url("http://localhost:8000/sse"),
        "http://localhost:8000"
    );
    assert_eq!(
        resolve_endpoint_url("http://127.0.0.1:54321", "/messages?sessionId=abc"),
        "http://127.0.0.1:54321/messages?sessionId=abc"
    );
    assert_eq!(
        resolve_endpoint_url(
            "http://127.0.0.1:54321",
            "http://127.0.0.1:9999/messages?sessionId=def"
        ),
        "http://127.0.0.1:9999/messages?sessionId=def"
    );
}

type SessionMap = Arc<Mutex<HashMap<String, mpsc::Sender<String>>>>;

#[derive(Clone)]
struct MockBridgeState {
    sessions: SessionMap,
}

#[derive(Deserialize)]
struct SessionQuery {
    #[serde(rename = "sessionId")]
    session_id: Option<String>,
}

#[derive(Deserialize)]
struct JsonRpcIn {
    id: Option<Value>,
    method: String,
}

async fn handle_mock_sse(
    State(state): State<MockBridgeState>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let sid = Uuid::new_v4().to_string();
    let (tx, rx) = mpsc::channel::<String>(32);

    {
        let mut guard = state.sessions.lock().await;
        guard.insert(sid.clone(), tx);
    }

    let endpoint = format!("/messages?sessionId={sid}");
    let first = Event::default().event("endpoint").data(endpoint);
    let stream = tokio_stream::once(Ok(first)).chain(
        ReceiverStream::new(rx).map(|payload| Ok(Event::default().event("message").data(payload))),
    );

    Sse::new(stream)
}

async fn handle_mock_messages(
    State(state): State<MockBridgeState>,
    Query(q): Query<SessionQuery>,
    Json(req): Json<JsonRpcIn>,
) -> Result<StatusCode, StatusCode> {
    let sid = q.session_id.ok_or(StatusCode::BAD_REQUEST)?;
    let tx = {
        let guard = state.sessions.lock().await;
        guard.get(&sid).cloned().ok_or(StatusCode::NOT_FOUND)?
    };

    let resp = match req.method.as_str() {
        "initialize" => Some(json!({
            "jsonrpc": "2.0",
            "id": req.id,
            "result": {
                "protocolVersion": "2025-11-25",
                "serverInfo": {
                    "name": "Universal Neo4j GraphRAG Server",
                    "version": "0.1.0"
                }
            }
        })),
        "notifications/initialized" => None,
        "tools/list" => Some(json!({
            "jsonrpc": "2.0",
            "id": req.id,
            "result": {
                "tools": [
                    { "name": "read_cypher" },
                    { "name": "write_cypher" },
                    { "name": "get_graph_schema" },
                    { "name": "run_configured_tool" }
                ]
            }
        })),
        "tools/call" => Some(json!({
            "jsonrpc": "2.0",
            "id": req.id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "{\"labels\":[\"Movie\",\"Person\"],\"relationship_types\":[\"ACTED_IN\"]}"
                    }
                ],
                "isError": false
            }
        })),
        "prompts/list" => Some(json!({
            "jsonrpc": "2.0",
            "id": req.id,
            "result": {
                "prompts": [
                    { "name": "system_instructions" },
                    { "name": "get_prompt_by_name" }
                ]
            }
        })),
        "prompts/get" => Some(json!({
            "jsonrpc": "2.0",
            "id": req.id,
            "result": {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": "Jsi expertní databázový asistent pro Neo4j."
                        }
                    }
                ]
            }
        })),
        "resources/read" => Some(json!({
            "jsonrpc": "2.0",
            "id": req.id,
            "result": {
                "contents": [
                    {
                        "uri": "node://Movie/1",
                        "mimeType": "application/json",
                        "text": "[{\"props\":{\"title\":\"The Matrix\"}}]"
                    }
                ]
            }
        })),
        _ => None,
    };

    if let Some(body) = resp {
        let _ = tx.send(body.to_string()).await;
    }

    Ok(StatusCode::ACCEPTED)
}

#[tokio::test]
async fn test_run_graphrag_client_checks_over_sse() {
    let state = MockBridgeState {
        sessions: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/sse", get(handle_mock_sse))
        .route("/messages", post(handle_mock_messages))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");

    let server_task = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let sse_url = format!("http://{addr}/sse");
    let summary = timeout(Duration::from_secs(10), run_graphrag_client_checks(&sse_url))
        .await
        .expect("client checks timed out")
        .expect("client checks failed");

    assert!(summary.tools.contains(&"get_graph_schema".to_string()));
    assert!(
        summary
            .schema_output
            .as_deref()
            .unwrap_or("")
            .contains("\"Movie\"")
    );
    assert!(summary.prompts.contains(&"system_instructions".to_string()));
    assert!(
        summary
            .system_instructions_output
            .as_deref()
            .unwrap_or("")
            .contains("Neo4j")
    );
    assert!(
        summary
            .resource_output
            .as_deref()
            .unwrap_or("")
            .contains("The Matrix")
    );

    server_task.abort();
}
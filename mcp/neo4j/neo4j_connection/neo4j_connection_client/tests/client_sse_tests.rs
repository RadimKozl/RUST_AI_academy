use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use neo4j_connection_client::{
    fetch_graph_statistics_over_sse, format_tool_result, parse_sse_event_block,
    resolve_endpoint_url,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex};
use tokio::time::timeout;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;
use uuid::Uuid;

#[test]
fn test_parse_sse_event_block_with_event_and_multiline_data() {
    let raw = ": comment line\r\nevent: endpoint\r\ndata: /messages?sessionId=abc-123\r\n";
    let parsed = parse_sse_event_block(raw);
    assert_eq!(
        parsed,
        Some((
            "endpoint".to_string(),
            "/messages?sessionId=abc-123".to_string()
        ))
    );

    let default_event = parse_sse_event_block("data: line1\ndata: line2");
    assert_eq!(
        default_event,
        Some(("message".to_string(), "line1\nline2".to_string()))
    );

    assert_eq!(parse_sse_event_block(": only comment\n\n"), None);
}

#[test]
fn test_resolve_endpoint_url() {
    assert_eq!(
        resolve_endpoint_url("http://localhost:8000", "/messages?sessionId=1"),
        "http://localhost:8000/messages?sessionId=1"
    );
    assert_eq!(
        resolve_endpoint_url("http://localhost:8000/", "messages?sessionId=1"),
        "http://localhost:8000/messages?sessionId=1"
    );
    assert_eq!(
        resolve_endpoint_url("http://localhost:8000", "http://127.0.0.1:9000/messages?sessionId=2"),
        "http://127.0.0.1:9000/messages?sessionId=2"
    );
}

#[test]
fn test_format_tool_result_formats_json_text() {
    let payload = json!({
        "content": [
            {
                "type": "text",
                "text": "{\"nodes\":1132,\"relationships\":17694}"
            }
        ],
        "isError": false
    });

    let formatted = format_tool_result(&payload).expect("format_tool_result should succeed");
    assert!(formatted.contains("\"nodes\": 1132"));
    assert!(formatted.contains("\"relationships\": 17694"));
}

type MockSessions = Arc<Mutex<HashMap<String, mpsc::Sender<String>>>>;

#[derive(Clone)]
struct MockServerState {
    sessions: MockSessions,
}

#[derive(Deserialize)]
struct MockQuery {
    #[serde(rename = "sessionId")]
    session_id: Option<String>,
}

#[derive(Deserialize)]
struct MockJsonRpcRequest {
    id: Option<Value>,
    method: String,
}

async fn mock_sse(
    State(state): State<MockServerState>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let session_id = Uuid::new_v4().to_string();
    let (tx, rx) = mpsc::channel::<String>(16);

    {
        let mut guard = state.sessions.lock().await;
        guard.insert(session_id.clone(), tx);
    }

    let endpoint = format!("/messages?sessionId={session_id}");
    let first = Event::default().event("endpoint").data(endpoint);
    let stream = tokio_stream::once(Ok(first)).chain(ReceiverStream::new(rx).map(|msg| {
        Ok(Event::default().event("message").data(msg))
    }));

    Sse::new(stream)
}

async fn mock_messages(
    State(state): State<MockServerState>,
    Query(q): Query<MockQuery>,
    Json(req): Json<MockJsonRpcRequest>,
) -> Result<StatusCode, StatusCode> {
    let sid = q.session_id.ok_or(StatusCode::BAD_REQUEST)?;
    let tx = {
        let guard = state.sessions.lock().await;
        guard.get(&sid).cloned().ok_or(StatusCode::NOT_FOUND)?
    };

    match req.method.as_str() {
        "initialize" => {
            let resp = json!({
                "jsonrpc": "2.0",
                "id": req.id,
                "result": {
                    "protocolVersion": "2025-11-25",
                    "serverInfo": {
                        "name": "mock-movies-server",
                        "version": "0.1.0"
                    }
                }
            });
            let _ = tx.send(resp.to_string()).await;
            Ok(StatusCode::ACCEPTED)
        }
        "notifications/initialized" => Ok(StatusCode::ACCEPTED),
        "tools/call" => {
            let resp = json!({
                "jsonrpc": "2.0",
                "id": req.id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": "{\"nodes\":42,\"relationships\":108}"
                        }
                    ],
                    "isError": false
                }
            });
            let _ = tx.send(resp.to_string()).await;
            Ok(StatusCode::ACCEPTED)
        }
        _ => Ok(StatusCode::ACCEPTED),
    }
}

#[tokio::test]
async fn test_fetch_graph_statistics_over_sse_end_to_end() {
    let mock_state = MockServerState {
        sessions: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/sse", get(mock_sse))
        .route("/messages", post(mock_messages))
        .with_state(mock_state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");

    let server_handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let sse_url = format!("http://{addr}/sse");
    let result = timeout(
        Duration::from_secs(10),
        fetch_graph_statistics_over_sse(&sse_url),
    )
    .await
    .expect("test should not time out")
    .expect("fetch_graph_statistics_over_sse should succeed");

    let formatted = format_tool_result(&result).expect("format_tool_result should succeed");
    assert!(formatted.contains("\"nodes\": 42"));
    assert!(formatted.contains("\"relationships\": 108"));

    server_handle.abort();
}
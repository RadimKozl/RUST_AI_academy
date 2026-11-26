use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use futures_util::StreamExt;
use neo4rs::{ConfigBuilder, Graph};
use serde_json::json;
use tokio::time::timeout;
use tower::ServiceExt;
use universal_mcp_server::{
    build_router, determine_read_only_mode, dispatch_json_rpc, expand_env_placeholders,
    expand_json_env_vars, load_server_config_from_path, resolve_config_path, GraphRagAppState,
    ServerConfig, ServerError,
};

fn temp_test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "universal_mcp_server_test_{name}_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

async fn build_unconnected_graph() -> Arc<Graph> {
    let cfg = ConfigBuilder::new()
        .uri("bolt://127.0.0.1:7687")
        .user("neo4j")
        .password("password")
        .db("neo4j")
        .build()
        .expect("valid Neo4j config");

    let graph = Graph::connect(cfg).await.expect("construct Graph");
    Arc::new(graph)
}

#[test]
fn test_env_placeholder_and_json_expansion() {
    let mut vars = HashMap::new();
    vars.insert("NEO4J_URI".to_string(), "bolt://db.internal:7687".to_string());
    vars.insert("EMPTY_VAR".to_string(), "".to_string());

    let lookup = |k: &str| vars.get(k).cloned();

    assert_eq!(
        expand_env_placeholders("${NEO4J_URI:-bolt://localhost:7687}", &lookup),
        "bolt://db.internal:7687"
    );
    assert_eq!(
        expand_env_placeholders("${MISSING_VAR:-default_val}", &lookup),
        "default_val"
    );
    assert_eq!(
        expand_env_placeholders("${EMPTY_VAR:-fallback_when_empty}", &lookup),
        "fallback_when_empty"
    );

    let raw_json = json!({
        "neo4j": {
            "uri": "${NEO4J_URI:-bolt://localhost:7687}",
            "database": "${NEO4J_DATABASE:-movies}"
        },
        "tags": ["${EMPTY_VAR:-a}", "b"]
    });

    let expanded = expand_json_env_vars(raw_json, &lookup);
    assert_eq!(expanded["neo4j"]["uri"], "bolt://db.internal:7687");
    assert_eq!(expanded["neo4j"]["database"], "movies");
    assert_eq!(expanded["tags"][0], "a");
}

#[test]
fn test_resolve_config_path_and_load_config() {
    let base_dir = temp_test_dir("cfg");
    assert_eq!(
        resolve_config_path(&base_dir, "./config.json"),
        base_dir.join("config.json")
    );
    assert_eq!(
        resolve_config_path(&base_dir, "custom.json"),
        base_dir.join("custom.json")
    );

    let abs_path = Path::new("C:\\custom_dir\\my_config.json");
    assert_eq!(resolve_config_path(&base_dir, "C:\\custom_dir\\my_config.json"), abs_path);

    let missing = load_server_config_from_path(&base_dir.join("nonexistent.json"))
        .expect("missing config returns defaults");
    assert_eq!(missing.mcp_transport, "sse");
    assert!(!missing.read_only);

    let cfg_path = base_dir.join("config.json");
    let cfg_json = json!({
        "mcp_transport": "sse",
        "read_only": false,
        "neo4j": {
            "uri": "bolt://127.0.0.1:7687",
            "username": "neo4j",
            "password": "secret",
            "database": "moviedatabase"
        },
        "agent_instructions": "Test instructions",
        "prompts": [
            {
                "name": "analyze_graph_schema",
                "description": "Schema check",
                "template": "Analyze labels and relations"
            }
        ],
        "custom_tools": [
            {
                "name": "find_nodes_by_property",
                "description": "Search nodes",
                "cypher_query": "MATCH (n) RETURN n LIMIT $limit",
                "parameters": ["limit"],
                "write": false
            }
        ]
    });
    fs::write(&cfg_path, serde_json::to_string_pretty(&cfg_json).unwrap()).expect("write config");

    let loaded = load_server_config_from_path(&cfg_path).expect("load valid config");
    assert_eq!(loaded.neo4j.database, "moviedatabase");
    assert_eq!(loaded.prompts.len(), 1);
    assert_eq!(loaded.custom_tools.len(), 1);

    let _ = fs::remove_dir_all(&base_dir);
}

#[test]
fn test_determine_read_only_mode_precedence() {
    assert!(!determine_read_only_mode(false, false));
    assert!(determine_read_only_mode(true, false));
    assert!(determine_read_only_mode(false, true));
}

#[tokio::test]
async fn test_read_only_violation_without_db() {
    let graph = build_unconnected_graph().await;
    let state = GraphRagAppState::new(graph, ServerConfig::default(), true);

    let err = state
        .execute_write_cypher("CREATE (:TestNode {id: '1'})", None)
        .await
        .expect_err("write should fail in read_only mode");
    assert!(matches!(err, ServerError::ReadOnlyViolation));
}

#[tokio::test]
async fn test_dispatch_json_rpc_initialize_tools_and_prompts() {
    let graph = build_unconnected_graph().await;
    let mut cfg = ServerConfig::default();
    cfg.agent_instructions = "Custom Neo4j assistant".to_string();

    let state = GraphRagAppState::new(graph, cfg, false);

    let init_resp = dispatch_json_rpc(
        &state,
        serde_json::from_value(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }))
        .unwrap(),
    )
    .await
    .expect("initialize response");
    let init_val = serde_json::to_value(&init_resp).unwrap();
    assert_eq!(init_val["result"]["protocolVersion"], "2025-11-25");
    assert_eq!(init_val["result"]["instructions"], "Custom Neo4j assistant");

    let tools_resp = dispatch_json_rpc(
        &state,
        serde_json::from_value(json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }))
        .unwrap(),
    )
    .await
    .expect("tools/list response");
    let tools_val = serde_json::to_value(&tools_resp).unwrap();
    let tool_names: Vec<&str> = tools_val["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    assert!(tool_names.contains(&"read_cypher"));
    assert!(tool_names.contains(&"write_cypher"));
    assert!(tool_names.contains(&"get_graph_schema"));
    assert!(tool_names.contains(&"run_configured_tool"));

    let prompt_resp = dispatch_json_rpc(
        &state,
        serde_json::from_value(json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "prompts/get",
            "params": { "name": "system_instructions" }
        }))
        .unwrap(),
    )
    .await
    .expect("prompts/get response");
    let prompt_val = serde_json::to_value(&prompt_resp).unwrap();
    assert_eq!(
        prompt_val["result"]["messages"][0]["content"]["text"],
        "Custom Neo4j assistant"
    );
}

#[tokio::test]
async fn test_sse_and_messages_session_validation() {
    let graph = build_unconnected_graph().await;
    let state = GraphRagAppState::new(graph, ServerConfig::default(), false);
    let app = build_router(state);

    let sse_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/sse")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("GET /sse");
    assert_eq!(sse_resp.status(), StatusCode::OK);

    let mut body_stream = sse_resp.into_body().into_data_stream();
    let first_chunk = timeout(Duration::from_secs(3), body_stream.next())
        .await
        .expect("SSE first event timeout")
        .expect("SSE stream item")
        .expect("SSE chunk bytes");
    let first_str = String::from_utf8_lossy(&first_chunk);
    assert!(first_str.contains("event: endpoint"));
    assert!(first_str.contains("/messages?sessionId="));

    let missing_sid_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/messages")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("POST /messages without sessionId");
    assert_eq!(missing_sid_resp.status(), StatusCode::BAD_REQUEST);

    let unknown_sid_resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/messages?sessionId=nonexistent-session")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("POST /messages with unknown sessionId");
    assert_eq!(unknown_sid_resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
#[ignore = "Requires running Neo4j instance configured via NEO4J_* environment variables"]
async fn test_server_neo4j_schema_and_node_resource() {
    let _ = dotenvy::dotenv();

    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".into());
    let username = std::env::var("NEO4J_USERNAME").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".into());
    let database = std::env::var("NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".into());

    let mut cfg = ServerConfig::default();
    cfg.neo4j.uri = uri;
    cfg.neo4j.username = username;
    cfg.neo4j.password = password;
    cfg.neo4j.database = database;

    let neo4j_cfg = ConfigBuilder::new()
        .uri(&cfg.neo4j.uri)
        .user(&cfg.neo4j.username)
        .password(&cfg.neo4j.password)
        .db(cfg.neo4j.database.as_str())
        .build()
        .expect("valid Neo4j config");

    let graph = Arc::new(
        timeout(Duration::from_secs(10), Graph::connect(neo4j_cfg))
            .await
            .expect("connect timeout")
            .expect("connect to Neo4j"),
    );

    let state = GraphRagAppState::new(graph, cfg, true);
    let schema = timeout(Duration::from_secs(10), state.get_graph_schema())
        .await
        .expect("schema timeout")
        .expect("get_graph_schema");
    assert!(schema.get("statistics").is_some());

    let node_txt = timeout(Duration::from_secs(10), state.read_node_resource("Movie", "1"))
        .await
        .expect("node resource timeout")
        .expect("read_node_resource");
    assert!(!node_txt.is_empty());
}
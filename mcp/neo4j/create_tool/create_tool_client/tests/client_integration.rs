use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use create_tool_client::{
    call_tool_json, connect_stdio_server, default_server_binary_path, list_tools,
};
use serde_json::{json, Map, Value};
use tokio::time::timeout;

fn args_object(value: Value) -> Option<Map<String, Value>> {
    value.as_object().cloned()
}

#[test]
fn test_default_server_binary_path_respects_env_override() {
    let custom_bin = PathBuf::from("/tmp/custom_create_tool_server");
    // SAFETY: Single-threaded unit test modifying a test-scoped environment variable.
    unsafe {
        env::set_var("MCP_SERVER_BIN", &custom_bin);
    }

    let resolved = default_server_binary_path();
    assert_eq!(resolved, custom_bin);

    unsafe {
        env::remove_var("MCP_SERVER_BIN");
    }
}

#[tokio::test]
async fn test_connect_stdio_server_fails_fast_on_missing_binary() {
    let missing_path = PathBuf::from("/nonexistent/path/to/create_tool_server_bin");
    let start = Instant::now();

    let res = connect_stdio_server(&missing_path, Duration::from_secs(2)).await;
    let elapsed = start.elapsed();

    assert!(res.is_err(), "Expected error when spawning missing binary");
    assert!(
        elapsed < Duration::from_secs(5),
        "Connection attempt took too long: {elapsed:?}"
    );
}

#[tokio::test]
#[ignore = "Requires compiled create_tool_server binary and reachable Neo4j instance"]
async fn test_end_to_end_client_against_server() -> Result<(), Box<dyn std::error::Error>> {
    let server_bin = default_server_binary_path();
    assert!(
        server_bin.exists(),
        "Server binary not found at {}. Build `create_tool_server` first.",
        server_bin.display()
    );

    let client = connect_stdio_server(&server_bin, Duration::from_secs(15)).await?;

    let tools = list_tools(&client).await?;
    assert!(tools.iter().any(|t| t.name == "graph_statistics"));
    assert!(tools.iter().any(|t| t.name == "get_movies_by_genre"));

    let stats = call_tool_json(&client, "graph_statistics", None, Duration::from_secs(10)).await?;
    assert!(!stats.content.is_empty());

    let genre_args = args_object(json!({ "genre": "Action", "limit": 5 }));
    let movies = timeout(
        Duration::from_secs(10),
        call_tool_json(&client, "get_movies_by_genre", genre_args, Duration::from_secs(10)),
    )
    .await??;
    assert!(!movies.content.is_empty());

    client.cancel().await?;
    Ok(())
}
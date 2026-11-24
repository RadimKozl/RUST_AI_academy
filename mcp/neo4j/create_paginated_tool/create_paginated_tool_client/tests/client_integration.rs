use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use create_paginated_tool_client::{
    call_tool_json, connect_stdio_server, default_server_binary_path, list_resource_templates,
    list_tools, read_movie_resource,
};
use serde_json::{json, Map, Value};
use tokio::time::timeout;

fn args_object(value: Value) -> Option<Map<String, Value>> {
    value.as_object().cloned()
}

#[test]
fn test_default_server_binary_path_respects_env_override() {
    let custom_bin = PathBuf::from("/tmp/custom_create_paginated_tool_server");
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
    let missing_path = PathBuf::from("/nonexistent/path/to/create_paginated_tool_server_bin");
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
#[ignore = "Requires compiled create_paginated_tool_server binary and reachable Neo4j instance"]
async fn test_end_to_end_client_against_server() -> Result<(), Box<dyn std::error::Error>> {
    let server_bin = default_server_binary_path();
    assert!(
        server_bin.exists(),
        "Server binary not found at {}. Build `create_paginated_tool_server` first.",
        server_bin.display()
    );

    let client = connect_stdio_server(&server_bin, Duration::from_secs(15)).await?;

    let tools = list_tools(&client).await?;
    assert!(tools.iter().any(|t| t.name == "list_movies_by_genre"));

    let templates = list_resource_templates(&client).await?;
    assert!(templates.iter().any(|t| t.uri_template == "movie://{tmdb_id}"));

    let paged_args = args_object(json!({ "genre": "Action", "page_size": 3, "cursor": 0 }));
    let paged = timeout(
        Duration::from_secs(10),
        call_tool_json(&client, "list_movies_by_genre", paged_args, Duration::from_secs(10)),
    )
    .await??;
    assert!(!paged.content.is_empty());

    let movie_res = read_movie_resource(&client, "16420", Duration::from_secs(10)).await?;
    assert!(!movie_res.is_empty());

    client.cancel().await?;
    Ok(())
}
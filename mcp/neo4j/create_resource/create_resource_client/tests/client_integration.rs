use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use create_resource_client::{
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
    let custom_bin = PathBuf::from("/tmp/custom_create_resource_server");
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
    let missing_path = PathBuf::from("/nonexistent/path/to/create_resource_server_bin");
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
async fn test_connect_stdio_server_times_out_on_hanging_process() {
    let temp_dir = env::temp_dir().join("rust_ai_academy_resource_mcp_hang_test");
    fs::create_dir_all(&temp_dir).expect("Failed to create temp test dir");
    let _script_path = temp_dir.join("hang_server.sh");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::write(&_script_path, "#!/bin/sh\nsleep 10\n").expect("Failed to write mock script");
        let mut perms = fs::metadata(&_script_path)
            .expect("Failed to read script metadata")
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&_script_path, perms).expect("Failed to set executable mode");

        let start = Instant::now();
        let res = connect_stdio_server(&_script_path, Duration::from_millis(300)).await;
        let elapsed = start.elapsed();

        assert!(res.is_err(), "Expected handshake timeout error");
        assert!(
            elapsed < Duration::from_secs(2),
            "Timeout did not trigger promptly: {elapsed:?}"
        );
    }

    let _ = fs::remove_dir_all(&temp_dir);
}

/// Full end-to-end test against compiled `create_resource_server` binary and Neo4j.
/// Run with:
/// `cargo test --test client_integration -- --ignored --nocapture`
#[tokio::test]
#[ignore = "Requires compiled create_resource_server binary and reachable Neo4j instance"]
async fn test_end_to_end_client_against_server() -> Result<(), Box<dyn std::error::Error>> {
    let server_bin = default_server_binary_path();
    assert!(
        server_bin.exists(),
        "Server binary not found at {}. Build `create_resource_server` first.",
        server_bin.display()
    );

    let client = connect_stdio_server(&server_bin, Duration::from_secs(15)).await?;

    let tools = list_tools(&client).await?;
    assert!(
        tools.iter().any(|t| t.name == "graph_statistics"),
        "Expected 'graph_statistics' tool to be registered"
    );
    assert!(
        tools.iter().any(|t| t.name == "get_movies_by_genre"),
        "Expected 'get_movies_by_genre' tool to be registered"
    );

    let stats = call_tool_json(&client, "graph_statistics", None, Duration::from_secs(10)).await?;
    assert!(
        !stats.content.is_empty(),
        "Expected non-empty graph_statistics result"
    );

    let genre_args = args_object(json!({ "genre": "Sci-Fi", "limit": 3 }));
    let movies = call_tool_json(
        &client,
        "get_movies_by_genre",
        genre_args,
        Duration::from_secs(10),
    )
    .await?;
    assert!(
        !movies.content.is_empty(),
        "Expected non-empty get_movies_by_genre result"
    );

    let templates = list_resource_templates(&client).await?;
    assert!(
        templates
            .iter()
            .any(|tpl| tpl.uri_template == "movie://{tmdb_id}"),
        "Expected 'movie://{{tmdb_id}}' resource template"
    );

    let movie_res = timeout(
        Duration::from_secs(10),
        read_movie_resource(&client, "603", Duration::from_secs(10)),
    )
    .await??;
    assert!(
        !movie_res.contents.is_empty(),
        "Expected non-empty resource contents for movie://603"
    );

    client.cancel().await?;
    Ok(())
}
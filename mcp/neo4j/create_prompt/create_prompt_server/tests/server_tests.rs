use std::sync::Arc;
use std::time::Duration;

use create_prompt_server::{AppState, MoviesGraphMcpServer};
use neo4rs::Graph;
use rmcp::{
    model::{
        CallToolRequestParams, GetPromptRequestParams, ReadResourceRequestParams,
    },
    ServiceExt, ServerHandler,
};
use serde::de::DeserializeOwned;
use serde_json::{json, Map, Value};
use tokio::io::duplex;
use tokio::time::timeout;

fn args_map(val: Value) -> Option<Map<String, Value>> {
    val.as_object().cloned()
}

fn build_params<T: DeserializeOwned>(val: Value) -> T {
    serde_json::from_value(val).expect("Failed to deserialize MCP request params")
}

/// Builds a `MoviesGraphMcpServer` connected to Neo4j when `NEO4J_URI` (or default bolt) is reachable.
async fn try_connect_server() -> Result<MoviesGraphMcpServer, Box<dyn std::error::Error>> {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".into());
    let username = std::env::var("NEO4J_USERNAME").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".into());
    let database = std::env::var("NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".into());

    let graph = Arc::new(Graph::new(&uri, &username, &password).await?);
    let state = Arc::new(AppState { graph, database });
    Ok(MoviesGraphMcpServer::new(state))
}

#[test]
fn test_server_info_capabilities() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to build Tokio runtime");

    rt.block_on(async {
        let Ok(server) = try_connect_server().await else {
            let _ = std::any::type_name::<MoviesGraphMcpServer>();
            return;
        };

        let info = server.get_info();
        assert!(
            info.capabilities.tools.is_some(),
            "Tools capability should be enabled"
        );
        assert!(
            info.capabilities.resources.is_some(),
            "Resources capability should be enabled"
        );
        assert!(
            info.capabilities.prompts.is_some(),
            "Prompts capability should be enabled"
        );
        assert!(
            info.instructions
                .as_deref()
                .unwrap_or_default()
                .contains("Movies GraphRAG"),
            "Expected server instructions to mention Movies GraphRAG"
        );
    });
}

#[tokio::test]
async fn test_prompts_and_validation_over_duplex() {
    let Ok(server) = try_connect_server().await else {
        return;
    };

    let (server_io, client_io) = duplex(8192);
    let server_handle = tokio::spawn(async move {
        if let Ok(running) = server.serve(server_io).await {
            let _ = running.waiting().await;
        }
    });

    let client = timeout(Duration::from_secs(5), ().serve(client_io))
        .await
        .expect("Handshake timed out")
        .expect("Client serve failed");

    let prompts = client.list_all_prompts().await.expect("list_all_prompts");
    assert!(prompts.iter().any(|p| p.name == "movie_night_planner"));
    assert!(prompts.iter().any(|p| p.name == "discovery_prompt"));
    assert!(prompts.iter().any(|p| p.name == "analyze_preferences"));

    let planner = client
        .get_prompt(build_params::<GetPromptRequestParams>(json!({
            "name": "movie_night_planner",
            "arguments": null
        })))
        .await
        .expect("movie_night_planner should succeed");
    let _ = planner;

    let discovery = client
        .get_prompt(build_params::<GetPromptRequestParams>(json!({
            "name": "discovery_prompt",
            "arguments": args_map(json!({
                "genre": "Sci-Fi",
                "decade": "1990",
                "mood": "dark"
            }))
        })))
        .await
        .expect("discovery_prompt should succeed");
    let _ = discovery;

    let prefs = client
        .get_prompt(build_params::<GetPromptRequestParams>(json!({
            "name": "analyze_preferences",
            "arguments": args_map(json!({
                "favorite_movies": "The Matrix, Blade Runner 2049"
            }))
        })))
        .await
        .expect("analyze_preferences should succeed");
    let _ = prefs;

    let err_genre = client
        .call_tool(build_params::<CallToolRequestParams>(json!({
            "name": "get_movies_by_genre",
            "arguments": args_map(json!({ "genre": "   ", "limit": 5 }))
        })))
        .await;
    assert!(
        err_genre.is_err(),
        "Expected invalid_params error for empty genre"
    );

    let err_paged_genre = client
        .call_tool(build_params::<CallToolRequestParams>(json!({
            "name": "list_movies_by_genre",
            "arguments": args_map(json!({ "genre": "", "page_size": 10, "cursor": 0 }))
        })))
        .await;
    assert!(
        err_paged_genre.is_err(),
        "Expected invalid_params error for empty genre in list_movies_by_genre"
    );

    let _ = client.cancel().await;
    server_handle.abort();
}

/// Requires a running Neo4j instance with the Movies dataset.
/// Run with:
/// `cargo test --test server_tests -- --ignored --nocapture`
#[tokio::test]
#[ignore = "Requires running Neo4j instance with Movies data"]
async fn test_server_tools_and_resources_against_neo4j() -> Result<(), Box<dyn std::error::Error>> {
    let server = try_connect_server().await?;
    let (server_io, client_io) = duplex(8192);

    let server_handle = tokio::spawn(async move {
        let running = server.serve(server_io).await?;
        let _ = running.waiting().await;
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    });

    let client = timeout(Duration::from_secs(10), ().serve(client_io)).await??;

    let stats = client
        .call_tool(build_params::<CallToolRequestParams>(json!({
            "name": "graph_statistics",
            "arguments": null
        })))
        .await?;
    let _ = stats;

    let by_genre = client
        .call_tool(build_params::<CallToolRequestParams>(json!({
            "name": "get_movies_by_genre",
            "arguments": args_map(json!({ "genre": "Sci-Fi", "limit": 5 }))
        })))
        .await?;
    let _ = by_genre;

    let paged = client
        .call_tool(build_params::<CallToolRequestParams>(json!({
            "name": "list_movies_by_genre",
            "arguments": args_map(json!({ "genre": "Sci-Fi", "page_size": 2, "cursor": 0 }))
        })))
        .await?;
    let _ = paged;

    let templates = client.list_resource_templates(None).await?;
    let _ = templates;

    let movie = client
        .read_resource(build_params::<ReadResourceRequestParams>(json!({
            "uri": "movie://603"
        })))
        .await?;
    let _ = movie;

    client.cancel().await?;
    server_handle.abort();
    Ok(())
}
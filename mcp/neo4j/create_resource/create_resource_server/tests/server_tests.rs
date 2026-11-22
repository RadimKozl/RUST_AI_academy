use std::sync::Arc;
use std::time::Duration;

use create_resource_server::{
    format_movie_markdown, AppState, CastMember, MovieDetails, MoviesResourceServer,
};
use neo4rs::Graph;
use rmcp::{
    model::{CallToolRequestParams, ReadResourceRequestParams},
    ServerHandler, ServiceExt,
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

async fn try_connect_server() -> Result<MoviesResourceServer, Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "neo4j://127.0.0.1:7687".into());
    let username = std::env::var("NEO4J_USERNAME").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".into());
    let database = std::env::var("NEO4J_DATABASE").unwrap_or_else(|_| "moviedatabase".into());

    let graph = Arc::new(Graph::new(&uri, &username, &password).await?);
    let state = Arc::new(AppState { graph, database });
    Ok(MoviesResourceServer::new(state))
}

#[test]
fn test_format_movie_markdown_renders_sections_and_roles() {
    let details = MovieDetails {
        title: "The Matrix".to_string(),
        released: "1999".to_string(),
        tagline: Some("Welcome to the Real World.".to_string()),
        runtime: "136".to_string(),
        plot: "A computer hacker learns about the true nature of reality.".to_string(),
        genres: vec!["Action".to_string(), "Sci-Fi".to_string()],
        directors: vec!["Lana Wachowski".to_string(), "Lilly Wachowski".to_string()],
        actors: vec![
            CastMember {
                name: "Keanu Reeves".to_string(),
                role: Some("Neo".to_string()),
            },
            CastMember {
                name: "Extra Actor".to_string(),
                role: None,
            },
        ],
    };

    let md = format_movie_markdown(&details);
    assert!(md.starts_with("# The Matrix (1999)\n\n_Welcome to the Real World._"));
    assert!(md.contains("**Runtime:** 136 minutes"));
    assert!(md.contains("**Genres:** Action, Sci-Fi"));
    assert!(md.contains("**Director(s):** Lana Wachowski, Lilly Wachowski"));
    assert!(md.contains("## Plot\nA computer hacker learns about the true nature of reality."));
    assert!(md.contains("## Cast\n- Keanu Reeves as Neo\n- Extra Actor"));
}

#[tokio::test]
async fn test_server_capabilities_and_validation_over_duplex() {
    let Ok(server) = try_connect_server().await else {
        // Fallback path when Neo4j is offline during unit test runs.
        let _ = std::any::type_name::<MoviesResourceServer>();
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

    let (server_io, client_io) = duplex(8192);
    let server_handle = tokio::spawn(async move {
        if let Ok(running) = server.serve(server_io).await {
            let _ = running.waiting().await;
        }
    });

    let client = timeout(Duration::from_secs(5), ().serve(client_io))
        .await
        .expect("MCP handshake timed out")
        .expect("Client serve failed");

    let tools = client.list_all_tools().await.expect("list_all_tools");
    assert!(tools.iter().any(|t| t.name == "graph_statistics"));
    assert!(tools.iter().any(|t| t.name == "get_movies_by_genre"));

    let templates = client
        .list_resource_templates(None)
        .await
        .expect("list_resource_templates");
    assert!(
        templates
            .resource_templates
            .iter()
            .any(|tpl| tpl.uri_template == "movie://{tmdb_id}")
    );

    let empty_genre_res = client
        .call_tool(build_params::<CallToolRequestParams>(json!({
            "name": "get_movies_by_genre",
            "arguments": args_map(json!({ "genre": "   ", "limit": 5 }))
        })))
        .await;
    assert!(
        empty_genre_res.is_err(),
        "Expected invalid_params error for blank genre"
    );

    let bad_scheme_res = client
        .read_resource(build_params::<ReadResourceRequestParams>(json!({
            "uri": "http://example.com/603"
        })))
        .await;
    assert!(
        bad_scheme_res.is_err(),
        "Expected error for unsupported resource URI scheme"
    );

    let _ = client.cancel().await;
    server_handle.abort();
}

/// Requires a running Neo4j instance with the Movies dataset (`moviedatabase`).
/// Run with:
/// `cargo test --test server_tests -- --ignored --nocapture`
#[tokio::test]
#[ignore = "Requires running Neo4j instance with moviedatabase data"]
async fn test_server_tools_and_resource_against_neo4j() -> Result<(), Box<dyn std::error::Error>> {
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
    assert!(!stats.content.is_empty());

    let movies = client
        .call_tool(build_params::<CallToolRequestParams>(json!({
            "name": "get_movies_by_genre",
            "arguments": args_map(json!({ "genre": "Sci-Fi", "limit": 3 }))
        })))
        .await?;
    assert!(!movies.content.is_empty());

    let movie_res = client
        .read_resource(build_params::<ReadResourceRequestParams>(json!({
            "uri": "movie://603"
        })))
        .await?;
    assert!(!movie_res.contents.is_empty());

    client.cancel().await?;
    server_handle.abort();
    Ok(())
}
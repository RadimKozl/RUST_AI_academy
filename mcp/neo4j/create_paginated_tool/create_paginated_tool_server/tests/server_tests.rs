use std::sync::Arc;
use std::time::Duration;

use create_paginated_tool_server::{
    format_movie_markdown, AppState, CastMember, ListMoviesByGenreParams, MovieDetails,
    MoviesPaginatedServer, PaginatedMoviesResponse,
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

async fn try_connect_server() -> Result<MoviesPaginatedServer, Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "neo4j://127.0.0.1:7687".into());
    let username = std::env::var("NEO4J_USERNAME").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".into());
    let database = std::env::var("NEO4J_DATABASE").unwrap_or_else(|_| "moviedatabase".into());

    let graph = Arc::new(Graph::new(&uri, &username, &password).await?);
    let state = Arc::new(AppState { graph, database });
    Ok(MoviesPaginatedServer::new(state))
}

#[test]
fn test_pagination_params_and_markdown_formatting() {
    let params: ListMoviesByGenreParams = serde_json::from_value(json!({
        "genre": "Comedy",
        "page_size": 5,
        "cursor": 10
    }))
    .expect("deserialize ListMoviesByGenreParams");
    assert_eq!(params.genre, "Comedy");
    assert_eq!(params.page_size, Some(5));
    assert_eq!(params.cursor, Some(10));

    let response = PaginatedMoviesResponse {
        genre: "Comedy".to_string(),
        movies: Vec::new(),
        next_cursor: Some(15),
        page: 3,
        page_size: 5,
        has_more: true,
    };
    let encoded = serde_json::to_string(&response).expect("serialize PaginatedMoviesResponse");
    assert!(encoded.contains("\"next_cursor\":15"));

    let details = MovieDetails {
        title: "The Matrix".to_string(),
        released: "1999".to_string(),
        tagline: Some("Welcome to the Real World".to_string()),
        runtime: "136".to_string(),
        plot: "A computer hacker learns about the true nature of reality.".to_string(),
        genres: vec!["Action".to_string(), "Sci-Fi".to_string()],
        directors: vec!["Lana Wachowski".to_string()],
        actors: vec![CastMember {
            name: "Keanu Reeves".to_string(),
            role: Some("Neo".to_string()),
        }],
    };
    let md = format_movie_markdown(&details);
    assert!(md.contains("# The Matrix (1999)"));
    assert!(md.contains("- Keanu Reeves as Neo"));
}

#[tokio::test]
async fn test_tools_and_resources_over_duplex() {
    let Ok(server) = try_connect_server().await else {
        return;
    };

    let info = server.get_info();
    assert!(info.capabilities.tools.is_some());
    assert!(info.capabilities.resources.is_some());

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
    assert!(tools.iter().any(|t| t.name == "list_movies_by_genre"));

    let templates = client
        .list_resource_templates(None)
        .await
        .expect("list_resource_templates");
    assert!(templates.resource_templates.iter().any(|t| t.uri_template == "movie://{tmdb_id}"));

    let _ = client.cancel().await;
    server_handle.abort();
}

#[tokio::test]
#[ignore = "Requires running Neo4j instance with moviedatabase data"]
async fn test_paginated_tool_and_resource_against_neo4j() -> Result<(), Box<dyn std::error::Error>> {
    let server = try_connect_server().await?;
    let (server_io, client_io) = duplex(8192);

    let server_handle = tokio::spawn(async move {
        let running = server.serve(server_io).await?;
        let _ = running.waiting().await;
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    });

    let client = timeout(Duration::from_secs(10), ().serve(client_io)).await??;

    let page1 = client
        .call_tool(build_params::<CallToolRequestParams>(json!({
            "name": "list_movies_by_genre",
            "arguments": args_map(json!({ "genre": "Action", "page_size": 3, "cursor": 0 }))
        })))
        .await?;
    assert!(!page1.content.is_empty());

    let res = client
        .read_resource(build_params::<ReadResourceRequestParams>(json!({
            "uri": "movie://16420"
        })))
        .await?;
    assert!(!res.contents.is_empty());

    client.cancel().await?;
    server_handle.abort();
    Ok(())
}
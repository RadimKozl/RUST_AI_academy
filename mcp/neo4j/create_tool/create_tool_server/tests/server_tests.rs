use std::sync::Arc;
use std::time::Duration;

use create_tool_server::{AppState, GetMoviesByGenreParams, GraphStats, MovieSummary, MoviesToolServer};
use neo4rs::Graph;
use rmcp::{model::CallToolRequestParams, ServerHandler, ServiceExt};
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

async fn try_connect_server() -> Result<MoviesToolServer, Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "neo4j://127.0.0.1:7687".into());
    let username = std::env::var("NEO4J_USERNAME").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".into());
    let database = std::env::var("NEO4J_DATABASE").unwrap_or_else(|_| "moviedatabase".into());

    let graph = Arc::new(Graph::new(&uri, &username, &password).await?);
    let state = Arc::new(AppState { graph, database });
    Ok(MoviesToolServer::new(state))
}

#[test]
fn test_params_and_models_serialization() {
    let parsed: GetMoviesByGenreParams = serde_json::from_value(json!({
        "genre": "Action",
        "limit": 5
    }))
    .expect("deserialize params");
    assert_eq!(parsed.genre, "Action");
    assert_eq!(parsed.limit, Some(5));

    let stats = GraphStats {
        nodes: 1132,
        relationships: 17694,
    };
    let stats_json = serde_json::to_string(&stats).expect("serialize stats");
    assert!(stats_json.contains("1132"));

    let movie = MovieSummary {
        title: Some("The Matrix".to_string()),
        imdb_rating: Some(8.7),
        released: Some(922838400),
    };
    let movie_json = serde_json::to_string(&movie).expect("serialize movie");
    assert!(movie_json.contains("imdbRating"));
}

#[tokio::test]
async fn test_tools_list_and_validation_over_duplex() {
    let Ok(server) = try_connect_server().await else {
        let _ = std::any::type_name::<MoviesToolServer>();
        return;
    };

    let info = server.get_info();
    assert!(info.capabilities.tools.is_some());

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

    let empty_genre_res = client
        .call_tool(build_params::<CallToolRequestParams>(json!({
            "name": "get_movies_by_genre",
            "arguments": args_map(json!({ "genre": "   ", "limit": 5 }))
        })))
        .await;
    assert!(empty_genre_res.is_err());

    let _ = client.cancel().await;
    server_handle.abort();
}

#[tokio::test]
#[ignore = "Requires running Neo4j instance with moviedatabase data"]
async fn test_tools_against_neo4j() -> Result<(), Box<dyn std::error::Error>> {
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
            "arguments": args_map(json!({ "genre": "Action", "limit": 5 }))
        })))
        .await?;
    assert!(!movies.content.is_empty());

    client.cancel().await?;
    server_handle.abort();
    Ok(())
}
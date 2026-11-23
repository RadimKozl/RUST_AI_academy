use std::sync::Arc;

use create_tool_server::{AppState, MoviesToolServer};
use neo4rs::{query, ConfigBuilder, Graph};
use rmcp::{transport::stdio, ServiceExt};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    // Log exclusively to stderr so stdout remains reserved for MCP JSON-RPC frames.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "neo4j://127.0.0.1:7687".into());
    let username = std::env::var("NEO4J_USERNAME").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".into());
    let database = std::env::var("NEO4J_DATABASE").unwrap_or_else(|_| "moviedatabase".into());

    info!(
        uri = %uri,
        username = %username,
        database = %database,
        "Connecting to Neo4j"
    );

    let config = ConfigBuilder::new()
        .uri(&uri)
        .user(&username)
        .password(&password)
        .db(database.as_str())
        .build()?;

    let graph = Arc::new(Graph::connect(config).await?);
    graph.run(query("RETURN 1")).await?;

    let state = Arc::new(AppState { graph, database });
    let server = MoviesToolServer::new(state);

    let running = server.serve(stdio()).await?;
    running.waiting().await?;

    Ok(())
}
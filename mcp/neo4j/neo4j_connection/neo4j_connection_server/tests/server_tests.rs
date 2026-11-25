use std::sync::Arc;
use std::time::Duration;

use neo4j_connection_server::{AppState, GraphStatistics, MoviesGraphServer};
use neo4rs::{query, ConfigBuilder, Graph};
use rmcp::{model::ProtocolVersion, ServerHandler};
use serde_json::Value;
use tokio::time::timeout;

#[tokio::test]
async fn test_server_info_and_capabilities() {
    let config = ConfigBuilder::new()
        .uri("bolt://127.0.0.1:7687")
        .user("neo4j")
        .password("password")
        .db("neo4j")
        .build()
        .expect("valid neo4j config");

    // Graph::connect is async in neo4rs 0.8 and creates the driver/pool lazily.
    let graph = Graph::connect(config)
        .await
        .expect("construct Graph client");
    let state = Arc::new(AppState {
        graph: Arc::new(graph),
        database: "neo4j".to_string(),
    });

    let server = MoviesGraphServer::new(state);
    let info = server.get_info();

    assert_eq!(info.protocol_version, ProtocolVersion::V_2025_11_25);
    assert!(info.capabilities.tools.is_some());
    assert_eq!(info.server_info.name, "Movies GraphRAG Server");
    assert_eq!(
        info.instructions.as_deref(),
        Some("Movies GraphRAG MCP server backed by Neo4j.")
    );
}

#[test]
fn test_graph_statistics_serialization_roundtrip() {
    let stats = GraphStatistics {
        nodes: 1132,
        relationships: 17694,
    };

    let serialized = serde_json::to_string(&stats).expect("serialize GraphStatistics");
    let parsed: Value = serde_json::from_str(&serialized).expect("parse JSON");

    assert_eq!(parsed["nodes"], 1132);
    assert_eq!(parsed["relationships"], 17694);
}

#[tokio::test]
#[ignore = "Requires reachable Neo4j instance configured via NEO4J_* environment variables"]
async fn test_server_query_graph_statistics_against_neo4j() {
    let _ = dotenvy::dotenv();

    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://127.0.0.1:7687".into());
    let username = std::env::var("NEO4J_USERNAME").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".into());
    let database = std::env::var("NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".into());

    let config = ConfigBuilder::new()
        .uri(&uri)
        .user(&username)
        .password(&password)
        .db(database.as_str())
        .build()
        .expect("valid Neo4j config");

    let graph = timeout(Duration::from_secs(10), Graph::connect(config))
        .await
        .expect("connect timeout")
        .expect("connect to Neo4j");

    timeout(Duration::from_secs(10), graph.run(query("RETURN 1")))
        .await
        .expect("ping timeout")
        .expect("Neo4j RETURN 1 check");

    let state = Arc::new(AppState {
        graph: Arc::new(graph),
        database,
    });
    let server = MoviesGraphServer::new(state);

    let stats = timeout(Duration::from_secs(10), server.query_graph_statistics())
        .await
        .expect("query_graph_statistics timeout")
        .expect("query_graph_statistics succeeds");

    assert!(stats.nodes >= 0);
    assert!(stats.relationships >= 0);

    let serialized = serde_json::to_string(&stats).expect("serialize stats");
    let json_val: Value = serde_json::from_str(&serialized).expect("parse serialized stats");
    assert_eq!(json_val["nodes"].as_i64(), Some(stats.nodes));
    assert_eq!(
        json_val["relationships"].as_i64(),
        Some(stats.relationships)
    );
}
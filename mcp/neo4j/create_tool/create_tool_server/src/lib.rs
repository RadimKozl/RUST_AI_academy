use std::sync::Arc;

use neo4rs::{query, Graph};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ErrorData as McpError, InitializeResult, ServerCapabilities},
    schemars, tool, tool_handler, tool_router, ServerHandler,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{error, info, warn};

fn push_text_content<T>(out: &mut Vec<T>, text: String) -> Result<(), McpError>
where
    T: DeserializeOwned,
{
    let item = serde_json::from_value(json!({
        "type": "text",
        "text": text
    }))
    .map_err(|e| McpError::internal_error(format!("Failed building tool content: {e}"), None))?;
    out.push(item);
    Ok(())
}

#[derive(Clone)]
pub struct AppState {
    pub graph: Arc<Graph>,
    pub database: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct GraphStats {
    pub nodes: i64,
    pub relationships: i64,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct MovieSummary {
    pub title: Option<String>,
    #[serde(rename = "imdbRating")]
    pub imdb_rating: Option<f64>,
    pub released: Option<i64>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetMoviesByGenreParams {
    #[schemars(description = "The genre to search for (e.g., \"Action\", \"Drama\", \"Comedy\")")]
    pub genre: String,
    #[schemars(description = "Maximum number of movies to return (default: 10)")]
    pub limit: Option<i64>,
}

#[derive(Clone)]
pub struct MoviesToolServer {
    state: Arc<AppState>,
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MoviesToolServer {
    pub fn new(state: Arc<AppState>) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Count the number of nodes and relationships in the Neo4j graph.")]
    async fn graph_statistics(&self) -> Result<CallToolResult, McpError> {
        let cypher = r#"RETURN COUNT {()} AS nodes, COUNT {()-[]-()} AS relationships"#;
        let mut result = self
            .state
            .graph
            .execute(query(cypher))
            .await
            .map_err(|e| {
                error!(error = %e, "graph_statistics query failed");
                McpError::internal_error(format!("Neo4j query failed: {e}"), None)
            })?;

        let stats = if let Some(row) = result
            .next()
            .await
            .map_err(|e| McpError::internal_error(format!("Neo4j row read failed: {e}"), None))?
        {
            GraphStats {
                nodes: row.get("nodes").unwrap_or(0),
                relationships: row.get("relationships").unwrap_or(0),
            }
        } else {
            GraphStats {
                nodes: 0,
                relationships: 0,
            }
        };

        let json_str = serde_json::to_string(&stats).map_err(|e| {
            McpError::internal_error(format!("JSON serialization failed: {e}"), None)
        })?;

        let mut out = Vec::new();
        push_text_content(&mut out, json_str)?;
        Ok(CallToolResult::success(out))
    }

    #[tool(description = "Get movies by genre from the Neo4j database ordered by IMDb rating.")]
    async fn get_movies_by_genre(
        &self,
        Parameters(params): Parameters<GetMoviesByGenreParams>,
    ) -> Result<CallToolResult, McpError> {
        let genre = params.genre.trim().to_string();
        if genre.is_empty() {
            return Err(McpError::invalid_params(
                "Missing or empty required argument: 'genre'".to_string(),
                None,
            ));
        }

        let limit = params.limit.unwrap_or(10).clamp(1, 200);
        info!(genre = %genre, limit = limit, "Searching for movies by genre");

        let cypher = r#"
            MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: $genre})
            RETURN m.title AS title,
                   m.imdbRating AS imdbRating,
                   m.released AS released
            ORDER BY coalesce(m.imdbRating, 0) DESC
            LIMIT $limit
        "#;

        let mut result = self
            .state
            .graph
            .execute(query(cypher).param("genre", genre.clone()).param("limit", limit))
            .await
            .map_err(|e| {
                error!(error = %e, genre = %genre, "Query failed for get_movies_by_genre");
                McpError::internal_error(format!("Neo4j query failed: {e}"), None)
            })?;

        let mut movies = Vec::new();
        while let Some(row) = result
            .next()
            .await
            .map_err(|e| McpError::internal_error(format!("Neo4j row read failed: {e}"), None))?
        {
            movies.push(MovieSummary {
                title: row.get("title").ok(),
                imdb_rating: row.get("imdbRating").ok(),
                released: row.get("released").ok(),
            });
        }

        if movies.is_empty() {
            warn!(genre = %genre, "No movies found for genre");
        } else {
            info!(genre = %genre, count = movies.len(), "Found movies for genre");
        }

        let json_str = serde_json::to_string(&movies).map_err(|e| {
            McpError::internal_error(format!("JSON serialization failed: {e}"), None)
        })?;

        let mut out = Vec::new();
        push_text_content(&mut out, json_str)?;
        Ok(CallToolResult::success(out))
    }
}

#[tool_handler]
impl ServerHandler for MoviesToolServer {
    fn get_info(&self) -> InitializeResult {
        let capabilities = ServerCapabilities::builder().enable_tools().build();

        InitializeResult::new(capabilities)
            .with_instructions("RUST_AI_academy Movies GraphRAG MCP server backed by Neo4j.")
    }
}
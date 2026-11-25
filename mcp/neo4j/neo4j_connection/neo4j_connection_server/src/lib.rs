use std::sync::Arc;

use neo4rs::{query, Graph};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolResult, Implementation, ProtocolVersion, ServerCapabilities, ServerConfig,
        ToolsCapability,
    },
    schemars::{self, JsonSchema},
    tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;
use tracing::{error, info};

#[derive(Debug, Error)]
pub enum AppError {
    #[error("Neo4j query failed: {0}")]
    Neo4j(#[from] neo4rs::Error),
    #[error("JSON serialization failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone)]
pub struct AppState {
    pub graph: Arc<Graph>,
    pub database: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GraphStatisticsParams {}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GraphStatistics {
    pub nodes: i64,
    pub relationships: i64,
}

#[derive(Clone)]
pub struct MoviesGraphServer {
    state: Arc<AppState>,
    pub tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MoviesGraphServer {
    pub fn new(state: Arc<AppState>) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    pub async fn query_graph_statistics(&self) -> Result<GraphStatistics, AppError> {
        let cypher = query("RETURN COUNT {()} AS nodes, COUNT {()-[]->()} AS relationships");
        let mut result = self.state.graph.execute(cypher).await?;

        if let Some(row) = result.next().await? {
            let nodes: i64 = row.get("nodes").unwrap_or(0);
            let relationships: i64 = row.get("relationships").unwrap_or(0);
            Ok(GraphStatistics {
                nodes,
                relationships,
            })
        } else {
            Ok(GraphStatistics {
                nodes: 0,
                relationships: 0,
            })
        }
    }

    #[tool(description = "Count the number of nodes and relationships in the graph.")]
    async fn graph_statistics(
        &self,
        _params: Parameters<GraphStatisticsParams>,
    ) -> Result<CallToolResult, McpError> {
        info!(database = %self.state.database, "Executing graph_statistics tool");

        let stats = self.query_graph_statistics().await.map_err(|err| {
            error!(error = %err, "Failed to query graph_statistics from Neo4j");
            McpError::internal_error(format!("Database query failed: {err}"), None)
        })?;

        let payload = serde_json::to_string(&stats).map_err(|err| {
            error!(error = %err, "Failed to serialize graph statistics");
            McpError::internal_error(format!("Serialization failed: {err}"), None)
        })?;

        let text_content = serde_json::from_value(json!({
            "type": "text",
            "text": payload
        }))
        .map_err(|err| {
            error!(error = %err, "Failed to build text content item");
            McpError::internal_error(format!("Content construction failed: {err}"), None)
        })?;

        Ok(CallToolResult::success(vec![text_content]))
    }
}

#[tool_handler]
impl ServerHandler for MoviesGraphServer {
    fn get_info(&self) -> ServerConfig {
        let mut capabilities = ServerCapabilities::default();
        let mut tools_cap = ToolsCapability::default();
        tools_cap.list_changed = Some(false);
        capabilities.tools = Some(tools_cap);

        let mut server_info = Implementation::default();
        server_info.name = "Movies GraphRAG Server".to_string();
        server_info.version = env!("CARGO_PKG_VERSION").to_string();

        let mut config = ServerConfig::default();
        config.protocol_version = ProtocolVersion::V_2025_11_25;
        config.capabilities = capabilities;
        config.server_info = server_info;
        config.instructions = Some("Movies GraphRAG MCP server backed by Neo4j.".to_string());
        config
    }
}
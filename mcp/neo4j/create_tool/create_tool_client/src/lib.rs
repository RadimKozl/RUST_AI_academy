use std::path::{Path, PathBuf};
use std::time::Duration;

use rmcp::{
    model::{CallToolRequestParams, CallToolResult, Tool},
    service::RunningService,
    transport::{ConfigureCommandExt, TokioChildProcess},
    RoleClient, ServiceExt,
};
use tokio::process::Command;
use tokio::time::timeout;

pub type McpClientHandle = RunningService<RoleClient, ()>;

pub fn default_server_binary_path() -> PathBuf {
    if let Ok(explicit) = std::env::var("MCP_SERVER_BIN") {
        return PathBuf::from(explicit);
    }

    let release_rel = PathBuf::from("../create_tool_server/target/release/create_tool_server");
    if release_rel.exists() {
        return release_rel;
    }

    PathBuf::from("../create_tool_server/target/debug/create_tool_server")
}

pub async fn connect_stdio_server(
    server_binary: &Path,
    handshake_timeout: Duration,
) -> Result<McpClientHandle, Box<dyn std::error::Error>> {
    let cmd = Command::new(server_binary).configure(|c| {
        for key in [
            "NEO4J_URI",
            "NEO4J_USERNAME",
            "NEO4J_PASSWORD",
            "NEO4J_DATABASE",
        ] {
            if let Ok(val) = std::env::var(key) {
                c.env(key, val);
            }
        }
    });

    let transport = TokioChildProcess::new(cmd)?;
    let client = timeout(handshake_timeout, ().serve(transport))
        .await
        .map_err(|_| "Timed out waiting for MCP server handshake")??;

    Ok(client)
}

pub async fn list_tools(client: &McpClientHandle) -> Result<Vec<Tool>, Box<dyn std::error::Error>> {
    Ok(client.list_all_tools().await?)
}

pub async fn call_tool_json(
    client: &McpClientHandle,
    name: &str,
    arguments: Option<serde_json::Map<String, serde_json::Value>>,
    call_timeout: Duration,
) -> Result<CallToolResult, Box<dyn std::error::Error>> {
    let mut params = CallToolRequestParams::new(name.to_string());
    if let Some(args) = arguments {
        params = params.with_arguments(args);
    }

    let result = timeout(call_timeout, client.call_tool(params))
        .await
        .map_err(|_| format!("Timed out calling tool '{name}'"))??;

    Ok(result)
}
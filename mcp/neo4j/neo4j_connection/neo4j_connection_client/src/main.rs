use anyhow::Result;
use neo4j_connection_client::{fetch_graph_statistics_over_sse, format_tool_result};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let server_url = "http://localhost:8000/sse";
    let tool_result = fetch_graph_statistics_over_sse(server_url).await?;
    let formatted = format_tool_result(&tool_result)?;

    println!("Result from MCP Server:");
    println!("{}", "-".repeat(40));
    println!("{formatted}");
    println!("{}", "-".repeat(40));

    Ok(())
}
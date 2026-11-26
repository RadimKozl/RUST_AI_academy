use anyhow::Result;
use tracing_subscriber::EnvFilter;
use universal_mcp_client::run_graphrag_client_checks;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let server_url =
        std::env::var("MCP_SERVER_URL").unwrap_or_else(|_| "http://localhost:8000/sse".to_string());
    println!("=== I am connecting to the MCP Server via SSE. ({server_url}) ===");

    let summary = run_graphrag_client_checks(&server_url).await?;
    println!("MCP session successfully initialized via SSE.\n");

    println!("--- [TEST 1] Retrieving the list of tools ---");
    println!("Available tools: {:?}\n", summary.tools);

    println!("--- [TEST 1b] Calling the 'get_graph_schema' tool ---");
    println!(
        "Result of the scheme:\n{}\n",
        summary.schema_output.unwrap_or_default()
    );

    println!("--- [TEST 2] Retrieving system instructions (Prompts) ---");
    println!("Available prompts: {:?}", summary.prompts);
    println!(
        "Content system prompt: {}\n",
        summary.system_instructions_output.unwrap_or_default()
    );

    println!("--- [TEST 3] Reading MCP Resource ---");
    let resource_uri =
        std::env::var("MCP_RESOURCE_URI").unwrap_or_else(|_| "node://Movie/11443".to_string());
    println!("Loading resource from URI: {resource_uri}");
    println!(
        "Resource content:\n{}\n",
        summary.resource_output.unwrap_or_default()
    );

    Ok(())
}

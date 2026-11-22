use std::env;
use std::time::Duration;

use create_resource_client::{
    call_tool_json, connect_stdio_server, default_server_binary_path, list_resource_templates,
    list_tools, read_movie_resource,
};
use rmcp::model::ResourceContents;
use serde_json::json;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let tmdb_id = env::args().nth(1).unwrap_or_else(|| "16420".to_string());
    let server_bin = default_server_binary_path();
    info!(binary = %server_bin.display(), tmdb_id = %tmdb_id, "Connecting MCP client to server binary");

    let client = connect_stdio_server(&server_bin, Duration::from_secs(15)).await?;

    let tools = list_tools(&client).await?;
    println!("=== Tools ({}) ===", tools.len());
    for tool in &tools {
        println!(
            "- {}: {}",
            tool.name,
            tool.description.as_deref().unwrap_or("<no description>")
        );
    }

    let stats = call_tool_json(&client, "graph_statistics", None, Duration::from_secs(10)).await?;
    println!("\n[graph_statistics] -> {:?}", stats.content);

    let sci_fi_args = json!({ "genre": "Sci-Fi", "limit": 3 })
        .as_object()
        .cloned();
    let movies = call_tool_json(
        &client,
        "get_movies_by_genre",
        sci_fi_args,
        Duration::from_secs(10),
    )
    .await?;
    println!("\n[get_movies_by_genre] -> {:?}", movies.content);

    let templates = list_resource_templates(&client).await?;
    println!("\n=== Resource Templates ({}) ===", templates.len());
    for tpl in &templates {
        println!("- {} ({})", tpl.name, tpl.uri_template);
    }

    let resource_uri = format!("movie://{tmdb_id}");
    println!("\nFetching resource: {resource_uri}");
    let movie_res = read_movie_resource(&client, &tmdb_id, Duration::from_secs(10)).await?;

    println!("\nResult from the server:");
    println!("{}", "-".repeat(60));
    for content in &movie_res.contents {
        match content {
            ResourceContents::TextResourceContents { text, .. } => println!("{text}"),
            other => println!("{other:?}"),
        }
    }
    println!("{}", "-".repeat(60));

    if let Err(err) = client.cancel().await {
        error!("Failed to cleanly cancel client session: {err}");
    }

    Ok(())
}
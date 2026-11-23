use std::env;
use std::time::Duration;

use create_tool_client::{
    call_tool_json, connect_stdio_server, default_server_binary_path, list_tools,
};
use serde_json::{json, Value};
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

    let genre = env::args().nth(1).unwrap_or_else(|| "Action".to_string());
    let limit: i64 = env::args()
        .nth(2)
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(5);

    let server_bin = default_server_binary_path();
    info!(
        binary = %server_bin.display(),
        genre = %genre,
        limit = limit,
        "Connecting MCP client to server binary"
    );

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

    let genre_args = json!({ "genre": genre, "limit": limit })
        .as_object()
        .cloned();
    let movies = call_tool_json(
        &client,
        "get_movies_by_genre",
        genre_args,
        Duration::from_secs(10),
    )
    .await?;

    println!("\n[get_movies_by_genre] (genre={genre}, limit={limit}):");
    println!("{}", "-".repeat(60));
    for content in &movies.content {
        let val = serde_json::to_value(content)?;
        if let Some(text) = val.get("text").and_then(Value::as_str) {
            if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                println!("{}", serde_json::to_string_pretty(&parsed)?);
            } else {
                println!("{text}");
            }
        } else {
            println!("{val}");
        }
    }
    println!("{}", "-".repeat(60));

    if let Err(err) = client.cancel().await {
        error!("Failed to cleanly cancel client session: {err}");
    }

    Ok(())
}

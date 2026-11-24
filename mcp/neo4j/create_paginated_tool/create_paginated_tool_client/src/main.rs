use std::env;
use std::io::{self, BufRead, Write};
use std::time::Duration;

use create_paginated_tool_client::{
    call_tool_json, connect_stdio_server, default_server_binary_path, list_resource_templates,
    list_tools, read_movie_resource,
};
use rmcp::model::ResourceContents;
use serde_json::{json, Value};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

fn print_tool_result(result: &rmcp::model::CallToolResult) -> Result<(), Box<dyn std::error::Error>> {
    for content in &result.content {
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
    Ok(())
}

async fn run_interactive_pagination(
    client: &create_paginated_tool_client::McpClientHandle,
) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        print!("\n--- Fetch Page ---\ngenre (required) [Enter value]: ");
        io::stdout().flush()?;
        let Some(genre_line) = lines.next() else {
            break;
        };
        let genre = genre_line?.trim().to_string();
        if genre.is_empty() {
            println!("Genre is required!");
            continue;
        }

        print!("page_size (optional, default: 10) [Enter value]: ");
        io::stdout().flush()?;
        let page_size: i64 = lines
            .next()
            .transpose()?
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(10);

        print!("cursor (optional, default: 0) [Enter value]: ");
        io::stdout().flush()?;
        let cursor: i64 = lines
            .next()
            .transpose()?
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0);

        let args = json!({
            "genre": genre,
            "page_size": page_size,
            "cursor": cursor
        })
        .as_object()
        .cloned();

        println!("\nCalling list_movies_by_genre...");
        let res = call_tool_json(client, "list_movies_by_genre", args, Duration::from_secs(10)).await?;
        println!("\nStructured Response:");
        println!("{}", "-".repeat(60));
        print_tool_result(&res)?;
        println!("{}", "-".repeat(60));

        print!("Do you want to fetch another page/genre? (y/n): ");
        io::stdout().flush()?;
        let cont = lines
            .next()
            .transpose()?
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_default();
        if cont != "y" {
            println!("Exiting interactive pagination loop.");
            break;
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let args: Vec<String> = env::args().skip(1).collect();
    let interactive = args.iter().any(|a| a == "--interactive" || a == "-i");
    let positional: Vec<&String> = args
        .iter()
        .filter(|a| *a != "--interactive" && *a != "-i")
        .collect();

    let genre = positional
        .first()
        .map(|s| s.as_str())
        .unwrap_or("Action")
        .to_string();
    let page_size: i64 = positional
        .get(1)
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(5);
    let cursor: i64 = positional
        .get(2)
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(0);
    let tmdb_id = positional.get(3).map(|s| s.as_str()).unwrap_or("16420");

    let server_bin = default_server_binary_path();
    info!(
        binary = %server_bin.display(),
        genre = %genre,
        page_size = page_size,
        cursor = cursor,
        tmdb_id = %tmdb_id,
        interactive = interactive,
        "Connecting MCP client to paginated server binary"
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
    println!("\n[graph_statistics]:");
    print_tool_result(&stats)?;

    let paged_args = json!({
        "genre": genre,
        "page_size": page_size,
        "cursor": cursor
    })
    .as_object()
    .cloned();
    let paged = call_tool_json(
        &client,
        "list_movies_by_genre",
        paged_args,
        Duration::from_secs(10),
    )
    .await?;
    println!("\n[list_movies_by_genre] (genre={genre}, page_size={page_size}, cursor={cursor}):");
    println!("{}", "-".repeat(60));
    print_tool_result(&paged)?;
    println!("{}", "-".repeat(60));

    let templates = list_resource_templates(&client).await?;
    println!("\n=== Resource Templates ({}) ===", templates.len());
    for tpl in &templates {
        println!("- {} ({})", tpl.name, tpl.uri_template);
    }

    let movie_res = read_movie_resource(&client, tmdb_id, Duration::from_secs(10)).await?;
    println!("\n[movie://{tmdb_id}]:");
    println!("{}", "-".repeat(60));
    for content in &movie_res {
        match content {
            ResourceContents::TextResourceContents { text, .. } => println!("{text}"),
            other => println!("{other:?}"),
        }
    }
    println!("{}", "-".repeat(60));

    if interactive {
        run_interactive_pagination(&client).await?;
    }

    if let Err(err) = client.cancel().await {
        error!("Failed to cleanly cancel client session: {err}");
    }

    Ok(())
}
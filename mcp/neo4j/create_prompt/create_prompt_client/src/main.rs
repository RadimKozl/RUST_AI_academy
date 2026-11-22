use std::time::Duration;

use create_prompt_client::{
    call_tool_json, connect_stdio_server, default_server_binary_path, get_prompt_with_args,
    list_prompts, list_resource_templates, list_tools, read_movie_resource,
};
use serde_json::json;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let server_bin = default_server_binary_path();
    info!(binary = %server_bin.display(), "Connecting MCP client to server binary");

    let client = connect_stdio_server(&server_bin, Duration::from_secs(15)).await?;

    if let Some(info_res) = client.peer_info() {
        let server_name = info_res
            .server_info
            .as_ref()
            .map(|s| s.name.as_str())
            .unwrap_or("<unknown server>");
        let server_version = info_res
            .server_info
            .as_ref()
            .map(|s| s.version.as_str())
            .unwrap_or("<unknown version>");
        println!("Connected to {server_name} (v{server_version})");
    }

    let tools = list_tools(&client).await?;
    println!("\n=== Tools ({}) ===", tools.len());
    for tool in &tools {
        println!(
            "- {}: {}",
            tool.name,
            tool.description.as_deref().unwrap_or("<no description>")
        );
    }

    let stats = call_tool_json(&client, "graph_statistics", None, Duration::from_secs(10)).await?;
    println!("\n[graph_statistics] -> {:?}", stats.content);

    let sci_fi_args = json!({ "genre": "Sci-Fi", "limit": 5 })
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

    let movie_doc = read_movie_resource(&client, "603", Duration::from_secs(10)).await?;
    println!("\n[movie://603] -> {:?}", movie_doc.contents);

    let prompts = list_prompts(&client).await?;
    println!("\n=== Prompts ({}) ===", prompts.len());
    for prompt in &prompts {
        println!(
            "- {}: {}",
            prompt.name,
            prompt.description.as_deref().unwrap_or("<no description>")
        );
    }

    let discovery_args = json!({
        "genre": "Sci-Fi",
        "decade": "1990",
        "mood": "dark"
    })
    .as_object()
    .cloned();
    let discovery = get_prompt_with_args(&client, "discovery_prompt", discovery_args).await?;
    println!(
        "\n[discovery_prompt] -> {:?}",
        discovery
    );

    if let Err(err) = client.cancel().await {
        error!("Failed to cleanly cancel client session: {err}");
    }

    Ok(())
}
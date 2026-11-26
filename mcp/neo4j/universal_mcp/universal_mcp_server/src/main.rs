use std::env;
use std::path::PathBuf;

use anyhow::Result;
use tracing_subscriber::EnvFilter;
use universal_mcp_server::{resolve_config_path, run_server_from_config};

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli_read_only = env::args().any(|arg| arg == "--read-only");
    let config_env = env::var("CONFIG_FILE").unwrap_or_else(|_| "config.json".to_string());
    let exe_dir = env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    let config_path = resolve_config_path(&exe_dir, &config_env);
    run_server_from_config(&config_path, cli_read_only).await
}
use anyhow::Result;
use clap::Parser;
use polars_huggingface_pipeline::{
    compile_and_split_dataset, download_source_files, push_to_huggingface_hub, ExportConfig,
};
use reqwest::Client;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "ETL pipeline: HuggingFace JSON -> Polars Parquet -> HF Hub Target Repo")]
struct Args {
    #[arg(short = 's', long, default_value = "KRadim/rustAI_tutorial_dataset")]
    source_repo: String,

    #[arg(short = 't', long, default_value = "KRadim/rustAI_tutorial_dataset")]
    target_repo: String,

    /// Hugging Face API Token (READ and WRITE permissions)
    #[arg(short = 'k', long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    #[arg(short = 'o', long, default_value = "../datasets")]
    pub output_dir: std::path::PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let hf_token = args.hf_token.unwrap_or_default();

    let config = ExportConfig {
        source_repo: args.source_repo,
        target_repo: args.target_repo,
        hf_token: hf_token.clone(),
        output_dir: PathBuf::from("./huggingface_dataset"),
        filenames: vec![
            "gold_dataset_comma.json".into(),
            "gold_dataset_exclamation_mark.json".into(),
            "gold_dataset_none.json".into(),
            "gold_dataset_period.json".into(),
            "gold_dataset_question_mark.json".into(),
        ],
    };

    let client = Client::new();

    println!("📥 Downloading source files from HF: {}...", config.source_repo);
    let downloaded_files = download_source_files(&client, &config).await?;

    println!("🚀 Executing Polars ETL pipeline...");
    compile_and_split_dataset(&downloaded_files, &config.output_dir)?;

    if !hf_token.is_empty() {
        println!("🔐 Uploading Parquet split files to target HF repo: {}...", config.target_repo);
        push_to_huggingface_hub(&client, &config).await?;
        println!("🔥 Pipeline successful: https://huggingface.co/datasets/{}", config.target_repo);
    } else {
        println!("⚠️ No HF_TOKEN provided. Processed Parquet files saved locally at {:?}", config.output_dir);
    }

    Ok(())
}

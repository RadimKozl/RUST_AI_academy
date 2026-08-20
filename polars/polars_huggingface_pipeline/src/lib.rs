use anyhow::{bail, Context, Result};
use hf_hub::repository::AddSource;
use hf_hub::HFClient;
use polars::prelude::*;
use reqwest::header::AUTHORIZATION;
use reqwest::Client;
use std::fs::File;
use std::path::{Path, PathBuf};

pub struct ExportConfig {
    pub source_repo: String,
    pub target_repo: String,
    pub hf_token: String,
    pub output_dir: PathBuf,
    pub filenames: Vec<String>,
}

pub async fn download_source_files(client: &Client, config: &ExportConfig) -> Result<Vec<PathBuf>> {
    let mut downloaded_paths = Vec::new();
    let cache_dir = config.output_dir.join("raw_cache");
    tokio::fs::create_dir_all(&cache_dir).await?;

    for file_name in &config.filenames {
        let download_url = format!(
            "https://huggingface.co/datasets/{}/resolve/main/gold_datasets_output/{}",
            config.source_repo, file_name
        );

        let local_file_path = cache_dir.join(file_name);

        let mut request = client.get(&download_url);
        if !config.hf_token.is_empty() {
            request = request.header(AUTHORIZATION, format!("Bearer {}", config.hf_token));
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            bail!("Failed to download {}: HTTP {}", file_name, response.status());
        }

        let bytes = response.bytes().await?;

        if bytes.starts_with(b"version https://git-lfs") {
            bail!("Downloaded file {} is a Git LFS pointer, not JSON.", file_name);
        }

        tokio::fs::write(&local_file_path, bytes).await?;
        downloaded_paths.push(local_file_path);
    }

    Ok(downloaded_paths)
}

pub fn compile_and_split_dataset(input_files: &[PathBuf], output_dir: &Path) -> Result<()> {
    let mut dataframes = Vec::new();

    for path in input_files {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file {:?}", path))?;

        let cursor = std::io::Cursor::new(content);

        let df = JsonReader::new(cursor)
            .finish()
            .map_err(anyhow::Error::from)
            .with_context(|| format!("Failed to parse JSON file {:?}", path))?;

        dataframes.push(df.lazy());
    }

    let combined_lazy = concat(&dataframes, UnionArgs::default())
        .map_err(anyhow::Error::from)
        .context("Failed to concatenate LazyFrames")?;

    let mut df = combined_lazy
        .collect()
        .map_err(anyhow::Error::from)
        .context("Failed to collect LazyFrame execution plan")?;

    let total_rows = df.height();
    tracing::info!(target: "etl", "Total rows loaded: {}", total_rows);

    df = df
        .sample_n_literal(total_rows, false, Some(true), Some(42))
        .map_err(anyhow::Error::from)
        .context("Failed to shuffle DataFrame")?;

    let train_end = (total_rows as f64 * 0.80) as usize;
    let valid_end = train_end + (total_rows as f64 * 0.10) as usize;

    let train_df = df.slice(0, train_end);
    let valid_df = df.slice(train_end as i64, valid_end - train_end);
    let test_df = df.slice(valid_end as i64, total_rows - valid_end);

    save_parquet_part(&train_df, &output_dir.join("train"), "train_data.parquet")?;
    save_parquet_part(&valid_df, &output_dir.join("validation"), "validation_data.parquet")?;
    save_parquet_part(&test_df, &output_dir.join("test"), "test_data.parquet")?;

    Ok(())
}

fn save_parquet_part(df: &DataFrame, dir: &Path, file_name: &str) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let file_path = dir.join(file_name);
    let file = File::create(&file_path)?;

    let mut df_to_write = df.clone();
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df_to_write)
        .map_err(anyhow::Error::from)
        .with_context(|| format!("Failed writing Parquet to {:?}", file_path))?;

    tracing::info!(target: "etl", "Saved dataset chunk: {:?}", file_path);
    Ok(())
}

pub async fn push_to_huggingface_hub(_client: &Client, config: &ExportConfig) -> Result<()> {
    upload_dataset_file(
        &config.hf_token,
        &config.target_repo,
        &config.output_dir.join("train/train_data.parquet"),
        "train/train_data.parquet",
    )
    .await?;

    upload_dataset_file(
        &config.hf_token,
        &config.target_repo,
        &config.output_dir.join("validation/validation_data.parquet"),
        "validation/validation_data.parquet",
    )
    .await?;

    upload_dataset_file(
        &config.hf_token,
        &config.target_repo,
        &config.output_dir.join("test/test_data.parquet"),
        "test/test_data.parquet",
    )
    .await?;

    Ok(())
}

pub async fn upload_dataset_file(
    hf_token: &str,
    repo_id: &str,
    local_path: &Path,
    remote_path: &str,
) -> Result<()> {
    // Explicitly pass the token to the client
    let client = if hf_token.is_empty() {
        HFClient::new().context("Failed to build HFClient without token")?
    } else {
        HFClient::builder()
            .token(hf_token.to_string())
            .build()
            .context("Failed to build HFClient with provided token")?
    };

    let parts: Vec<&str> = repo_id.split('/').collect();
    if parts.len() != 2 {
        bail!("Invalid repo_id format. Expected 'owner/repo'");
    }

    let repo = client.dataset(parts[0].to_string(), parts[1].to_string());

    repo.upload_file()
        .source(AddSource::file(local_path))
        .path_in_repo(remote_path)
        .commit_message(&format!("Upload {}", remote_path))
        .send()
        .await
        .context("Failed to upload dataset file via HFClient")?;

    tracing::info!(target: "etl", "Successfully uploaded to HF: {}", remote_path);
    Ok(())
}

pub async fn download_model_files(repo_id: &str) -> Result<PathBuf> {
    let client = HFClient::new().context("Failed to build HFClient")?;
    
    let parts: Vec<&str> = repo_id.split('/').collect();
    let repo = if parts.len() == 2 {
        client.model(parts[0], parts[1])
    } else {
        client.model("openai-community", repo_id)
    };

    let path = repo
        .download_file()
        .filename("config.json")
        .send()
        .await
        .context("Failed to download model config")?;

    Ok(path)
} 
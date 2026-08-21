use anyhow::{bail, Context, Result};
use polars::prelude::*;
use std::fs::File;
use std::path::{Path, PathBuf};

pub struct PipelineConfig {
    pub input_files: Vec<PathBuf>,
    pub output_dir: PathBuf,
    pub train_ratio: f64,
}

pub fn execute_etl_pipeline(config: PipelineConfig) -> Result<()> {
    if config.input_files.is_empty() {
        bail!("No input files provided for ETL processing.");
    }

    let mut dataframes = Vec::new();
    for path in &config.input_files {
        let file = File::open(path).with_context(|| format!("Failed to open file {:?}", path))?;
        let df = JsonReader::new(file)
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
        .context("Failed to execute Polars plan")?;

    let total_rows = df.height();
    if total_rows == 0 {
        bail!("Dataset is empty after concatenation.");
    }

    // Shuffle data with a deterministic seed
    df = df
        .sample_n_literal(total_rows, false, Some(true), Some(42))
        .map_err(anyhow::Error::from)
        .context("Failed to shuffle dataset")?;

    let train_end = (total_rows as f64 * config.train_ratio) as usize;
    let train_df = df.slice(0, train_end);
    let test_df = df.slice(train_end as i64, total_rows - train_end);

    save_parquet(&train_df, &config.output_dir.join("train.parquet"))?;
    save_parquet(&test_df, &config.output_dir.join("test.parquet"))?;

    Ok(())
}

fn save_parquet(df: &DataFrame, destination: &Path) -> Result<()> {
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    let file = File::create(destination)?;
    let mut df_clone = df.clone();
    
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df_clone)
        .map_err(anyhow::Error::from)
        .with_context(|| format!("Failed writing Parquet file {:?}", destination))?;

    Ok(())
}
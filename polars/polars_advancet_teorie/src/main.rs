use anyhow::Result;
use polars_advancet_teorie::{execute_etl_pipeline, PipelineConfig};
use std::path::PathBuf;

fn main() -> Result<()> {
    let config = PipelineConfig {
        input_files: vec![
            PathBuf::from("../datasets/input_1.json"),
            PathBuf::from("../datasets/input_2.json"),
        ],
        output_dir: PathBuf::from("../datasets/data/output"),
        train_ratio: 0.8,
    };

    println!("🚀 Starting Polars ETL Engine...");
    execute_etl_pipeline(config)?;
    println!("✅ ETL Pipeline finished successfully!");

    Ok(())
}
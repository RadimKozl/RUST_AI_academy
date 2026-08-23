use anyhow::Result;
use polars_data_conversion::FeatureExtractor;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    let db_path = "data/market_data.db";
    let db_url = format!("sqlite://{}", db_path);
    let parquet_path = "dataset_features.parquet";

    if !Path::new(db_path).exists() {
        anyhow::bail!("Database file '{}' not found.", db_path);
    }

    println!("Loading data from SQLite and calculating indicators...");
    let df = FeatureExtractor::process_from_db(&db_url).await?;

    println!("Exporting extracted tensor matrix to Parquet...");
    FeatureExtractor::export_to_parquet(&df, parquet_path)?;

    println!("Pipeline successfully executed. Features saved to '{}'", parquet_path);
    Ok(())
}
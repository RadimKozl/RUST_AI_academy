use anyhow::{Context, Result};
use polars::prelude::*;
use sqlx::{sqlite::SqlitePoolOptions, SqlitePool};
use std::fs::File;

pub struct FeatureExtractor;

#[derive(sqlx::FromRow)]
struct TickerRow {
    last_price: f64,
}

impl FeatureExtractor {
    /// Loads data from SQLite database `data/market_data.db` and calculates technical indicators
    pub async fn process_from_db(db_url: &str) -> Result<DataFrame> {
        let pool: SqlitePool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(db_url)
            .await
            .context("Failed to connect to SQLite database")?;

        let rows: Vec<TickerRow> = sqlx::query_as::<_, TickerRow>(
            "SELECT last_price FROM market_tickers ORDER BY id ASC",
        )
        .fetch_all(&pool)
        .await
        .context("Failed to query market_tickers from SQLite")?;

        if rows.is_empty() {
            anyhow::bail!("No data found in market_tickers table.");
        }

        let prices: Vec<f64> = rows.into_iter().map(|r| r.last_price).collect();
        let s_price = Series::new("last_price".into(), prices);

        // New_infer_height with casting Series -> Column
        let df = DataFrame::new_infer_height(vec![s_price.into()])?
            .lazy()
            .with_columns([
                col("last_price")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: 5,
                        min_periods: 1,
                        ..Default::default()
                    })
                    .alias("sma_5"),
                col("last_price")
                    .std(1)
                    .alias("volatility_5"),
            ])
            .with_columns([
                (col("sma_5") + (col("volatility_5") * lit(2.0))).alias("bband_upper"),
                (col("sma_5") - (col("volatility_5") * lit(2.0))).alias("bband_lower"),
            ])
            .collect()?;

        Ok(df)
    }

    /// Extract the numeric columns and write them to a Parquet file
    pub fn export_to_parquet(df: &DataFrame, parquet_path: &str) -> Result<()> {
        let mut feature_df = df.select(["last_price", "sma_5", "bband_upper", "bband_lower"])?;
        
        let mut file = File::create(parquet_path).context("Failed to create Parquet file")?;

        ParquetWriter::new(&mut file)
            .finish(&mut feature_df)
            .map_err(|e| anyhow::anyhow!("Failed to write dataset to Parquet: {}", e))?;

        Ok(())
    }
}
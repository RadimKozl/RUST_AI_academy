use anyhow::{Context, Result};
use polars::prelude::{col, lit, LazyCsvReader, LazyFileListReader, LazyFrame};
use std::path::Path;

/// Loads an e-commerce CSV file and constructs a Polars LazyFrame transformation graph.
pub fn build_ecommerce_lazyframe<P: AsRef<Path>>(file_path: P) -> Result<LazyFrame> {
    let path_str = file_path
        .as_ref()
        .to_str()
        .context("Invalid UTF-8 path string")?;

    // Fix: Convert &str to PlRefPath using .into()
    let df_lazy = LazyCsvReader::new(path_str.into())
        .with_has_header(true)
        .finish()
        .context("Failed to initialize CSV reader")?;

    let processed_lazy = df_lazy.limit(15).with_columns(vec![
        (lit("brand-") + col("brand")).alias("brand2"),
        (col("price") * lit(100.0)).alias("price2"),
        col("price")
            .mean()
            .over([col("category_id")])?
            .alias("avg_price_by_category"),
    ]);

    Ok(processed_lazy)
}

/// Executes the transformation graph and returns the concrete DataFrame.
pub fn process_ecommerce_data<P: AsRef<Path>>(file_path: P) -> Result<polars::prelude::DataFrame> {
    let lazy_df = build_ecommerce_lazyframe(file_path)?;
    let df = lazy_df
        .collect()
        .context("Failed to execute Polars query pipeline")?;
    Ok(df)
}
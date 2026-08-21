use anyhow::Result;
use polars::prelude::*;
use polars_internals::{extract_f64_column, process_advanced_expressions};

fn main() -> Result<()> {
    let df = df!(
        "transaction_id" => &[101, 102, 103, 104, 105, 106],
        "category" => &["A", "A", "B", "B", "A", "B"],
        "amount" => &[100.0, 300.0, 50.0, 150.0, 200.0, 50.0]
    )?;

    println!("--- Input DataFrame ---");
    println!("{}", df);

    let processed_df = process_advanced_expressions(df)?;

    println!("\n--- After applying Window Functions and Filter Context ---");
    println!("{}", processed_df);

    // Extraction pro ML / Tenzory
    let relative_scores = extract_f64_column(&processed_df, "relative_score")?;
    println!("\nExtracted 'relative_score' for ML Tensor: {:?}", relative_scores);

    Ok(())
}

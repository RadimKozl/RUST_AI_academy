use anyhow::Result;
use polars_base_pipeline::process_ecommerce_data;
use std::path::Path;

fn main() -> Result<()> {
    let path = "../datasets/2019-Nov.csv";

    if Path::new(path).exists() {
        let df = process_ecommerce_data(path)?;
        println!("{}", df);
    } else {
        println!("File path '{}' not found. Execution skipped.", path);
    }

    Ok(())
}
use anyhow::Result;
use lm_linfa_market_pipeline::MarketMLPipeline;
use std::path::Path;

fn main() -> Result<()> {
    let parquet_path = Path::new("data/dataset_features.parquet");
    let model_path = Path::new("models/decision_tree.bin");

    if !parquet_path.exists() {
        anyhow::bail!("The Parquet input file '{}' does not exist.", parquet_path.display());
    }

    println!("📖 Loading data and training Linfa Decision Tree...");
    let accuracy = MarketMLPipeline::train_and_save_model(parquet_path, model_path)?;
    println!("✅ Model saved in '{}'", model_path.display());
    println!("🎯 Accuracy: {:.2}%", accuracy * 100.0);

    println!("\n🔮 Live inference...");
    let live_sample = vec![95050.0, 95010.0, 95100.0, 94920.0];
    let signal = MarketMLPipeline::predict_live_sample(model_path, live_sample)?;

    match signal {
        1 => println!("📈 Signal: BUY"),
        _ => println!("📉 Signal: SELL / NEUTRAL"),
    }

    Ok(())
}
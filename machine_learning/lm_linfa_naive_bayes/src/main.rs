use anyhow::Result;
use lm_linfa_naive_bayes::{BayesType, WineBayesPipeline};
use std::path::Path;

fn main() -> Result<()> {
    let csv_path = Path::new("data/WineQT.csv");
    if !csv_path.exists() {
        anyhow::bail!(
            "The input CSV file '{}' does not exist in the root directory.",
            csv_path.display()
        );
    }

    println!("📖 Loading WineQT data...");
    let (train, valid) = WineBayesPipeline::load_dataset(csv_path)?;

    let model_configs = vec![
        (BayesType::Gaussian, Path::new("models/gaussian_nb.bin")),
        (BayesType::Bernoulli, Path::new("models/bernoulli_nb.bin")),
        (BayesType::Multinomial, Path::new("models/multinomial_nb.bin")),
    ];

    let mut trained_models = Vec::new();

    println!("\n🏋️ Training and evaluating Naive Bayes algorithms:\n");
    for (bayes_type, path) in &model_configs {
        let (model, metrics) = WineBayesPipeline::train_and_eval(&train, &valid, *bayes_type)?;
        WineBayesPipeline::save_metrics(&metrics, path)?;

        println!("==> Model: {:?}", metrics.model_type);
        println!("    Accuracy : {:.2}%", metrics.accuracy * 100.0);
        println!("    MCC      : {:.4}", metrics.mcc);
        println!("    Saved    : {}", path.display());
        println!("----------------------------------");

        trained_models.push((bayes_type, model));
    }

    let sample = vec![7.4, 0.3, 0.4, 2.0, 0.07, 15.0, 40.0, 0.995, 3.3, 0.7, 11.5];

    println!("\n🔮 Comparing inference results for the same wine sample:");
    for (bayes_type, model) in &trained_models {
        let pred = WineBayesPipeline::predict_model(model, &sample)?;
        let result_str = if pred == 1 {
            "🍷 Good (Quality)"
        } else {
            "🍷 Bad (Average/Poor)"
        };
        println!(" -> {:<12}: {}", format!("{:?}", bayes_type), result_str);
    }

    Ok(())
}
use anyhow::Result;
use lm_linfa_pls::PlsPipeline;

fn main() -> Result<()> {
    println!("=== 🧪 CLI Test: Linfa PLS Regression ===");

    let (_model, eval) = PlsPipeline::train_pls(1000, 10, 3, 42, 3)?;

    println!("Samples: {}, Symptoms: {}, Objectives: {}", eval.n_samples, eval.n_features, eval.n_targets);
    println!("\nActual B (first 2 lines): {:?}", &eval.true_b[..2]);
    println!("Estimated B (first 2 rows): {:?}", &eval.estimated_b[..2]);

    PlsPipeline::save_to_models(&eval, "pls_evaluation_results.json")?;
    println!("\n✅ Results successfully saved to 'models/' folder");

    Ok(())
}
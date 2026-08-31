use anyhow::Result;
use lm_linfa_tsne::TsnePipeline;

fn main() -> Result<()> {
    println!("=== 🧪 Linfa t-SNE Test Pipeline ===");

    let perplexity = 10.0;
    let approx_threshold = 0.1;

    println!("I'm calculating t-SNE for the Iris dataset...");
    let result = TsnePipeline::run_iris_tsne(perplexity, approx_threshold)?;

    println!("{} points scored.", result.points.len());
    
    // Sample first 5 points
    for pt in result.points.iter().take(5) {
        println!("  X: {:8.3} | Y: {:8.3} | Class: {}", pt.x, pt.y, pt.label);
    }

    let output_path = "iris_tsne_result.json";
    TsnePipeline::save_result(&result, output_path)?;
    println!("✅ Results successfully saved to '{}'", output_path);

    Ok(())
}
use gfrig_collaborative_scientific_mas::CollaborativeScientificMas;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Model running in a local Ollama instance (e.g., qwen3.5:9b or gpt-oss:20b)
    let mas = CollaborativeScientificMas::new("qwen3.5:9b");

    // --- TASK 1: RAG Syntax + Symbolic Differential Equation ---
    let task_1 = "Find the MathCore syntax for solving dy/dx = y * x^2. Then execute it.";
    println!("============================================================");
    println!("🚀 SCENARIO 1: {task_1}");
    println!("============================================================\n");

    let state_1 = mas.run(task_1).await?;
    println!("\n🧑‍🔬 FINAL RESULT (Scenario 1):\n{}\n", state_1.final_answer.as_deref().unwrap_or(""));

    // --- TASK 2: Factual Research (Wikipedia) + Visualization (Bar Chart) ---
    let task_2 = "Find the 5 largest countries by area. Then create a bar chart saved as 'top_5_countries.png'.";
    println!("============================================================");
    println!("🚀 SCENARIO 2: {task_2}");
    println!("============================================================\n");

    let state_2 = mas.run(task_2).await?;
    println!("\n🧑‍🔬 FINAL RESULT (Scenario 2):\n{}", state_2.final_answer.as_deref().unwrap_or(""));
    println!("\n📊 EXECUTION TRACE (Graph-Flow Audit):");
    println!("{}", serde_json::to_string_pretty(&state_2.trace)?);

    Ok(())
}
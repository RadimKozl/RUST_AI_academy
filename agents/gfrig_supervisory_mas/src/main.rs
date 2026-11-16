use gfrig_supervisory_mas::visualizer::save_mermaid_visualization;
use gfrig_supervisory_mas::SupervisoryScientificMas;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configuration of models in local Ollama (corresponds to the models in your Python files)
    let mas = SupervisoryScientificMas::new(
        "qwen3.5:9b", // Supervisor
        "qwen3.5:9b", // Researcher
        "qwen3.5:9b", // Coder
    );

    // --- SCENARIO 1: RAG Syntax + Symbolic Differential Equation ---
    let task_1 = "Find the MathCore syntax for solving dy/dx = y * x^2. Then execute it.";
    println!("============================================================");
    println!("🚀 SCENARIO 1: {task_1}");
    println!("============================================================");

    let state_1 = mas.run(task_1).await?;
    println!(
        "\n🧑‍🔬 FINAL ANSWER (Scenario 1):\n{}\n",
        state_1.final_answer.as_deref().unwrap_or("No answer")
    );

    // --- SCENARIO 2: Factual Research (Wikipedia) + Visualization (Bar Chart) ---
    let task_2 = "Find the 5 largest countries by area. Then create a bar chart saved as 'largest_countries_bar.png'.";
    println!("============================================================");
    println!("🚀 SCENARIO 2: {task_2}");
    println!("============================================================");

    let state_2 = mas.run(task_2).await?;
    println!(
        "\n🧑‍🔬 FINAL ANSWER (Scenario 2):\n{}\n",
        state_2.final_answer.as_deref().unwrap_or("No answer")
    );

    // Export graph topology and actual execution traces to Mermaid (equivalent to visualizer.py)
    save_mermaid_visualization("docs/supervisory_graph_trace.md", Some(&state_2))?;

    Ok(())
}
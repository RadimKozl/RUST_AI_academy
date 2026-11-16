// src/main.rs
use gfrig_swarm_mas::visualizer::save_mermaid_visualization;
use gfrig_swarm_mas::SwarmScientificMas;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🐝 Starting Rust Swarm Multi-Agent System...");

    let swarm = SwarmScientificMas::new(
        "qwen3.5:9b", // Researcher agent
        "qwen3.5:9b", // Coder agent
    );

    let task = "Find the MathCore syntax for solving dy/dx = y * x^2. Then execute it.";
    println!("📋 TASK: {task}\n");

    let final_state = swarm.run(task).await?;

    println!("\n============================================================");
    println!("🎯 SWARM FINAL ANSWER:");
    println!("============================================================");
    println!("{}", final_state.final_answer.as_deref().unwrap_or("No answer"));
    println!("============================================================");

    // Saving the Swarm chart visualization to Mermaid
    save_mermaid_visualization("docs/swarm_graph_topology.md", Some(&final_state))?;

    Ok(())
}
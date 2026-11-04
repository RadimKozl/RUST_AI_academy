use anyhow::Result;
use dotenvy::dotenv;
use gfrig_base_node::run_langgraph_workflow;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG + GRAPH-FLOW v0.8 (DYNAMIC ARCHITECTURE AUDIT) ===\n");

    let model_name = "qwen3.5:4b";
    let user_query = "Explain why stateful graphs are useful for AI agents.";

    println!("🚀 Executing Graph Workflow...\n");

    match run_langgraph_workflow(model_name, user_query).await {
        Ok(final_state) => {
            println!("--- Automatically Generated Agent Architecture (Mermaid) ---");
            println!("{}\n", final_state.architecture_mermaid);

            println!("--- Graph State Data ---");
            println!("Query: {}", final_state.query);
            println!("State Data Array: {:?}\n", final_state.data);

            println!("--- Final LLM Node Output ---");
            println!("{}\n", final_state.final_answer);

            // XAI / Neo4j Audit Trail (contains the complete state, including the graph topology)
            let json_audit = serde_json::to_string_pretty(&final_state).unwrap();
            println!("--- XAI Audit Trail (JSON with Dynamic Topology) ---");
            println!("{}", json_audit);
        }
        Err(e) => eprintln!("Error during graph execution: {}", e),
    }

    Ok(())
}
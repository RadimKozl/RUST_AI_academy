use anyhow::Result;
use dotenvy::dotenv;
use gfrig_sequential_node::{run_dynamic_workflow, NodeMode};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST DYNAMIC ROUTER PATTERN ==\n");

    let user_query = "We need to process incoming telemetry. Decide whether to run the math calculator or web scraping.";

    let workflow_steps = vec![
        (
            "router_node".to_string(),
            NodeMode::LlmRouter {
                model_name: "glm4:9b".to_string(),
                preamble: "You are a router agent. Analyze the query and output ONLY one of these exact strings: 'run_calculator' or 'run_scraping'.".to_string(),
            },
        ),
        (
            "dispatcher_node".to_string(),
            NodeMode::CustomTool {
                action_name: "Conditional Dispatcher".to_string(),
                logic: Arc::new(|ctx| {
                    Box::pin(async move {
                        // We read the decision that the LLM router saved a moment ago
                        let decision = ctx.get::<String>("router_decision").unwrap_or_default();
                        
                        match decision.as_str() {
                            d if d.contains("run_calculator") => {
                                Ok("Dispatcher executed: Calculator tool (2 + 2 = 4)".to_string())
                            }
                            d if d.contains("run_scraping") => {
                                Ok("Dispatcher executed: Web Scraping tool (Payload items: 42)".to_string())
                            }
                            _ => {
                                Ok(format!("Dispatcher executed default fallback for decision: '{}'", decision))
                            }
                        }
                    })
                }),
            },
        ),
        (
            "synthesize_node".to_string(),
            NodeMode::LlmAgent {
                model_name: "mistral:7b".to_string(),
                preamble: "You are an expert AI orchestrator summarizing the executed workflow path.".to_string(),
            },
        ),
    ];

    println!("🚀 Executing Workflow with LLM Router...\n");

    match run_dynamic_workflow(user_query, workflow_steps).await {
        Ok(final_state) => {
            println!("--- Architecture (Mermaid) ---");
            println!("{}\n", final_state.architecture_mermaid);

            println!("--- Accumulated Data Array ---");
            for (i, item) in final_state.data.iter().enumerate() {
                println!("  [{}] {}", i + 1, item);
            }
            println!();

            println!("--- Final LLM Synthesis Output ---");
            println!("{}\n", final_state.final_answer);

            let json_audit = serde_json::to_string_pretty(&final_state).unwrap();
            println!("--- XAI Audit Trail (JSON) ---");
            println!("{}", json_audit);
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(())
}
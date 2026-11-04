use anyhow::Result;
use dotenvy::dotenv;
use gfrig_parallel_node::{run_graph_workflow, NodeMode, UnifiedNode};
use std::sync::Arc;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST REAL-WORLD AGENTIC WORKFLOW: PARALLEL TELEMETRY & AI SYNTHESIS ===\n");

    let user_query = "Analyze current infrastructure health and provide a summary.";

    // 1. Node A: Entry point that defines the context and the query
    let node_start = UnifiedNode {
        id_str: "node_start".to_string(),
        label: "Telemetry Initializer".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Init Telemetry".to_string(),
            logic: Arc::new(|_ctx| {
                Box::pin(async move {
                    Ok("System telemetry monitoring initialized successfully.".to_string())
                })
            }),
        },
    };

    // 2. Parallel node (Fork-Join via tokio::join!):
    // Simultaneously downloads data from the server (API) and calculates local CPU/RAM usage statistics.
    let node_parallel_fetch = UnifiedNode {
        id_str: "node_parallel_fetch".to_string(),
        label: "Parallel Data Fetch (API + Metrics)".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Parallel Fetch".to_string(),
            logic: Arc::new(|_ctx| {
                Box::pin(async move {
                    // Branch 1: Simulating a call to an external cloud API
                    let api_call = async {
                        sleep(Duration::from_millis(80)).await;
                        "Cloud API status: 200 OK (Latency: 42ms, Active nodes: 12)".to_string()
                    };

                    // Branch 2: Simulation of local computation of system metrics
                    let local_metrics = async {
                        sleep(Duration::from_millis(50)).await;
                        "Local CPU Usage: 34.5%, RAM Load: 68.2% (Healthy)".to_string()
                    };

                    // We launch both things at once and wait for both
                    let (api_result, metrics_result) = tokio::join!(api_call, local_metrics);

                    Ok(format!("Collected -> [{}] AND [{}]", api_result, metrics_result))
                })
            }),
        },
    };

    // 3. Node D: Final LLM Agent (Synthesis via Rig and Ollama)
    // Takes the data gathered from previous steps and creates a human-readable report.
    let node_synthesis = UnifiedNode {
        id_str: "node_synthesis".to_string(),
        label: "AI Synthesis (LLM Agent)".to_string(),
        mode: NodeMode::LlmAgent {
            model_name: "glm4:9b".to_string(), // You can change this based on the models downloaded in Ollama
            preamble: "You are an elite DevOps AI assistant. Create a professional, concise executive summary based on the provided infrastructure telemetry data.".to_string(),
        },
    };

    let nodes = vec![node_start, node_parallel_fetch, node_synthesis];
    
    // Flow topology
    let edges = vec![
        ("node_start", "node_parallel_fetch"),
        ("node_parallel_fetch", "node_synthesis"),
    ];

    println!("🚀 Running Real-World Pipeline...\n");

    match run_graph_workflow(user_query, nodes, edges, "node_start").await {
        Ok(final_state) => {
            println!("--- 📊 Architecture (Mermaid) ---");
            println!("{}\n", final_state.architecture_mermaid);

            println!("--- 📝 Accumulated Execution Steps ---");
            for (i, item) in final_state.data.iter().enumerate() {
                println!("  [{}] {}", i + 1, item);
            }
            println!();

            println!("--- 🤖 Final LLM Executive Summary ---");
            println!("{}\n", final_state.final_answer);

            // Adding XAI output in JSON format
            let json_audit = serde_json::to_string_pretty(&final_state).unwrap();
            println!("--- 🔍 XAI Audit Trail (JSON) ---");
            println!("{}", json_audit);
        }
        Err(e) => eprintln!("Error during execution: {}", e),
    }

    Ok(())
}
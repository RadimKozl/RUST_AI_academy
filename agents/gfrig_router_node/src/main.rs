use anyhow::Result;
use chrono::Local;
use dotenvy::dotenv;
use gfrig_router_node::{run_graph_workflow, NodeMode, UnifiedNode};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST AGENTIC WORKFLOW: LLM SEMANTIC ROUTER (Rig + Ollama) ===\n");

    let user_question = "What is artificial intelligence?";

    let node_a = UnifiedNode {
        id_str: "node_a".to_string(),
        label: "Node A (Start)".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node A".to_string(),
            logic: Arc::new(|_ctx| {
                Box::pin(async move {
                    let now = Local::now().format("%S.%f").to_string();
                    Ok(format!("A - Timestamp: {}", now))
                })
            }),
        },
    };

    let node_llm_router = UnifiedNode {
        id_str: "node_router".to_string(),
        label: "LLM Semantic Router".to_string(),
        mode: NodeMode::LlmRouter {
            model_name: "glm4:9b".to_string(),
            preamble: "You are a precise routing agent. Analyze the given question. If the question asks for a definition or explanation starting with 'What is', reply strictly with 'b'. If it asks about a person, identity, or anything else, reply strictly with 'c'. Output ONLY the single letter 'b' or 'c'.".to_string(),
        },
    };

    let node_b = UnifiedNode {
        id_str: "node_b".to_string(),
        label: "Node B (Definition Branch)".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node B (Definition)".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    let decision = ctx.get::<String>("router_decision").unwrap_or_default();
                    if decision.contains("b") {
                        let now = Local::now().format("%S.%f").to_string();
                        Ok(format!("B (Executed by LLM choice) - Timestamp: {}", now))
                    } else {
                        Ok("B (Skipped - path not chosen)".to_string())
                    }
                })
            }),
        },
    };

    let node_c = UnifiedNode {
        id_str: "node_c".to_string(),
        label: "Node C (Alternative Branch)".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node C (Alternative)".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    let decision = ctx.get::<String>("router_decision").unwrap_or_default();
                    if decision.contains("c") {
                        let now = Local::now().format("%S.%f").to_string();
                        Ok(format!("C (Executed by LLM choice) - Timestamp: {}", now))
                    } else {
                        Ok("C (Skipped - path not chosen)".to_string())
                    }
                })
            }),
        },
    };

    let node_d = UnifiedNode {
        id_str: "node_d".to_string(),
        label: "Node D (AI Synthesis)".to_string(),
        mode: NodeMode::LlmAgent {
            model_name: "glm4:9b".to_string(),
            preamble: "You are an AI supervisor. Review the execution path and provide a concise summary of the workflow outcome.".to_string(),
        },
    };

    let nodes = vec![node_a, node_llm_router, node_b, node_c, node_d];

    let edges = vec![
        ("node_a", "node_router"),
        ("node_router", "node_b"),
        ("node_router", "node_c"),
        ("node_b", "node_d"),
        ("node_c", "node_d"),
    ];

    println!("🚀 Running LLM-Routed Workflow with Question: '{}'...\n", user_question);

    match run_graph_workflow(user_question, nodes, edges, "node_a").await {
        Ok(final_state) => {
            println!("--- 📊 Architecture (Mermaid) ---");
            println!("{}\n", final_state.architecture_mermaid);

            println!("--- 📝 Accumulated Execution Steps ---");
            for (i, item) in final_state.data.iter().enumerate() {
                println!("  [{}] {}", i + 1, item);
            }
            println!();

            println!("--- 🤖 Final LLM Synthesis ---");
            println!("{}\n", final_state.final_answer);

            let json_audit = serde_json::to_string_pretty(&final_state).unwrap();
            println!("--- 🔍 XAI Audit Trail (JSON) ---");
            println!("{}", json_audit);
        }
        Err(e) => eprintln!("Error during execution: {}", e),
    }

    Ok(())
}
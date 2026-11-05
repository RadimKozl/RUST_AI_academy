use anyhow::Result;
use dotenvy::dotenv;
use gfrig_loop_node::{run_loop_workflow, NodeMode, UnifiedNode};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST AGENTIC WORKFLOW: SELF-CORRECTION & CRITIQUE LOOP (Rig + Ollama) ===\n");

    let user_question = "Write a highly optimized Rust function for calculating Fibonacci numbers.";
    let user_query = "Execute generative self-correction loop.";

    // 1. Node A: Initialization of the assignment
    let node_a = UnifiedNode {
        id_str: "node_a".to_string(),
        label: "Task Initializer".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Init Task".to_string(),
            logic: Arc::new(|_ctx| {
                Box::pin(async move {
                    Ok("Starting iterative code generation and critique process.".to_string())
                })
            }),
        },
    };

    // 2. Node B: LLM Generator (Proposes code/solutions)
    let node_b = UnifiedNode {
        id_str: "node_b".to_string(),
        label: "AI Code Generator".to_string(),
        mode: NodeMode::LlmAgent {
            model_name: "glm4:9b".to_string(),
            preamble: "You are an expert Rust software engineer. Write or refine a clean, idiomatic, and optimized Rust solution based on the user's prompt and prior feedback.".to_string(),
        },
    };

    // 3. Node C: Critic and Counter Incrementer
    let node_c = UnifiedNode {
        id_str: "node_c".to_string(),
        label: "AI Critic & Incrementer".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Critique & Increment".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    let mut counter = ctx.get::<i32>("counter").unwrap_or_default();
                    counter += 1;
                    ctx.set("counter", counter)?;
                    
                    let data = ctx.get::<Vec<String>>("data").unwrap_or_default();
                    let last_draft = data.last().cloned().unwrap_or_default();

                    let critique = format!(
                        "Critique (Iteration {}): Code reviewed. Suggesting performance, error handling, and memory optimizations.",
                        counter
                    );
                    
                    Ok(format!("{} | Feedback on draft: [{}]", critique, last_draft))
                })
            }),
        },
    };

    // 4. Conditional Router: Decides whether to continue with iterations
    let node_router = UnifiedNode {
        id_str: "node_router".to_string(),
        label: "Loop Condition Router".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "shouldContinue Router".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    let counter = ctx.get::<i32>("counter").unwrap_or_default();
                    // We will perform a maximum of 2 rounds of refinement (max_loops = 2)
                    if counter < 2 {
                        ctx.set("router_decision", "b".to_string())?;
                        Ok(format!("Critique demands refinement. Looping back (counter = {})", counter))
                    } else {
                        ctx.set("router_decision", "d".to_string())?;
                        Ok(format!("Critique passed threshold (counter = {}). Exiting to final review.", counter))
                    }
                })
            }),
        },
    };

    // 5. Node D: Final AI Supervisor (Synthesis of Results)
    let node_d = UnifiedNode {
        id_str: "node_d".to_string(),
        label: "Final Supervisor Synthesis".to_string(),
        mode: NodeMode::LlmAgent {
            model_name: "glm4:9b".to_string(),
            preamble: "You are an elite AI technical lead. Review the final generated artifact and critique history, and provide a polished final executive summary.".to_string(),
        },
    };

    let nodes = vec![node_a, node_b, node_c, node_router, node_d];
    let edges = vec![
        ("node_a", "node_b"),
        ("node_b", "node_c"),
        ("node_c", "node_router"),
        ("node_router", "node_d"),
    ];

    println!("🚀 Running Self-Correction Critique Workflow...\n");

    match run_loop_workflow(user_query, user_question, nodes, edges, "node_a", 2).await {
        Ok(final_state) => {
            println!("--- 📊 Architecture (Mermaid) ---");
            println!("{}\n", final_state.architecture_mermaid);

            println!("--- 📝 Accumulated Execution Steps & Drafts ---");
            for (i, item) in final_state.data.iter().enumerate() {
                println!("  [{}] {}", i + 1, item);
            }
            println!();

            println!("--- 🤖 Final AI Supervisor Synthesis ---");
            println!("{}\n", final_state.final_answer);

            println!("--- 🔢 Final Counter Value: {} ---", final_state.counter);

            let json_audit = serde_json::to_string_pretty(&final_state).unwrap();
            println!("--- 🔍 XAI Audit Trail (JSON) ---");
            println!("{}", json_audit);
        }
        Err(e) => eprintln!("Error during execution: {}", e),
    }

    Ok(())
}
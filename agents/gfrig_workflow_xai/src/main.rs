use anyhow::Result;
use rig::providers::ollama;
use gfrig_workflow_xai::{AgentState, SqliteStorageManager, WorkflowGraph, XaiStep};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== RUST WORKFLOW GRAPH & XAI AUDIT DEMO ===\n");

    let db_path = "storage/workflow_xai.db";
    std::fs::create_dir_all("storage")?;
    let storage = SqliteStorageManager::new(db_path)?;

    // Initialization of the Ollama client and model via RIG
    let client = ollama::Ollama::new();
    let model = client.completion("glm4:9b");

    let agent = rig::agent::AgentBuilder::new(model)
        .preamble("You are an advanced AI research assistant with explainable reasoning capabilities.")
        .build();

    // 1. Graph Construction (WorkflowGraph Builder Pattern)
    let mut graph = WorkflowGraph::new();

    // NODE 1: Query Optimization (Prepare Query Node)
    let agent_clone1 = agent.clone();
    graph.add_node("prepare_query", move |state| {
        let agent = agent_clone1.clone();
        Box::pin(async move {
            let prompt = format!(
                "Improve this user query for better search or retrieval: '{}'. Return ONLY the improved query.",
                state.user_input
            );
            let response = agent.prompt(&prompt).await?;
            state.improved_query = response.output().trim().to_string();

            // XAI Record
            state.xai_audit_trail.push(XaiStep {
                node_name: "prepare_query".to_string(),
                input_data: state.user_input.clone(),
                generated_output: state.improved_query.clone(),
                reasoning: "Query was expanded with semantic synonyms to enhance retrieval quality.".to_string(),
            });
            Ok(())
        })
    });

    // NODE 2: Generating the assistant's response (Chat Node)
    let agent_clone2 = agent.clone();
    graph.add_node("chatbot_node", move |state| {
        let agent = agent_clone2.clone();
        Box::pin(async move {
            let prompt = format!(
                "Answer the user query based on the optimized query context.\nOriginal: {}\nOptimized: {}",
                state.user_input, state.improved_query
            );
            let response = agent.prompt(&prompt).await?;
            state.final_answer = response.output().trim().to_string();

            // XAI Record
            state.xai_audit_trail.push(XaiStep {
                node_name: "chatbot_node".to_string(),
                input_data: state.improved_query.clone(),
                generated_output: state.final_answer.clone(),
                reasoning: "Generated concise response matching user expectations using local LLM model.".to_string(),
            });
            Ok(())
        })
    });

    // 2. State initialization and graph execution
    let thread_id = "session_xai_001";
    let mut state = AgentState::new(
        thread_id,
        "What are the main trends in Edge AI and Explainable AI for 2026?",
    );

    // Launching the graph
    graph.invoke(&mut state).await?;

    // 3. Saving XAI traces to an SQLite database
    for step in &state.xai_audit_trail {
        storage.save_xai_step(thread_id, step)?;
    }

    // 4. Final Output and XAI Transparency Report
    println!("==========================================");
    println!("--------- FINAL GRAPH OUTPUT -------------");
    println!("==========================================");
    println!("{}\n", state.final_answer);

    println!("==========================================");
    println!("🔍 XAI AUDIT TRAIL & TRANSPARENCY REPORT:");
    println!("==========================================");
    for (idx, step) in state.xai_audit_trail.iter().enumerate() {
        println!("[{}] Node: {}", idx + 1, step.node_name);
        println!("    • Input:     {}", step.input_data);
        println!("    • Output:    {}", step.generated_output);
        println!("    • Reasoning: {}", step.reasoning);
        println!("------------------------------------------");
    }

    println!("\n✅ All audit steps successfully persisted to SQLite: {}", db_path);

    Ok(())
}
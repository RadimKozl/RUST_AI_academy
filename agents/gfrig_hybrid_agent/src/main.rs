use anyhow::Result;
use dotenvy::dotenv;
use std::sync::Arc;

use graph_flow::Task;
use gfrig_hybrid_agent::{DynamicWorkflowBuilder, GraphStepRecord, NodeMode, UnifiedNode};
use graph_flow::{InMemorySessionStorage, Session, SessionStorage};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST HYBRID LANGGRAPH AGENT: WIKIPEDIA RAG & SYNTHESIS ===\n");

    let topic = "Newton's laws of motion";
    let user_query = "Execute hybrid RAG and theory synthesis pipeline.";

    // 1. LLM Search Agent Node
    let llm_search_node = UnifiedNode {
        id_str: "llm_search".to_string(),
        label: "LLM Search Agent".to_string(),
        mode: NodeMode::LlmAgent {
            model_name: "glm4:9b".to_string(),
            preamble: "You are a helpful research agent. Provide a clear summary and context regarding the requested physical law.".to_string(),
        },
    };

    // 2. Theory Generation Node
    let theory_node = UnifiedNode {
        id_str: "theory_node".to_string(),
        label: "Theory Expert Node".to_string(),
        mode: NodeMode::LlmAgent {
            model_name: "glm4:9b".to_string(),
            preamble: "You are an expert in physics. From the given context, extract and explain the main physical theory and laws.".to_string(),
        },
    };

    let nodes = vec![llm_search_node, theory_node];
    let edges = vec![
        ("llm_search", "theory_node"),
    ];

    let mut workflow_builder = DynamicWorkflowBuilder::new("hybrid_rag_workflow");
    let mut task_map = std::collections::HashMap::new();

    for node in nodes {
        let task_arc = Arc::new(node);
        task_map.insert(task_arc.id().to_string(), task_arc.clone());
        workflow_builder = workflow_builder.add_task(task_arc);
    }

    for (from, to) in edges {
        workflow_builder = workflow_builder.add_edge(from, to);
    }

    let mermaid_code = workflow_builder.generate_mermaid();
    let _graph = workflow_builder.build()?;

    let storage = Arc::new(InMemorySessionStorage::new());
    let session_id = "session_hybrid_1".to_string();
    let session = Session::new_from_task(session_id.clone(), "llm_search");

    session.context.set("query", user_query.to_string())?;
    session.context.set("topic", topic.to_string())?;
    session.context.set("context", "".to_string())?;
    session.context.set("theory", "".to_string())?;
    session.context.set("formula", "".to_string())?;
    session.context.set("architecture_mermaid", mermaid_code.clone())?;
    session.context.set("audit_trail", Vec::<GraphStepRecord>::new())?;

    SessionStorage::save(&*storage, session).await?;

    println!("🚀 Executing Hybrid RAG Workflow for topic: '{}'...\n", topic);

    // 1. Execution of the LLM Search node (using the .response field instead of .output)
    let sess_1 = SessionStorage::get(&*storage, &session_id).await?.unwrap();
    if let Some(search_node) = task_map.get("llm_search") {
        println!("⏳ Running llm_search node...");
        let task_result = search_node.run(sess_1.context.clone()).await?;
        if let Some(output) = task_result.response {
            sess_1.context.set("context", output)?;
        }
    }
    SessionStorage::save(&*storage, sess_1).await?;

    // 2. Running the Theory node
    let sess_2 = SessionStorage::get(&*storage, &session_id).await?.unwrap();
    if let Some(theory_n) = task_map.get("theory_node") {
        println!("⏳ Running theory_node...");
        let task_result = theory_n.run(sess_2.context.clone()).await?;
        if let Some(output) = task_result.response {
            sess_2.context.set("theory", output)?;
        }
    }
    SessionStorage::save(&*storage, sess_2).await?;

    // Final data loading for display
    let final_sess = SessionStorage::get(&*storage, &session_id).await.unwrap().unwrap();
    let ctx = &final_sess.context;

    println!("\n--- 📊 Architecture (Mermaid) ---");
    println!("{}\n", ctx.get::<String>("architecture_mermaid").unwrap_or_default());

    println!("--- 🧠 Retrieved Context ---");
    println!("{}\n", ctx.get::<String>("context").unwrap_or_default());

    println!("--- 🔬 Generated Theory ---");
    println!("{}\n", ctx.get::<String>("theory").unwrap_or_default());

    // Construction and output of the XAI Audit Trail in JSON format
    let final_state = gfrig_hybrid_agent::PhysicsWorkflowState {
        topic: ctx.get::<String>("topic").unwrap_or_default(),
        context: ctx.get::<String>("context").unwrap_or_default(),
        theory: ctx.get::<String>("theory").unwrap_or_default(),
        formula: ctx.get::<String>("formula").unwrap_or_default(),
        architecture_mermaid: ctx.get::<String>("architecture_mermaid").unwrap_or_default(),
        audit_trail: ctx.get::<Vec<GraphStepRecord>>("audit_trail").unwrap_or_default(),
    };

    let json_audit = serde_json::to_string_pretty(&final_state)?;
    println!("--- 🔍 XAI Audit Trail (JSON) ---");
    println!("{}", json_audit);

    Ok(())
}
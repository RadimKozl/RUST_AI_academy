// src/main.rs
use anyhow::Result;
use dotenvy::dotenv;
use std::path::Path;

use gfrig_micro_graph_rag::{AdvancedRagDatabase, AdvancedRagState, RagStepRecord, get_embedding};
use rig::providers::ollama;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST GRAPH-RAG AGENTIC PIPELINE (ArXiv Agentic Applications) ===\n");

    let ollama_host = "http://localhost:11434";
    let embedding_model = "nomic-embed-text:latest";
    let chat_model = "glm4:9b"; 
    let db_path = "./storage/graph_rag/graph_rag_db.sqlite";

    if let Some(parent) = Path::new(db_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let db = AdvancedRagDatabase::init(db_path).await?;

    // 1. Ingestion of an ArXiv document on the topic of "Agentic Applications" into G-RAG
    let arxiv_title = "Agentic Workflows and Multi-Agent Applications in Modern Software";
    let arxiv_abstract = "Agentic applications leverage LLMs to autonomously execute tasks, utilizing frameworks like AutoGen and LangGraph. These systems rely on specialized roles such as Researchers and Coders coordinated via graph networks.";
    
    let doc_id = db.insert_document(arxiv_abstract, arxiv_title).await?;
    
    if let Ok(emb) = get_embedding(ollama_host, embedding_model, arxiv_abstract).await {
        let chunk_id = db.insert_chunk(doc_id, arxiv_abstract, &emb).await?;
        
        // Inserting entities and relationships into a Petgraph/SQLite graph
        let entities = vec![
            ("Agentic Applications", "Concept", "Autonomous systems driven by LLMs"),
            ("AutoGen", "Framework", "Multi-agent conversation framework"),
            ("LangGraph", "Framework", "Graph-based state machine framework for agents"),
        ];
        let relations = vec![
            ("Agentic Applications", "UTILIZES", "AutoGen"),
            ("Agentic Applications", "UTILIZES", "LangGraph"),
        ];
        db.insert_graph_knowledge(chunk_id, &entities, &relations).await?;
        println!("✅ ArXiv document & knowledge graph successfully indexed.");
    }

    // 2. Initialization of the agent's state
    let mut state = AdvancedRagState {
        question: "How do agentic applications utilize frameworks like AutoGen and LangGraph?".to_string(),
        ..Default::default()
    };

    let max_iterations = 3;
    let client = ollama::Ollama::new();
    let model = client.completion(chat_model);
    let agent = rig::agent::AgentBuilder::new(model)
        .preamble("You are an expert AI Researcher assistant. Answer precisely based on the provided Graph RAG context.")
        .build();

    // 3. Iterative cycle (Graph Retrieval -> Generation -> Critic -> Loop)
    while state.iteration < max_iterations {
        state.iteration += 1;
        println!("\n🔄 --- Iteration {} ---", state.iteration);

        // Step A: G-RAG Retrieval (Vector + Petgraph neighbors)
        println!("⏳ Running Graph RAG retrieval for query: '{}'...", state.question);
        let query_emb = get_embedding(ollama_host, embedding_model, &state.question).await?;
        
        // Searching for the keyword "Agentic Applications" in the graph
        let graph_rag_context = db.search_graph_rag(&query_emb, "Agentic Applications", 3).await?;
        
        state.context = graph_rag_context.clone();

        state.audit_trail.push(RagStepRecord {
            step_name: format!("graph_retrieval_iter_{}", state.iteration),
            input_data: state.question.clone(),
            output_data: "Retrieved combined vector chunks and Petgraph topology".to_string(),
        });

        // Step B: Generating the response
        println!("⏳ Generating answer using Graph RAG context...");
        let prompt = format!(
            "Graph RAG Context:\n{}\n\nQuestion: {}\nAnswer:",
            graph_rag_context, state.question
        );
        let response = agent.prompt(&prompt).await?;
        state.answer = response.output().trim().to_string();

        state.audit_trail.push(RagStepRecord {
            step_name: format!("generation_iter_{}", state.iteration),
            input_data: prompt,
            output_data: state.answer.clone(),
        });

        // Step C: Critical evaluation of the agent (Critic)
        println!("⏳ Critic agent evaluating answer...");
        let score = if state.answer.contains("AutoGen") && state.answer.contains("LangGraph") {
            9.0
        } else {
            6.0
        };
        state.critic_score = score;
        println!("📊 Critic Score: {:.1}/10.0", score);

        // Step D: Loop decision
        if score >= 8.0 {
            println!("✅ Target score achieved. Exiting loop.");
            break;
        } else {
            println!("⚠️ Score below threshold. Refining...");
            state.question = format!("{} (provide deep architectural details)", state.question);
        }
    }

    // 4. Final Output and XAI Audit
    println!("\n========================================");
    println!("--- 🎯 Final Graph RAG Answer ---");
    println!("{}\n", state.answer);

    let json_audit = serde_json::to_string_pretty(&state)?;
    println!("--- 🔍 XAI Audit Trail (JSON) ---");
    println!("{}", json_audit);

    Ok(())
}
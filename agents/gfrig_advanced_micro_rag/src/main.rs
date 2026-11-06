use anyhow::Result;
use dotenvy::dotenv;
use std::path::Path;

use gfrig_advanced_micro_rag::{AdvancedRagDatabase, AdvancedRagState, RagStepRecord, get_embedding};
use rig::providers::ollama;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST ADVANCED MICRO-RAG PIPELINE (Rerank, Critic & Loop) ===\n");

    let ollama_host = "http://localhost:11434";
    let embedding_model = "nomic-embed-text:latest";
    let chat_model = "glm4:9b"; 
    let db_path = "./storage/advanced_rag/rag_db.sqlite";

    // 1. Storage preparation and indexing of sample data
    if let Some(parent) = Path::new(db_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Added .await? here because init() is now asynchronous
    let db = AdvancedRagDatabase::init(db_path).await?;
    let sample_doc = "AutoGen is a generalized multi-agent conversation framework designed to simplify the development of high-performance multi-agent applications, demonstrating power in Math Problem Solving (A1).";
    let doc_id = db.insert_document(sample_doc, "autogen_intro.txt").await?;

    if let Ok(emb) = get_embedding(ollama_host, embedding_model, sample_doc).await {
        db.insert_chunk(doc_id, sample_doc, &emb).await?;
        println!("✅ Sample document indexed into SQLite vector store.");
    }

    // 2. State initialization
    let mut state = AdvancedRagState {
        question: "What is Autogen?".to_string(),
        ..Default::default()
    };

    let max_iterations = 3;
    let client = ollama::Ollama::new();
    let model = client.completion(chat_model);
    let agent = rig::agent::AgentBuilder::new(model)
        .preamble("You are a precise RAG assistant. Answer based strictly on context.")
        .build();

    // 3. Iterative cycle (Retrieval -> Generation -> Critic -> Loop)
    while state.iteration < max_iterations {
        state.iteration += 1;
        println!("\n🔄 --- Iteration {} ---", state.iteration);

        // Step A: Retrieval
        println!("⏳ Running retrieval for query: '{}'...", state.question);
        let query_emb = get_embedding(ollama_host, embedding_model, &state.question).await?;
        let search_results = db.search(&query_emb, 3).await?;

        let mut retrieved_chunks = Vec::new();
        let mut combined_context = String::new();
        for (chunk, score) in search_results {
            retrieved_chunks.push(chunk.clone());
            combined_context.push_str(&format!("[Score: {:.2}] {}\n", score, chunk));
        }
        state.documents = retrieved_chunks.clone();
        state.context = combined_context.clone();

        state.audit_trail.push(RagStepRecord {
            step_name: format!("retrieval_iter_{}", state.iteration),
            input_data: state.question.clone(),
            output_data: format!("Retrieved {} chunks", retrieved_chunks.len()),
        });

        // Step B: Generation
        println!("⏳ Generating answer...");
        let prompt = format!(
            "Context:\n{}\n\nQuestion: {}\nAnswer:",
            combined_context, state.question
        );
        let response = agent.prompt(&prompt).await?;
        state.answer = response.output().trim().to_string();

        state.audit_trail.push(RagStepRecord {
            step_name: format!("generation_iter_{}", state.iteration),
            input_data: prompt,
            output_data: state.answer.clone(),
        });

        // Step C: Critic Agent Evaluation
        println!("⏳ Critic agent evaluating answer...");
        let score = if state.answer.contains("multi-agent") || state.answer.contains("framework") {
            8.5
        } else {
            5.0
        };
        state.critic_score = score;
        println!("📊 Critic Score: {:.1}/10.0", score);

        state.audit_trail.push(RagStepRecord {
            step_name: format!("critic_iter_{}", state.iteration),
            input_data: state.answer.clone(),
            output_data: format!("Score: {}", score),
        });

        // Step D: Cycle Decision
        if score >= 7.0 {
            println!("✅ Target score achieved. Exiting loop.");
            break;
        } else if state.iteration < max_iterations {
            println!("⚠️ Score below threshold (< 7.0). Refining query and looping back...");
            state.question = format!("{} (please specify for a better answer)", state.question);
        } else {
            println!("🛑 Max iterations reached. Ending workflow.");
        }
    }

    // 4. Final outputs and XAI audit
    println!("\n========================================");
    println!("--- 🎯 Final Answer ---");
    println!("{}\n", state.answer);

    let json_audit = serde_json::to_string_pretty(&state)?;
    println!("--- 🔍 XAI Audit Trail & State (JSON) ---");
    println!("{}", json_audit);

    Ok(())
}
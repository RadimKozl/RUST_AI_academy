use anyhow::Result;
use dotenvy::dotenv;
use rig_micro_rag::{get_embedding, run_micro_rag, MicroRagDatabase};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG MICRO-RAG PIPELINE ===\n");

    let ollama_host = "http://localhost:11434";
    let embedding_model = "nomic-embed-text:latest";
    let chat_model = "qwen3.5:4b";
    let db_path = "./storage/micro_rag/rag_db.sqlite";

    // Creating a directory for storage
    if let Some(parent) = Path::new(db_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Initialization and population of the database with sample data (Index-Time RAG)
    let mut db = MicroRagDatabase::init(db_path)?;
    
    let sample_doc = "Autogen is a generalized multi-agent conversation framework designed to simplify the development of high-performance multi-agent applications, demonstrating power in Math Problem Solving (A1).";
    let doc_id = db.insert_document(sample_doc, "autogen_intro.txt", 0)?;

    // Calculating the embedding for a sample chunk and saving it
    if let Ok(emb) = get_embedding(ollama_host, embedding_model, sample_doc).await {
        db.insert_chunk(doc_id, sample_doc, &emb)?;
        println!("✅ Sample document indexed into SQLite & vector store.");
    }

    // Run RAG query
    let query = "What is Autogen?";
    println!("\nUser Query: '{}'\n", query);

    match run_micro_rag(db_path, ollama_host, embedding_model, chat_model, query).await {
        Ok(res) => {
            println!("--- Retrieved Context Chunks ---");
            for (i, ctx) in res.retrieved_context.iter().enumerate() {
                println!("{}. {}\n", i + 1, ctx);
            }

            println!("--- Final RAG Output ---");
            println!("{}\n", res.final_output);

            // XAI / Neo4j Audit Trail
            let json_audit = serde_json::to_string_pretty(&res.history).unwrap();
            println!("--- XAI Audit Trail (JSON) ---");
            println!("{}", json_audit);
        }
        Err(e) => eprintln!("Error during Micro-RAG execution: {}", e),
    }

    Ok(())
}
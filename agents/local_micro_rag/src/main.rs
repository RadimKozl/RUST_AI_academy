use anyhow::Result;
use local_micro_rag::{chunk_text, OllamaClient, RagDatabase};
use std::io::{self, Write};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Launching Rust Local Micro-RAG Pipeline ===");

    let ollama_host = "http://localhost:11434";
    let client = OllamaClient::new(ollama_host);

    let storage_dir = Path::new("./storage");
    std::fs::create_dir_all(storage_dir)?;
    let db_path = storage_dir.join("rag_database.db");

    let mut db = RagDatabase::init(db_path)?;

    // Document folder
    let docs_dir = Path::new("./data/documents");
    std::fs::create_dir_all(docs_dir)?;

    println!("I am processing documents from the folder: {:?}", docs_dir);
    for entry in std::fs::read_dir(docs_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let path_str = path.to_string_lossy().to_string();
            println!("I find a file: {}", path_str);

            let content = if path_str.ends_with(".png") || path_str.ends_with(".jpg") {
                let img_bytes = std::fs::read(&path)?;
                client.analyze_image(&img_bytes).await.unwrap_or_default()
            } else {
                std::fs::read_to_string(&path).unwrap_or_default()
            };

            if content.is_empty() {
                continue;
            }

            // Document summarization
            let summary_prompt = format!("Provide a brief summary (max. 3 sentences) of the text:\n{}", content);
            let summary = client.generate_text(&summary_prompt).await.unwrap_or_default();

            let doc_id = db.insert_document(&path_str, &content, &summary)?;

            // Rozsekání na chunky a výpočet embeddingů
            let chunks = chunk_text(&content, 800, 200);
            for chunk in chunks {
                if let Ok(emb) = client.generate_embedding(&chunk).await {
                    db.insert_chunk(doc_id, &chunk, &emb)?;
                }
            }
            println!("✔ File {} processed", path_str);
        }
    }

    println!("\nAgent ready. Enter '/bye' to exit.");

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut query = String::new();
        stdin.read_line(&mut query)?;
        let query = query.trim();

        if query == "/bye" {
            println!("I am ending the conversation.");
            break;
        }

        if query.is_empty() {
            continue;
        }

        // Generating an embedding for the query and searching in the DB
        if let Ok(query_emb) = client.generate_embedding(query).await {
            let results = db.search(&query_emb, 3)?;

            if results.is_empty() {
                println!("No relevant documents were found.");
                continue;
            }

            let mut context = String::new();
            for res in &results {
                context.push_str(&format!(
                    "Document ({}), Score: {:.2}\nExcerpt: {}\n\n",
                    res.path, res.score, res.chunk_text
                ));
            }

            let prompt = format!(
                "You are a RAG assistant. Answer the query based on the context.\nContext:\n{}\nQuery: {}\nAnswer:",
                context, query
            );

            match client.generate_text(&prompt).await {
                Ok(answer) => {
                    println!("\n--- Response ---");
                    println!("{}\n", answer);
                }
                Err(e) => eprintln!("Error generating response: {}", e),
            }
        } else {
            eprintln!("Error creating query embedding.");
        }
    }

    Ok(())
}

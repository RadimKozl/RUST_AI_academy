# Micro-RAG Pipeline in Rust (Rig Framework & SQLite)

This project translates the LangChain Micro-RAG tutorials (`langchain_microRAG_embdedding.ipynb` and `langchain_microRAG_pipeline.ipynb`) into a lightweight, high-performance Rust architecture using the **Rig** framework, **SQLite** (`rusqlite`), and local models via **Ollama**.

It implements an **Index-Time RAG** pattern where documents and vector embeddings are stored in SQLite, and retrieval is performed using custom **cosine similarity** vector search.

---

## ⚠️ Important Note Before Running

> **Prerequisite Action:** If you encountered a `FOREIGN KEY constraint failed` error or modified your database schemas, you **must delete the old SQLite database file** before running `cargo run`.
>
> **Command to clean database storage (PowerShell):**
> ```powershell
> Remove-Item -Recurse -Force .\storage\
> ```
> Or manually remove `./storage/micro_rag/rag_db.sqlite`.

---

## 🚀 Key Features

* **Custom Vector Store in SQLite**: Stores documents, chunk texts, and JSON-encoded embedding vectors directly in SQLite (`rag_db.sqlite`).
* **Cosine Similarity Search**: Native vector search implementation without external heavy vector DB dependencies (e.g., Chroma or Faiss).
* **Ollama Embeddings & LLM**: Integration with local models (`nomic-embed-text:latest` for embeddings and `qwen3.5:4b` or `mistral:7b` for text generation).
* **XAI & Audit Trail Ready**: Logged step records (`MicroRagStepRecord`) track query embeddings, vector retrieval scores, and LLM output generation for explainability and graph integration.

---

## 📁 Project Structure

```text
rig_micro_rag/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                   # Core SQLite manager, cosine similarity, and RAG pipeline
│   └── main.rs                  # Executable runner executing indexing and RAG querying
└── tests/
    └── micro_rag_tests.rs       # Integration test suite for vector search and database ops
```

---

## ⚙️ Dependencies (`Cargo.toml`)

Ensure your `Cargo.toml` includes the following dependencies:

```toml
[package]
name = "rig_micro_rag"
version = "0.1.0"
edition = "2021"

[dependencies]
rig = "0.46.0"
tokio = { version = "1.38", features = ["full"] }
rusqlite = { version = "0.31", features = ["bundled"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
dotenvy = "0.15"
reqwest = { version = "0.12", features = ["json"] }
```

---

## 💻 Code Overview

### 1. Library Logic (`src/lib.rs`)

```rust
use anyhow::{anyhow, Context, Result};
use rig::providers::ollama;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroRagStepRecord {
    pub step_name: String,
    pub input_value: String,
    pub generated_output: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MicroRagResult {
    pub query: String,
    pub retrieved_context: Vec<String>,
    pub final_output: String,
    pub history: Vec<MicroRagStepRecord>,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    embedding: Option<Vec<f32>>,
}

pub struct MicroRagDatabase {
    conn: Connection,
}

impl MicroRagDatabase {
    pub fn init<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let conn = Connection::open(db_path)
            .with_context(|| "Failed to open SQLite database for Micro-RAG")?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source TEXT,
                faiss_index_id INTEGER UNIQUE
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS vector_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                chunk_text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(id)
            )",
            [],
        )?;

        Ok(Self { conn })
    }

    pub fn insert_document(&mut self, content: &str, source: &str, faiss_index_id: i64) -> Result<i64> {
        self.conn.execute(
            "INSERT OR REPLACE INTO documents (content, source, faiss_index_id) VALUES (?1, ?2, ?3)",
            params![content, source, faiss_index_id],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn insert_chunk(&mut self, doc_id: i64, chunk_text: &str, embedding: &[f32]) -> Result<()> {
        let emb_json = serde_json::to_string(embedding)?;
        self.conn.execute(
            "INSERT INTO vector_chunks (doc_id, chunk_text, embedding_json) VALUES (?1, ?2, ?3)",
            params![doc_id, chunk_text, emb_json],
        )?;
        Ok(())
    }

    pub fn search(&self, query_emb: &[f32], limit: usize) -> Result<Vec<(String, f32)>> {
        let mut stmt = self.conn.prepare("SELECT chunk_text, embedding_json FROM vector_chunks")?;
        let rows = stmt.query_map([], |row| {
            let chunk_text: String = row.get(0)?;
            let emb_json: String = row.get(1)?;
            Ok((chunk_text, emb_json))
        })?;

        let mut scored_chunks = Vec::new();
        for row in rows {
            let (chunk_text, emb_json) = row?;
            let emb: Vec<f32> = serde_json::from_str(&emb_json).unwrap_or_default();
            if !emb.is_empty() {
                let score = cosine_similarity(query_emb, &emb);
                scored_chunks.push((chunk_text, score));
            }
        }

        scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_chunks.truncate(limit);
        Ok(scored_chunks)
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

pub async fn get_embedding(host: &str, model: &str, text: &str) -> Result<Vec<f32>> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/embeddings", host.trim_end_matches('/'));
    let payload = EmbeddingRequest {
        model: model.to_string(),
        prompt: text.to_string(),
    };

    let res: EmbeddingResponse = client.post(&url).json(&payload).send().await?.json().await?;
    res.embedding.ok_or_else(|| anyhow!("Ollama returned an empty embedding"))
}

pub async fn run_micro_rag(
    db_path: &str,
    ollama_host: &str,
    embedding_model: &str,
    chat_model: &str,
    query: &str,
) -> Result<MicroRagResult> {
    let mut history = Vec::new();

    let query_emb = get_embedding(ollama_host, embedding_model, query).await?;

    history.push(MicroRagStepRecord {
        step_name: "query_embedding".to_string(),
        input_value: query.to_string(),
        generated_output: format!("Vector generated with dimension {}", query_emb.len()),
    });

    let db = MicroRagDatabase::init(db_path)?;
    let search_results = db.search(&query_emb, 3)?;

    let mut context_texts = Vec::new();
    let mut combined_context = String::new();
    for (chunk, score) in search_results {
        context_texts.push(chunk.clone());
        combined_context.push_str(&format!("[Score: {:.2}] {}\n", score, chunk));
    }

    history.push(MicroRagStepRecord {
        step_name: "vector_retrieval".to_string(),
        input_value: query.to_string(),
        generated_output: format!("Retrieved {} context chunks", context_texts.len()),
    });

    let client = ollama::Ollama::new();
    let model = client.completion(chat_model);

    let agent = rig::agent::AgentBuilder::new(model)
        .preamble("You are a precise RAG assistant. Answer the question based strictly on the provided context.")
        .build();

    let final_prompt = format!(
        "Context:\n{}\n\nQuestion: {}\nAnswer:",
        combined_context, query
    );

    let response = agent.prompt(&final_prompt).await?;
    let final_output = response.output().trim().to_string();

    history.push(MicroRagStepRecord {
        step_name: "llm_generation".to_string(),
        input_value: final_prompt,
        generated_output: final_output.clone(),
    });

    Ok(MicroRagResult {
        query: query.to_string(),
        retrieved_context: context_texts,
        final_output,
        history,
    })
}
```

### 2. Executable Runner (`src/main.rs`)

```rust
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

    if let Some(parent) = Path::new(db_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut db = MicroRagDatabase::init(db_path)?;
    
    let sample_doc = "Autogen is a generalized multi-agent conversation framework designed to simplify the development of high-performance multi-agent applications, demonstrating power in Math Problem Solving (A1).";
    let doc_id = db.insert_document(sample_doc, "autogen_intro.txt", 0)?;

    if let Ok(emb) = get_embedding(ollama_host, embedding_model, sample_doc).await {
        db.insert_chunk(doc_id, sample_doc, &emb)?;
        println!("✅ Sample document indexed into SQLite & vector store.");
    }

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

            let json_audit = serde_json::to_string_pretty(&res.history).unwrap();
            println!("--- XAI Audit Trail (JSON) ---");
            println!("{}", json_audit);
        }
        Err(e) => eprintln!("Error during Micro-RAG execution: {}", e),
    }

    Ok(())
}
```

---

## 🧪 Testing

Run automated tests for cosine similarity calculation and database operations:

```bash
cargo test
```

---

## 🏃 Running the Application

1. **Pull required models in Ollama:**
   ```bash
   ollama pull nomic-embed-text:latest
   ollama pull qwen3.5:4b
   ```

2. **Clean previous database files (if any exist):**
   ```powershell
   Remove-Item -Recurse -Force .\storage\
   ```

3. **Build and Run:**
   ```bash
   cargo run
   ```
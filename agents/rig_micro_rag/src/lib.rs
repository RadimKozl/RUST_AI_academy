use anyhow::{anyhow, Context, Result};
use rig::providers::ollama;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Step recording for XAI / Neo4j memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroRagStepRecord {
    pub step_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// Result of the Micro-RAG pipeline execution
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

/// SQLite persistence manager for Micro-RAG documents and chunks
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

    /// Searching for the nearest chunks using cosine similarity
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

/// Calculation of the cosine similarity of two vectors
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

/// Helper function to retrieve an embedding from the Ollama API
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

/// Main function that triggers the Micro-RAG pipeline with a Rig agent
pub async fn run_micro_rag(
    db_path: &str,
    ollama_host: &str,
    embedding_model: &str,
    chat_model: &str,
    query: &str,
) -> Result<MicroRagResult> {
    let mut history = Vec::new();

    // 1. Vectorization of the user query
    let query_emb = get_embedding(ollama_host, embedding_model, query).await?;

    history.push(MicroRagStepRecord {
        step_name: "query_embedding".to_string(),
        input_value: query.to_string(),
        generated_output: format!("Vector generated with dimension {}", query_emb.len()),
    });

    // 2. Retrieving relevant context from the SQLite database
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

    // 3. Initialization of the Rig agent and response generation
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
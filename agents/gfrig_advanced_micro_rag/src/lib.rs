use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sqlx::{SqlitePool, Row};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagStepRecord {
    pub step_name: String,
    pub input_data: String,
    pub output_data: String,
}

/// Extended state corresponding to an advanced LangGraph RAG pipeline
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdvancedRagState {
    pub question: String,
    pub documents: Vec<String>,
    pub context: String,
    pub answer: String,
    pub critic_score: f32,
    pub iteration: i32,
    pub audit_trail: Vec<RagStepRecord>,
}

/// Asynchronous SQLite database manager using sqlx
#[derive(Clone)]
pub struct AdvancedRagDatabase {
    pool: SqlitePool,
}

impl AdvancedRagDatabase {
    pub async fn init<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let path_str = db_path.as_ref().to_str().ok_or_else(|| anyhow!("Invalid DB path"))?;
        let database_url = format!("sqlite://{}?mode=rwc", path_str);

        let pool = SqlitePool::connect(&database_url)
            .await
            .with_context(|| "Failed to connect to SQLite database via sqlx")?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source TEXT
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS vector_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                chunk_text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(id)
            )",
        )
        .execute(&pool)
        .await?;

        Ok(Self { pool })
    }

    pub async fn insert_document(&self, content: &str, source: &str) -> Result<i64> {
        let res = sqlx::query("INSERT INTO documents (content, source) VALUES (?1, ?2)")
            .bind(content)
            .bind(source)
            .execute(&self.pool)
            .await?;
        Ok(res.last_insert_rowid())
    }

    pub async fn insert_chunk(&self, doc_id: i64, chunk_text: &str, embedding: &[f32]) -> Result<()> {
        let emb_json = serde_json::to_string(embedding)?;
        sqlx::query("INSERT INTO vector_chunks (doc_id, chunk_text, embedding_json) VALUES (?1, ?2, ?3)")
            .bind(doc_id)
            .bind(chunk_text)
            .bind(emb_json)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn search(&self, query_emb: &[f32], limit: usize) -> Result<Vec<(String, f32)>> {
        let rows = sqlx::query("SELECT chunk_text, embedding_json FROM vector_chunks")
            .fetch_all(&self.pool)
            .await?;

        let mut scored_chunks = Vec::new();
        for row in rows {
            let chunk_text: String = row.get("chunk_text");
            let emb_json: String = row.get("embedding_json");
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

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    embedding: Option<Vec<f32>>,
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
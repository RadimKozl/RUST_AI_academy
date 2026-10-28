use anyhow::{anyhow, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;

// --- DATA STRUCTURES FOR OLLAMA API ---

#[derive(Serialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub prompt: String,
}

#[derive(Deserialize)]
pub struct EmbeddingResponse {
    pub embedding: Option<Vec<f32>>,
}

#[derive(Serialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct GenerateResponse {
    pub response: String,
}

// --- STRUCTURES FOR DATABASE AND SEARCH ---

#[derive(Debug, Clone)]
pub struct DocumentRecord {
    pub id: i64,
    pub path: String,
    pub content: String,
    pub summary: String,
}

#[derive(Debug, Clone)]
pub struct ChunkRecord {
    pub id: i64,
    pub doc_id: i64,
    pub chunk_text: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub doc_id: i64,
    pub path: String,
    pub summary: String,
    pub chunk_text: String,
    pub score: f32,
}

// --- CLIENT FOR OLLAMA API ---

pub struct OllamaClient {
    client: reqwest::Client,
    host: String,
    embedding_model: String,
    chat_model: String,
    image_model: String,
}

impl OllamaClient {
    pub fn new(host: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            host: host.trim_end_matches('/').to_string(),
            embedding_model: "nomic-embed-text:latest".to_string(),
            chat_model: "gemma3n:e4b".to_string(),
            image_model: "qwen3-vl:2b".to_string(),
        }
    }

    pub async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.host);
        let payload = EmbeddingRequest {
            model: self.embedding_model.clone(),
            prompt: text.to_string(),
        };

        let res: EmbeddingResponse = self.client.post(&url).json(&payload).send().await?.json().await?;
        res.embedding.ok_or_else(|| anyhow!("Ollama returns an empty embedding"))
    }

    pub async fn generate_text(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/api/generate", self.host);
        let payload = GenerateRequest {
            model: self.chat_model.clone(),
            prompt: prompt.to_string(),
            stream: false,
            images: None,
        };

        let res: GenerateResponse = self.client.post(&url).json(&payload).send().await?.json().await?;
        Ok(res.response)
    }

    pub async fn analyze_image(&self, image_bytes: &[u8]) -> Result<String> {
        use base64::engine::general_purpose::STANDARD;
        use base64::Engine;

        let url = format!("{}/api/generate", self.host);
        let b64_img = STANDARD.encode(image_bytes);

        let payload = GenerateRequest {
            model: self.image_model.clone(),
            prompt: "Describe the content of the image in detail:".to_string(),
            stream: false,
            images: Some(vec![b64_img]),
        };

        let res: GenerateResponse = self.client.post(&url).json(&payload).send().await?.json().await?;
        Ok(res.response)
    }
}

// --- MANAGEMENT OF SQLITE DATABASE AND VECTORS ---

pub struct RagDatabase {
    conn: Connection,
}

impl RagDatabase {
    pub fn init<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                content TEXT,
                summary TEXT
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                chunk_text TEXT,
                embedding_json TEXT,
                FOREIGN KEY(doc_id) REFERENCES documents(id)
            )",
            [],
        )?;

        Ok(Self { conn })
    }

    pub fn insert_document(&mut self, path: &str, content: &str, summary: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT OR REPLACE INTO documents (path, content, summary) VALUES (?1, ?2, ?3)",
            params![path, content, summary],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn insert_chunk(&mut self, doc_id: i64, chunk_text: &str, embedding: &[f32]) -> Result<()> {
        let emb_json = serde_json::to_string(embedding)?;
        self.conn.execute(
            "INSERT INTO chunks (doc_id, chunk_text, embedding_json) VALUES (?1, ?2, ?3)",
            params![doc_id, chunk_text, emb_json],
        )?;
        Ok(())
    }

    pub fn search(&self, query_emb: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.doc_id, d.path, d.summary, c.chunk_text, c.embedding_json 
             FROM chunks c JOIN documents d ON c.doc_id = d.id",
        )?;

        let rows = stmt.query_map([], |row| {
            let doc_id: i64 = row.get(0)?;
            let path: String = row.get(1)?;
            let summary: String = row.get(2)?;
            let chunk_text: String = row.get(3)?;
            let emb_json: String = row.get(4)?;
            Ok((doc_id, path, summary, chunk_text, emb_json))
        })?;

        let mut results = Vec::new();
        for row in rows {
            let (doc_id, path, summary, chunk_text, emb_json) = row?;
            let emb: Vec<f32> = serde_json::from_str(&emb_json).unwrap_or_default();

            if !emb.is_empty() {
                let score = cosine_similarity(query_emb, &emb);
                results.push(SearchResult {
                    doc_id,
                    path,
                    summary,
                    chunk_text,
                    score,
                });
            }
        }

        // Sort by score in descending order
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }
}

// --- HELPER FUNCTIONS: CHUNKING & VECTOR MATH ---

pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return chunks;
    }

    let mut start = 0;
    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk);

        if end == chars.len() {
            break;
        }
        start = if end > overlap { end - overlap } else { end };
    }
    chunks
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot_product += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a.sqrt() * norm_b.sqrt())
    }
}
// src/lib.rs
use anyhow::{anyhow, Context, Result};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use sqlx::{Connection, Row, SqlitePool}; // Added import of the Connection trait
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagStepRecord {
    pub step_name: String,
    pub input_data: String,
    pub output_data: String,
}

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityNode {
    pub id: i64,
    pub name: String,
    pub category: String,
    pub description: String,
}

/// Asynchronous SQLite database manager combined with Petgraph G-RAG
#[derive(Clone)]
pub struct AdvancedRagDatabase {
    pool: SqlitePool,
    db_path_str: String,
}

impl AdvancedRagDatabase {
    pub async fn init<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let path_str = db_path.as_ref().to_str().ok_or_else(|| anyhow!("Invalid DB path"))?;
        let database_url = format!("sqlite://{}?mode=rwc", path_str);

        let pool = SqlitePool::connect(&database_url)
            .await
            .with_context(|| "Failed to connect to SQLite database via sqlx")?;

        // 1. Vector and document tables
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

        // 2. Graph tables for G-RAG (Entities and Relationships)
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT,
                description TEXT
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER,
                target_id INTEGER,
                relation_type TEXT,
                FOREIGN KEY(source_id) REFERENCES entities(id),
                FOREIGN KEY(target_id) REFERENCES entities(id)
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS chunk_entities (
                chunk_id INTEGER,
                entity_id INTEGER,
                FOREIGN KEY(chunk_id) REFERENCES vector_chunks(id),
                FOREIGN KEY(entity_id) REFERENCES entities(id)
            )",
        )
        .execute(&pool)
        .await?;

        Ok(Self { pool, db_path_str: path_str.to_string() })
    }

    pub async fn insert_document(&self, content: &str, source: &str) -> Result<i64> {
        let res = sqlx::query("INSERT INTO documents (content, source) VALUES (?1, ?2)")
            .bind(content)
            .bind(source)
            .execute(&self.pool)
            .await?;
        Ok(res.last_insert_rowid())
    }

    pub async fn insert_chunk(&self, doc_id: i64, chunk_text: &str, embedding: &[f32]) -> Result<i64> {
        let emb_json = serde_json::to_string(embedding)?;
        let res = sqlx::query("INSERT INTO vector_chunks (doc_id, chunk_text, embedding_json) VALUES (?1, ?2, ?3)")
            .bind(doc_id)
            .bind(chunk_text)
            .bind(emb_json)
            .execute(&self.pool)
            .await?;
        Ok(res.last_insert_rowid())
    }

    /// Inserting entities and relationships into an SQLite graph structure
    pub async fn insert_graph_knowledge(&self, chunk_id: i64, entities: &[(&str, &str, &str)], relations: &[(&str, &str, &str)]) -> Result<()> {
        let mut conn = sqlx::sqlite::SqliteConnection::connect(&format!("sqlite://{}", self.db_path_str)).await?;

        for (name, category, desc) in entities {
            sqlx::query("INSERT OR IGNORE INTO entities (name, category, description) VALUES (?1, ?2, ?3)")
                .bind(name)
                .bind(category)
                .bind(desc)
                .execute(&mut conn)
                .await?;

            let entity_row = sqlx::query("SELECT id FROM entities WHERE name = ?1")
                .bind(name)
                .fetch_one(&mut conn)
                .await?;
            let entity_id: i64 = entity_row.get("id");

            sqlx::query("INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id) VALUES (?1, ?2)")
                .bind(chunk_id)
                .bind(entity_id)
                .execute(&mut conn)
                .await?;
        }

        for (src, rel_type, tgt) in relations {
            let src_row = sqlx::query("SELECT id FROM entities WHERE name = ?1").bind(src).fetch_optional(&mut conn).await?;
            let tgt_row = sqlx::query("SELECT id FROM entities WHERE name = ?1").bind(tgt).fetch_optional(&mut conn).await?;

            if let (Some(s_row), Some(t_row)) = (src_row, tgt_row) {
                let src_id: i64 = s_row.get("id");
                let tgt_id: i64 = t_row.get("id");

                sqlx::query("INSERT INTO relations (source_id, target_id, relation_type) VALUES (?1, ?2, ?3)")
                    .bind(src_id)
                    .bind(tgt_id)
                    .bind(rel_type)
                    .execute(&mut conn)
                    .await?;
            }
        }

        Ok(())
    }

    /// Combined G-RAG Search: Vector Search + Petgraph Multi-Hop Traversal
    pub async fn search_graph_rag(&self, query_emb: &[f32], keyword: &str, limit: usize) -> Result<String> {
        let mut conn = sqlx::sqlite::SqliteConnection::connect(&format!("sqlite://{}", self.db_path_str)).await?;

        // 1. Classic vector search over chunks
        let rows = sqlx::query("SELECT id, chunk_text, embedding_json FROM vector_chunks")
            .fetch_all(&mut conn)
            .await?;

        let mut scored_chunks = Vec::new();
        for row in rows {
            let chunk_id: i64 = row.get("id");
            let chunk_text: String = row.get("chunk_text");
            let emb_json: String = row.get("embedding_json");
            let emb: Vec<f32> = serde_json::from_str(&emb_json).unwrap_or_default();
            if !emb.is_empty() {
                let score = cosine_similarity(query_emb, &emb);
                scored_chunks.push((chunk_id, chunk_text, score));
            }
        }
        scored_chunks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored_chunks.truncate(limit);

        // 2. Hydrating data into Petgraph for graph context
        let mut graph: DiGraph<EntityNode, String> = DiGraph::new();
        let mut node_map: HashMap<i64, NodeIndex> = HashMap::new();

        let ent_rows = sqlx::query("SELECT id, name, category, description FROM entities").fetch_all(&mut conn).await?;
        for r in ent_rows {
            let node = EntityNode {
                id: r.get("id"),
                name: r.get("name"),
                category: r.get("category"),
                description: r.get("description"),
            };
            let idx = graph.add_node(node.clone());
            node_map.insert(node.id, idx);
        }

        let rel_rows = sqlx::query("SELECT source_id, target_id, relation_type FROM relations").fetch_all(&mut conn).await?;
        for r in rel_rows {
            let src: i64 = r.get("source_id");
            let tgt: i64 = r.get("target_id");
            let rel: String = r.get("relation_type");
            if let (Some(&s_idx), Some(&t_idx)) = (node_map.get(&src), node_map.get(&tgt)) {
                graph.add_edge(s_idx, t_idx, rel);
            }
        }

        // 3. Constructing the output context (Vectors + Graph relations)
        let mut combined_output = String::new();
        combined_output.push_str("=== VECTOR RETRIEVAL CHUNKS ===\n");
        for (_, text, score) in &scored_chunks {
            combined_output.push_str(&format!("[Score: {:.2}] {}\n", score, text));
        }

        combined_output.push_str("\n=== GRAPH RAG TOPOLOGY (Petgraph Context) ===\n");
        let kw_lower = keyword.to_lowercase();
        for (_, &idx) in &node_map {
            let node_data = &graph[idx];
            if node_data.name.to_lowercase().contains(&kw_lower) || node_data.description.to_lowercase().contains(&kw_lower) {
                combined_output.push_str(&format!("Entity: {} [{}]\nDescription: {}\n", node_data.name, node_data.category, node_data.description));
                
                // Traversing neighbors (1-hop traversal)
                for edge in graph.edges_directed(idx, petgraph::Direction::Outgoing) {
                    let target = edge.target();
                    combined_output.push_str(&format!("  --[{}]--> {} [{}]\n", edge.weight(), graph[target].name, graph[target].category));
                }
            }
        }

        Ok(combined_output)
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
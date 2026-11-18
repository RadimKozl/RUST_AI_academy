use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Reasoning Memory: ReasoningTrace -> ReasoningStep -> ToolCall
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_id: usize,
    pub agent_id: String,
    pub thought: String,
    pub tool_called: Option<String>,
    pub confidence: f64,
}

/// Short-term Memory: Conversation -> Message -> Next_message (AGON Intent)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgonIntentMessage {
    pub id: usize,
    pub sender: String,
    pub target: String,
    pub intent_type: String,
    pub payload: String,
    pub confidence: f64,
}

/// Long-term Memory (Graph Node and Edge): Entity -> RELATED TO -> Entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEntity {
    pub name: String,
    pub entity_type: String, // e.g., "Concept", "ArxivPaper", "AgonConstruct"
    pub description: String,
    pub embedding: Vec<f32>, // For hybrid vector-graph search
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRelation {
    pub source: String,
    pub relation: String, // e.g., "RELATED_TO", "IMPLEMENTS", "CONTRADICTS"
    pub target: String,
    pub weight: f32,
}

/// Complete, isolated GraphRAG (G-RAG) for a single agent
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersonalGraphRag {
    pub owner_agent: String,
    pub reasoning_trace: Vec<ReasoningStep>,
    pub entities: HashMap<String, GraphEntity>,
    pub relations: Vec<GraphRelation>,
}

impl PersonalGraphRag {
    pub fn new(owner: &str) -> Self {
        Self {
            owner_agent: owner.to_string(),
            reasoning_trace: Vec::new(),
            entities: HashMap::new(),
            relations: Vec::new(),
        }
    }

    /// Inserts an entity and creates a simple deterministic hash-embedding (replace `rig::embeddings` in production)
    pub fn upsert_entity(&mut self, name: &str, entity_type: &str, description: &str) {
        let embedding = Self::compute_lightweight_embedding(description);
        self.entities.insert(
            name.to_string(),
            GraphEntity {
                name: name.to_string(),
                entity_type: entity_type.to_string(),
                description: description.to_string(),
                embedding,
            },
        );
    }

    pub fn add_relation(&mut self, source: &str, relation: &str, target: &str, weight: f32) {
        self.relations.push(GraphRelation {
            source: source.to_string(),
            relation: relation.to_string(),
            target: target.to_string(),
            weight,
        });
    }

    /// GraphRAG query: 1. Vector similarity for input nodes -> 2. 1-hop graph expansion along edges
    pub fn query_graph_rag(&self, query: &str, top_k: usize) -> String {
        if self.entities.is_empty() {
            return format!("[G-RAG {}]: Paměť je zatím prázdná.", self.owner_agent);
        }

        let q_emb = Self::compute_lightweight_embedding(query);
        let mut scored_nodes: Vec<(&GraphEntity, f32)> = self
            .entities
            .values()
            .map(|ent| (ent, Self::cosine_similarity(&q_emb, &ent.embedding)))
            .collect();

        scored_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut context_lines = Vec::new();
        let mut visited_neighbors = HashSet::new();

        for (entity, score) in scored_nodes.into_iter().take(top_k) {
            context_lines.push(format!(
                "Node [{} : {}] (sim={:.2}): {}",
                entity.name, entity.entity_type, score, entity.description
            ));

            // Graph expansion (1-hop traversal)
            for rel in &self.relations {
                if rel.source == entity.name && visited_neighbors.insert(rel.target.clone()) {
                    if let Some(target_ent) = self.entities.get(&rel.target) {
                        context_lines.push(format!(
                            "  └─[:{} (w={:.2})]-> [{}]: {}",
                            rel.relation, rel.weight, target_ent.name, target_ent.description
                        ));
                    }
                }
            }
        }

        context_lines.join("\n")
    }

    fn compute_lightweight_embedding(text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; 32];
        for (i, byte) in text.to_lowercase().bytes().enumerate() {
            vec[i % 32] += (byte as f32) / 255.0;
        }
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        vec.iter_mut().for_each(|x| *x /= norm);
        vec
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}
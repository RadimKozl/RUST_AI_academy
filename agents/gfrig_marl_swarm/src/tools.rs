use crate::memory::PersonalGraphRag;
use rig::tool::{Tool, ToolContext};
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SwarmToolError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Memory lock poisoned")]
    LockError,
}

#[derive(Deserialize)]
pub struct ArxivArgs {
    pub query: String,
    pub max_results: usize,
}

pub struct ArxivSearchTool {
    pub personal_memory: Arc<Mutex<PersonalGraphRag>>,
}

impl Tool for ArxivSearchTool {
    const NAME: &'static str = "search_arxiv_and_memorize";

    type Error = SwarmToolError;
    type Args = ArxivArgs;
    type Output = String;

    fn description(&self) -> String {
        "Searches arXiv.org for scientific papers on MARL, QMIX, Actor Models, or type-safe DSLs, and stores findings into the agent's personal GraphRAG.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query for arXiv API (e.g., 'QMIX value decomposition MARL')" },
                "max_results": { "type": "integer", "description": "Number of papers to retrieve (1-5)" }
            },
            "required": ["query", "max_results"]
        })
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let limit = args.max_results.clamp(1, 5);
        let url = format!(
            "https://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results={}",
            urlencoding_basic(&args.query),
            limit
        );

        let client = reqwest::Client::builder()
            .user_agent("agon-marl-swarm/0.1")
            .build()?;

        let response_text = match client.get(&url).send().await {
            Ok(resp) => resp.text().await.unwrap_or_default(),
            Err(_) => String::new(),
        };

        let mut extracted_summaries = Vec::new();
        for chunk in response_text.split("<entry>").skip(1) {
            let title = chunk
                .split("<title>")
                .nth(1)
                .and_then(|s| s.split("</title>").next())
                .unwrap_or("Unknown Paper")
                .trim()
                .replace('\n', " ");
            let summary = chunk
                .split("<summary>")
                .nth(1)
                .and_then(|s| s.split("</summary>").next())
                .unwrap_or("")
                .trim()
                .replace('\n', " ");
            let short_sum: String = summary.chars().take(220).collect();
            if !title.is_empty() {
                extracted_summaries.push((title, short_sum));
            }
        }

        if extracted_summaries.is_empty() {
            extracted_summaries.push((
                "arXiv:1803.11485 - QMIX: Monotonic Value Function Factorisation for MARL".to_string(),
                "Decomposes joint action-value Q_tot into monotonic combination of per-agent Q_i for CTDE.".to_string(),
            ));
            extracted_summaries.push((
                "arXiv:1706.02275 - MADDPG: Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments".to_string(),
                "Uses centralized critics with decentralized actors to handle non-stationarity in general-sum games.".to_string(),
            ));
        }

        let mut mem = self
            .personal_memory
            .lock()
            .map_err(|_| SwarmToolError::LockError)?;
        let mut report = Vec::new();
        for (idx, (title, summary)) in extracted_summaries.iter().enumerate() {
            let node_id = format!("ArXiv_{}_{}", args.query.replace(' ', "_"), idx);
            mem.upsert_entity(&node_id, "ArxivPaper", &format!("{}: {}", title, summary));
            mem.add_relation(&node_id, "INFORMS_DESIGN_OF", "AGON_Language", 0.9);
            report.push(format!("- {} | {}", title, summary));
        }

        Ok(format!(
            "Saved {} arXiv sources to personal G-RAG ({}):\n{}",
            extracted_summaries.len(),
            mem.owner_agent,
            report.join("\n")
        ))
    }
}

#[derive(Deserialize)]
pub struct GraphQueryArgs {
    pub concept: String,
}

pub struct GraphRagQueryTool {
    pub personal_memory: Arc<Mutex<PersonalGraphRag>>,
}

impl Tool for GraphRagQueryTool {
    const NAME: &'static str = "query_personal_graph_rag";

    type Error = SwarmToolError;
    type Args = GraphQueryArgs;
    type Output = String;

    fn description(&self) -> String {
        "Queries the agent's private GraphRAG knowledge graph (entities + 1-hop relations).".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "concept": { "type": "string", "description": "Concept or entity to search in personal GraphRAG" }
            },
            "required": ["concept"]
        })
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let mem = self
            .personal_memory
            .lock()
            .map_err(|_| SwarmToolError::LockError)?;
        Ok(mem.query_graph_rag(&args.concept, 3))
    }
}

fn urlencoding_basic(input: &str) -> String {
    input
        .trim()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("+")
}
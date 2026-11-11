use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;

/// XAI record of a single node or decision in a graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiStep {
    pub node_name: String,
    pub input_data: String,
    pub generated_output: String,
    pub reasoning: String,
}

/// Shared state of an agent traversing a graph (equivalent to LangGraph State)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub thread_id: String,
    pub user_input: String,
    pub improved_query: String,
    pub chat_history: Vec<(String, String)>,
    pub summary: Option<String>,
    pub final_answer: String,
    pub xai_audit_trail: Vec<XaiStep>,
}

impl AgentState {
    pub fn new(thread_id: &str, user_input: &str) -> Self {
        Self {
            thread_id: thread_id.to_string(),
            user_input: user_input.to_string(),
            improved_query: String::new(),
            chat_history: Vec::new(),
            summary: None,
            final_answer: String::new(),
            xai_audit_trail: Vec::new(),
        }
    }
}

/// Trait defining a workflow node with support for HRTB lifetimes
pub trait WorkflowNode: Send + Sync {
    fn execute<'a>(&'a self, state: &'a mut AgentState) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

/// Wrapper for closure implementing WorkflowNode
pub struct FnNode<F>
where
    F: for<'a> Fn(&'a mut AgentState) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> + Send + Sync,
{
    f: F,
}

impl<F> WorkflowNode for FnNode<F>
where
    F: for<'a> Fn(&'a mut AgentState) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> + Send + Sync,
{
    fn execute<'a>(&'a self, state: &'a mut AgentState) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        (self.f)(state)
    }
}

/// Object-Oriented Workflow Graph Orchestrator
pub struct WorkflowGraph {
    nodes: Vec<(String, Box<dyn WorkflowNode>)>,
}

impl WorkflowGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Registering a node into the graph using the .add_node() method, with support for arbitrary state lifecycles
    pub fn add_node<F>(&mut self, name: &str, f: F) -> &mut Self
    where
        F: for<'a> Fn(&'a mut AgentState) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> + Send + Sync + 'static,
    {
        self.nodes.push((name.to_string(), Box::new(FnNode { f })));
        self
    }

    /// Sequential execution of the entire graph
    pub async fn invoke(&self, state: &mut AgentState) -> Result<()> {
        println!("🚀 [WorkflowGraph] Starting execution flow...");
        for (node_name, node) in &self.nodes {
            println!("  ├── Executing node: [{}]", node_name);
            node.execute(state).await?;
        }
        println!("✅ [WorkflowGraph] Execution flow completed successfully.\n");
        Ok(())
    }
}

/// SQLite persistent manager for XAI audits and logs
pub struct SqliteStorageManager {
    db_path: String,
}

impl SqliteStorageManager {
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)
            .with_context(|| format!("Failed to open DB at {}", db_path))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS workflow_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                node_name TEXT NOT NULL,
                input_data TEXT NOT NULL,
                output_data TEXT NOT NULL,
                reasoning TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        Ok(Self {
            db_path: db_path.to_string(),
        })
    }

    pub fn save_xai_step(&self, thread_id: &str, step: &XaiStep) -> Result<()> {
        let conn = Connection::open(&self.db_path)?;
        conn.execute(
            "INSERT INTO workflow_logs (thread_id, node_name, input_data, output_data, reasoning) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![thread_id, step.node_name, step.input_data, step.generated_output, step.reasoning],
        )?;
        Ok(())
    }
}
# Rust Agentic Engine: LLM Semantic Router & Conditional Routing

This project implements an advanced agentic workflow engine in **Rust**, porting LangGraph-style conditional edge patterns (semantic routing) into a high-performance backend. It combines structured session state management (`graph-flow`), local LLM semantic intelligence (`rig` with Ollama), and comprehensive Explainable AI (XAI) audit logging.

---

## Architecture & Core Concepts

Unlike static or parallel (Fork-Join) workflows, conditional routing allows an agent to **dynamically decide its execution path at runtime** based on semantic evaluation of the user's query.

### Key Components

1. **`UnifiedNode`**: A flexible abstraction layer supporting three operational modes:
   - **`CustomTool`**: Deterministic code blocks, APIs, timestamps, or state guards.
   - **`LlmRouter`**: Semantic router powered by Ollama via `rig`, analyzing input text to decide the next execution branch (`b` or `c`).
   - **`LlmAgent`**: Final supervisor node that synthesizes accumulated context into a structured human-readable answer.
2. **State Guards**: Mechanism inside branch nodes (`Node B` and `Node C`) that inspect the session's `router_decision` variable to determine whether to execute branch-specific logic or skip gracefully.
3. **XAI Audit Trail**: Every node execution, input question, and generated output is recorded chronologically into a JSON-serializable audit structure for monitoring and graph database integration (e.g., Neo4j).

---

## Code Overview

### 1. Core Library (`src/lib.rs`)

Handles session context storage, graph topology building, mermaid diagram generation, and task dispatching.

```rust
use anyhow::Result;
use async_trait::async_trait;
use graph_flow::{
    Context, FlowRunner, GraphBuilder, GraphError, InMemorySessionStorage,
    NextAction, Session, SessionStorage, Task, TaskResult,
};
use rig::providers::ollama;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStepRecord {
    pub task_id: String,
    pub input_data: String,
    pub output_data: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkflowState {
    pub query: String,
    pub question: String,
    pub data: Vec<String>,
    pub final_answer: String,
    pub architecture_mermaid: String,
    pub audit_trail: Vec<GraphStepRecord>,
}

pub struct DynamicWorkflowBuilder {
    builder: GraphBuilder,
    edges: Vec<(String, String)>,
}

impl DynamicWorkflowBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            builder: GraphBuilder::new(name),
            edges: Vec::new(),
        }
    }

    pub fn add_task(mut self, task: Arc<dyn Task>) -> Self {
        self.builder = self.builder.add_task(task);
        self
    }

    pub fn add_edge(mut self, from: &str, to: &str) -> Self {
        self.builder = self.builder.add_edge(from, to);
        self.edges.push((from.to_string(), to.to_string()));
        self
    }

    pub fn build(self) -> Result<graph_flow::Graph, GraphError> {
        self.builder.build()
    }

    pub fn generate_mermaid(&self) -> String {
        let mut mermaid = String::from("graph TD\n");
        if let Some(first) = self.edges.first() {
            mermaid.push_str(&format!("    START --> {}\n", first.0));
        }
        for (from, to) in &self.edges {
            mermaid.push_str(&format!("    {} --> {}\n", from, to));
        }
        if let Some(last) = self.edges.last() {
            mermaid.push_str(&format!("    {} --> END\n", last.1));
        }
        mermaid
    }
}

pub type ToolFn = Arc<
    dyn Fn(Context) -> Pin<Box<dyn Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
>;

pub enum NodeMode {
    CustomTool {
        action_name: String,
        logic: ToolFn,
    },
    LlmAgent {
        model_name: String,
        preamble: String,
    },
    LlmRouter {
        model_name: String,
        preamble: String,
    },
}

pub struct UnifiedNode {
    pub id_str: String,
    pub label: String,
    pub mode: NodeMode,
}

#[async_trait]
impl Task for UnifiedNode {
    fn id(&self) -> &str {
        &self.id_str
    }

    async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
        let _query = context.get::<String>("query").unwrap_or_default();
        let question = context.get::<String>("question").unwrap_or_default();
        let mut data = context.get::<Vec<String>>("data").unwrap_or_default();

        let (output_data, next_action) = match &self.mode {
            NodeMode::CustomTool { action_name, logic } => {
                match logic(context.clone()).await {
                    Ok(res) => {
                        let result_str = format!("[{}] {}", action_name, res);
                        data.push(result_str.clone());
                        context.set("data", data)?;
                        (result_str, NextAction::ContinueAndExecute)
                    }
                    Err(e) => {
                        let err_msg = format!("Tool [{}] failed -> {}", action_name, e);
                        data.push(err_msg.clone());
                        context.set("data", data)?;
                        (err_msg, NextAction::ContinueAndExecute)
                    }
                }
            }

            NodeMode::LlmRouter { model_name, preamble } => {
                let client = ollama::Ollama::new();
                let model = client.completion(model_name);

                let agent = rig::agent::AgentBuilder::new(model)
                    .preamble(preamble)
                    .build();

                let prompt = format!("Question: {}\nDetermine the routing decision (return 'b' if starts with 'What is', otherwise 'c'):", question);

                match agent.prompt(&prompt).await {
                    Ok(response) => {
                        let decision = response.output().trim().to_lowercase();
                        context.set("router_decision", decision.clone())?;
                        let result_str = format!("Router decided path: {}", decision);
                        data.push(result_str.clone());
                        context.set("data", data)?;
                        (result_str, NextAction::ContinueAndExecute)
                    }
                    Err(e) => {
                        let err_msg = format!("Router Error: {}", e);
                        (err_msg, NextAction::ContinueAndExecute)
                    }
                }
            }

            NodeMode::LlmAgent { model_name, preamble } => {
                let client = ollama::Ollama::new();
                let model = client.completion(model_name);

                let agent = rig::agent::AgentBuilder::new(model)
                    .preamble(preamble)
                    .build();

                let prompt = format!("Context Data: {:?}\nQuestion: {}\nSynthesize final answer:", data, question);

                match agent.prompt(&prompt).await {
                    Ok(response) => {
                        let output = response.output().trim().to_string();
                        context.set("final_answer", output.clone())?;
                        (output, NextAction::End)
                    }
                    Err(e) => {
                        let err_msg = format!("LLM Execution Error: {}", e);
                        (err_msg, NextAction::End)
                    }
                }
            }
        };

        let mut audit = context.get::<Vec<GraphStepRecord>>("audit_trail").unwrap_or_default();
        audit.push(GraphStepRecord {
            task_id: self.id_str.clone(),
            input_data: question.clone(),
            output_data: output_data.clone(),
        });
        context.set("audit_trail", audit)?;

        Ok(TaskResult::new(Some(output_data), next_action))
    }
}

pub async fn run_conditional_workflow(
    user_query: &str,
    question: &str,
    nodes: Vec<UnifiedNode>,
    edges: Vec<(&str, &str)>,
    entry_node_id: &str,
) -> Result<WorkflowState> {
    if nodes.is_empty() {
        anyhow::bail!("Nodes list cannot be empty!");
    }

    let mut workflow_builder = DynamicWorkflowBuilder::new("conditional_workflow");

    for node in nodes {
        workflow_builder = workflow_builder.add_task(Arc::new(node));
    }

    for (from, to) in edges {
        workflow_builder = workflow_builder.add_edge(from, to);
    }

    let mermaid_code = workflow_builder.generate_mermaid();
    let graph = workflow_builder.build()?;

    let storage = Arc::new(InMemorySessionStorage::new());
    let session_id = "session_conditional_1".to_string();

    let session = Session::new_from_task(session_id.clone(), entry_node_id);

    session.context.set("query", user_query.to_string())?;
    session.context.set("question", question.to_string())?;
    session.context.set("data", Vec::<String>::new())?;
    session.context.set("final_answer", "".to_string())?;
    session.context.set("router_decision", "".to_string())?;
    session.context.set("audit_trail", Vec::<GraphStepRecord>::new())?;
    session.context.set("architecture_mermaid", mermaid_code)?;

    SessionStorage::save(&*storage, session).await?;

    let runner = FlowRunner::new(Arc::new(graph), storage.clone());
    runner.run(&session_id).await?;

    let final_session = SessionStorage::get(&*storage, &session_id).await.unwrap().unwrap();
    let ctx = &final_session.context;

    let final_state = WorkflowState {
        query: ctx.get::<String>("query").unwrap_or_default(),
        question: ctx.get::<String>("question").unwrap_or_default(),
        data: ctx.get::<Vec<String>>("data").unwrap_or_default(),
        final_answer: ctx.get::<String>("final_answer").unwrap_or_default(),
        architecture_mermaid: ctx.get::<String>("architecture_mermaid").unwrap_or_default(),
        audit_trail: ctx.get::<Vec<GraphStepRecord>>("audit_trail").unwrap_or_default(),
    };

    Ok(final_state)
}

pub async fn run_graph_workflow(
    question: &str,
    nodes: Vec<UnifiedNode>,
    edges: Vec<(&str, &str)>,
    entry_node_id: &str,
) -> Result<WorkflowState> {
    run_conditional_workflow("Workflow Execution", question, nodes, edges, entry_node_id).await
}
```

---

### 2. Execution Entry Point (`src/main.rs`)

Sets up nodes, initializes Ollama integration via Rig, executes the workflow, and outputs structured XAI JSON audit data.

```rust
use anyhow::Result;
use chrono::Local;
use dotenvy::dotenv;
use gfrig_router_node::{run_graph_workflow, NodeMode, UnifiedNode};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST AGENTIC WORKFLOW: LLM SEMANTIC ROUTER (Rig + Ollama) ===\n");

    let user_question = "What is artificial intelligence?";

    let node_a = UnifiedNode {
        id_str: "node_a".to_string(),
        label: "Node A (Start)".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node A".to_string(),
            logic: Arc::new(|_ctx| {
                Box::pin(async move {
                    let now = Local::now().format("%S.%f").to_string();
                    Ok(format!("A - Timestamp: {}", now))
                })
            }),
        },
    };

    let node_llm_router = UnifiedNode {
        id_str: "node_router".to_string(),
        label: "LLM Semantic Router".to_string(),
        mode: NodeMode::LlmRouter {
            model_name: "glm4:9b".to_string(),
            preamble: "You are a precise routing agent. Analyze the given question. If the question asks for a definition or explanation starting with 'What is', reply strictly with 'b'. If it asks about a person, identity, or anything else, reply strictly with 'c'. Output ONLY the single letter 'b' or 'c'.".to_string(),
        },
    };

    let node_b = UnifiedNode {
        id_str: "node_b".to_string(),
        label: "Node B (Definition Branch)".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node B (Definition)".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    let decision = ctx.get::<String>("router_decision").unwrap_or_default();
                    if decision.contains("b") {
                        let now = Local::now().format("%S.%f").to_string();
                        Ok(format!("B (Executed by LLM choice) - Timestamp: {}", now))
                    } else {
                        Ok("B (Skipped - path not chosen)".to_string())
                    }
                })
            }),
        },
    };

    let node_c = UnifiedNode {
        id_str: "node_c".to_string(),
        label: "Node C (Alternative Branch)".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node C (Alternative)".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    let decision = ctx.get::<String>("router_decision").unwrap_or_default();
                    if decision.contains("c") {
                        let now = Local::now().format("%S.%f").to_string();
                        Ok(format!("C (Executed by LLM choice) - Timestamp: {}", now))
                    } else {
                        Ok("C (Skipped - path not chosen)".to_string())
                    }
                })
            }),
        },
    };

    let node_d = UnifiedNode {
        id_str: "node_d".to_string(),
        label: "Node D (AI Synthesis)".to_string(),
        mode: NodeMode::LlmAgent {
            model_name: "glm4:9b".to_string(),
            preamble: "You are an AI supervisor. Review the execution path and provide a concise summary of the workflow outcome.".to_string(),
        },
    };

    let nodes = vec![node_a, node_llm_router, node_b, node_c, node_d];

    let edges = vec![
        ("node_a", "node_router"),
        ("node_router", "node_b"),
        ("node_router", "node_c"),
        ("node_b", "node_d"),
        ("node_c", "node_d"),
    ];

    println!("🚀 Running LLM-Routed Workflow with Question: '{}'...\n", user_question);

    match run_graph_workflow(user_question, nodes, edges, "node_a").await {
        Ok(final_state) => {
            println!("--- 📊 Architecture (Mermaid) ---");
            println!("{}\n", final_state.architecture_mermaid);

            println!("--- 📝 Accumulated Execution Steps ---");
            for (i, item) in final_state.data.iter().enumerate() {
                println!("  [{}] {}", i + 1, item);
            }
            println!();

            println!("--- 🤖 Final LLM Synthesis ---");
            println!("{}\n", final_state.final_answer);

            let json_audit = serde_json::to_string_pretty(&final_state).unwrap();
            println!("--- 🔍 XAI Audit Trail (JSON) ---");
            println!("{}", json_audit);
        }
        Err(e) => eprintln!("Error during execution: {}", e),
    }

    Ok(())
}
```

---

## Getting Started & Running

1. **Ensure Ollama is running locally and your target model is downloaded:**
   ```bash
   ollama serve
   ollama pull glm4:9b
   ```

2. **Execute the project workflow via Cargo:**
   ```bash
   cargo run
   ```

3. **Run integration tests to verify graph behavior and conditional routing:**
   ```bash
   cargo test
   ```
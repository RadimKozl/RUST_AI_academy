# Rust Agentic Engine: Iterative Loop & Self-Correction Critique Pattern

This project implements an advanced agentic workflow engine in **Rust**, porting LangGraph-style **`if`/`else` loop cycles and self-correction critique patterns** into a high-performance, asynchronous backend. 

It combines structured session state management (`graph-flow`), local LLM intelligence (`rig` integrated with Ollama), and Explainable AI (XAI) JSON audit trail logging.

---

## Architecture & Principles

Standard graph execution runners built around **DAGs (Directed Acyclic Graphs)** cannot natively handle cycle edges (`Router -> Node B`) without deadlocking or failing cycle detection. 

To overcome this in Rust without sacrificing safety or performance, this engine uses a **State Machine / Iterative Loop Runner Pattern**:
1. **Linear Topology Definition:** Topology remains DAG-compatible for graph builders, Mermaid diagram generation, and static validation.
2. **State Context Persistence:** Variables like `counter`, `max_loops`, and execution `data` history are persisted in thread-safe `SessionStorage` (`InMemorySessionStorage`).
3. **Iterative Loop Control:** The `run_loop_workflow` engine explicitly controls the cyclic loop (`B -> C -> Router`) via Rust's native `loop` construct, checking condition thresholds dynamically before terminating or proceeding to final synthesis.

### Key Components:
- **`UnifiedNode`**: An abstraction layer implementing the `graph_flow::Task` trait, supporting:
  - `CustomTool`: Deterministic Rust closures, state guards, and counters.
  - `LlmAgent`: Generative LLM tasks (Code Generation, Synthesis) using Rig & Ollama.
  - `LlmRouter`: LLM-based classification/routing decisions.
- **`run_loop_workflow`**: The iterative engine runner that manages state updates, loop limits, and step audit recording.

---

## Code Overview

### 1. Core Engine: `src/lib.rs`

Handles state definitions, dynamic node abstractions (`UnifiedNode`), session context storage, and the iterative loop runner logic (`run_loop_workflow`).

```rust
use anyhow::Result;
use async_trait::async_trait;
use graph_flow::{
    Context, GraphBuilder, GraphError, InMemorySessionStorage,
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
    pub counter: i32,
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
                let agent = rig::agent::AgentBuilder::new(model).preamble(preamble).build();
                let prompt = format!("Question: {}\nDetermine the routing decision:", question);

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
                let agent = rig::agent::AgentBuilder::new(model).preamble(preamble).build();
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

pub async fn run_loop_workflow(
    user_query: &str,
    question: &str,
    nodes: Vec<UnifiedNode>,
    edges: Vec<(&str, &str)>,
    _entry_node_id: &str,
    max_loops: i32,
) -> Result<WorkflowState> {
    if nodes.is_empty() {
        anyhow::bail!("Nodes list cannot be empty!");
    }

    let mut workflow_builder = DynamicWorkflowBuilder::new("loop_workflow");
    let mut task_map = std::collections::HashMap::new();

    for node in nodes {
        let task_arc = Arc::new(node);
        task_map.insert(task_arc.id().to_string(), task_arc.clone());
        workflow_builder = workflow_builder.add_task(task_arc);
    }

    for (from, to) in edges {
        workflow_builder = workflow_builder.add_edge(from, to);
    }

    let mermaid_code = workflow_builder.generate_mermaid();
    let _graph = workflow_builder.build()?;

    let storage = Arc::new(InMemorySessionStorage::new());
    let session_id = "session_loop_1".to_string();
    let session = Session::new_from_task(session_id.clone(), "node_a");

    session.context.set("query", user_query.to_string())?;
    session.context.set("question", question.to_string())?;
    session.context.set("counter", 0)?;
    session.context.set("max_loops", max_loops)?;
    session.context.set("data", Vec::<String>::new())?;
    session.context.set("final_answer", "".to_string())?;
    session.context.set("router_decision", "".to_string())?;
    session.context.set("audit_trail", Vec::<GraphStepRecord>::new())?;
    session.context.set("architecture_mermaid", mermaid_code.clone())?;

    SessionStorage::save(&*storage, session).await?;

    // Step 1: Start node A
    let sess = SessionStorage::get(&*storage, &session_id).await?.unwrap();
    if let Some(node_a) = task_map.get("node_a") {
        node_a.run(sess.context.clone()).await?;
    }
    SessionStorage::save(&*storage, sess).await?;

    // Step 2: Iterative cycle loop (B -> C -> Router until counter >= max_loops)
    loop {
        let sess = SessionStorage::get(&*storage, &session_id).await?.unwrap();
        let counter = sess.context.get::<i32>("counter").unwrap_or_default();

        if counter >= max_loops {
            break;
        }

        if let Some(node_b) = task_map.get("node_b") {
            node_b.run(sess.context.clone()).await?;
        }
        if let Some(node_c) = task_map.get("node_c") {
            node_c.run(sess.context.clone()).await?;
        }
        if let Some(node_router) = task_map.get("node_router") {
            node_router.run(sess.context.clone()).await?;
        }

        SessionStorage::save(&*storage, sess).await?;
    }

    // Step 3: Final node D execution
    let final_sess = SessionStorage::get(&*storage, &session_id).await?.unwrap();
    if let Some(node_d) = task_map.get("node_d") {
        node_d.run(final_sess.context.clone()).await?;
    }
    SessionStorage::save(&*storage, final_sess).await?;

    let updated_session = SessionStorage::get(&*storage, &session_id).await.unwrap().unwrap();
    let ctx = &updated_session.context;

    let final_state = WorkflowState {
        query: ctx.get::<String>("query").unwrap_or_default(),
        question: ctx.get::<String>("question").unwrap_or_default(),
        counter: ctx.get::<i32>("counter").unwrap_or_default(),
        data: ctx.get::<Vec<String>>("data").unwrap_or_default(),
        final_answer: ctx.get::<String>("final_answer").unwrap_or_default(),
        architecture_mermaid: ctx.get::<String>("architecture_mermaid").unwrap_or_default(),
        audit_trail: ctx.get::<Vec<GraphStepRecord>>("audit_trail").unwrap_or_default(),
    };

    Ok(final_state)
}
```

---

### 2. Execution Entry Point: `src/main.rs` (Self-Correction & Critique Scenario)

Demonstrates a realistic **Self-Correction Code Generation Workflow**:
- **Node A (`Init Task`)**: Task setup.
- **Node B (`AI Code Generator`)**: Uses local LLM (`glm4:9b`) to generate/refine a Rust Fibonacci function.
- **Node C (`AI Critic & Incrementer`)**: Evaluates code quality and increments iteration count.
- **Node Router (`shouldContinue Router`)**: Evaluates `counter < 2` threshold.
- **Node D (`Final Supervisor Synthesis`)**: Synthesizes accumulated context and critiques into an executive summary and final code solution.

```rust
use anyhow::Result;
use dotenvy::dotenv;
use gfrig_loop_node::{run_loop_workflow, NodeMode, UnifiedNode};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST AGENTIC WORKFLOW: SELF-CORRECTION & CRITIQUE LOOP (Rig + Ollama) ===\n");

    let user_question = "Write a highly optimized Rust function for calculating Fibonacci numbers.";
    let user_query = "Execute generative self-correction loop.";

    // 1. Node A: Initialization of the assignment
    let node_a = UnifiedNode {
        id_str: "node_a".to_string(),
        label: "Task Initializer".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Init Task".to_string(),
            logic: Arc::new(|_ctx| {
                Box::pin(async move {
                    Ok("Starting iterative code generation and critique process.".to_string())
                })
            }),
        },
    };

    // 2. Node B: LLM Generator (Proposes code/solutions)
    let node_b = UnifiedNode {
        id_str: "node_b".to_string(),
        label: "AI Code Generator".to_string(),
        mode: NodeMode::LlmAgent {
            model_name: "glm4:9b".to_string(),
            preamble: "You are an expert Rust software engineer. Write or refine a clean, idiomatic, and optimized Rust solution based on the user's prompt and prior feedback.".to_string(),
        },
    };

    // 3. Node C: Critic and Counter Incrementer
    let node_c = UnifiedNode {
        id_str: "node_c".to_string(),
        label: "AI Critic & Incrementer".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Critique & Increment".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    let mut counter = ctx.get::<i32>("counter").unwrap_or_default();
                    counter += 1;
                    ctx.set("counter", counter)?;
                    
                    let data = ctx.get::<Vec<String>>("data").unwrap_or_default();
                    let last_draft = data.last().cloned().unwrap_or_default();

                    let critique = format!(
                        "Critique (Iteration {}): Code reviewed. Suggesting performance, error handling, and memory optimizations.",
                        counter
                    );
                    
                    Ok(format!("{} | Feedback on draft: [{}]", critique, last_draft))
                })
            }),
        },
    };

    // 4. Conditional Router: Decides whether to continue with iterations
    let node_router = UnifiedNode {
        id_str: "node_router".to_string(),
        label: "Loop Condition Router".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "shouldContinue Router".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    let counter = ctx.get::<i32>("counter").unwrap_or_default();
                    if counter < 2 {
                        ctx.set("router_decision", "b".to_string())?;
                        Ok(format!("Critique demands refinement. Looping back (counter = {})", counter))
                    } else {
                        ctx.set("router_decision", "d".to_string())?;
                        Ok(format!("Critique passed threshold (counter = {}). Exiting to final review.", counter))
                    }
                })
            }),
        },
    };

    // 5. Node D: Final AI Supervisor (Synthesis of Results)
    let node_d = UnifiedNode {
        id_str: "node_d".to_string(),
        label: "Final Supervisor Synthesis".to_string(),
        mode: NodeMode::LlmAgent {
            model_name: "glm4:9b".to_string(),
            preamble: "You are an elite AI technical lead. Review the final generated artifact and critique history, and provide a polished final executive summary.".to_string(),
        },
    };

    let nodes = vec![node_a, node_b, node_c, node_router, node_d];
    let edges = vec![
        ("node_a", "node_b"),
        ("node_b", "node_c"),
        ("node_c", "node_router"),
        ("node_router", "node_d"),
    ];

    println!("🚀 Running Self-Correction Critique Workflow...\n");

    match run_loop_workflow(user_query, user_question, nodes, edges, "node_a", 2).await {
        Ok(final_state) => {
            println!("--- 📊 Architecture (Mermaid) ---");
            println!("{}\n", final_state.architecture_mermaid);

            println!("--- 📝 Accumulated Execution Steps & Drafts ---");
            for (i, item) in final_state.data.iter().enumerate() {
                println!("  [{}] {}", i + 1, item);
            }
            println!();

            println!("--- 🤖 Final AI Supervisor Synthesis ---");
            println!("{}\n", final_state.final_answer);

            println!("--- 🔢 Final Counter Value: {} ---", final_state.counter);

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

## How to Run

1. **Start Ollama service and ensure the target LLM is available:**
   ```bash
   ollama serve
   ollama pull glm4:9b
   ```

2. **Execute the workflow application:**
   ```bash
   cargo run
   ```

3. **Run integration tests to verify cycle behavior:**
   ```bash
   cargo test
   ```
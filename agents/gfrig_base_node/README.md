# Stateful Graph Workflows in Rust (Rig & Graph-Flow v0.8)

This project ports stateful, cyclic workflow concepts from **LangGraph** (Python) into a high-performance, type-safe Rust architecture using the **`graph-flow` (v0.8.x)** crate and the **`rig`** AI framework, integrated with local models via **Ollama**.

It introduces **Dynamic Architecture Generation** (automatically building Mermaid.js diagrams from registered graph edges) and a robust **Explainable AI (XAI) Audit Trail** designed for compliance and export to graph databases like **Neo4j**.

---

## 🚀 Key Features

* **Stateful Graph Execution (`graph-flow` v0.8)**: Manages shared context securely across tasks using session storage and asynchronous execution runners (`FlowRunner`).
* **Rig AI Framework Integration**: Leverages `rig` to invoke local LLMs (e.g., `qwen3.5:4b` via Ollama) inside graph tasks.
* **Dynamic Architecture Visualization (`DynamicWorkflowBuilder`)**: Automatically builds a Mermaid.js diagram of the agent's topology from the declared graph edges, eliminating hardcoded diagrams.
* **XAI & Neo4j-Ready Audit Trail**: Every task execution records its `task_id`, `input_data`, and `output_data` into a shared audit history (`audit_trail`), providing complete transparency into the agent's decision-making process.

---

## 📁 Project Structure

```
gfrig_base_node/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                   # Core graph definitions, DynamicWorkflowBuilder, and Task implementations
│   └── main.rs                  # Executable runner triggering the workflow and printing XAI JSON
└── tests/
    └── base_node_tests.rs       # Automated integration test suite for graph tasks
```

---

## ⚙️ Dependencies (`Cargo.toml`)

Ensure your `Cargo.toml` includes the required versions for `graph-flow` v0.8.x and `rig`:

```toml
[package]
name = "gfrig_base_node"
version = "0.1.0"
edition = "2024"

[dependencies]
rig = "0.46.0"
graph-flow = "0.8"
tokio = { version = "1.38", features = ["full"] }
anyhow = "1.0"
dotenvy = "0.15"
async-trait = "0.1"
reqwest = { version = "0.13.5", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rand = "0.10.3"
uuid = { version = "1.0", features = ["v4", "serde"] }
futures = "0.3"

[dev-dependencies]
tempfile = "3.10"
```

---

## 💻 Core Implementation Overview

### 1. Library Logic (`src/lib.rs`)
The library defines the workflow state (`WorkflowState`), task components (`InputTask` and `LlmAgentTask`), and the wrapper `DynamicWorkflowBuilder` that captures edge relationships to generate Mermaid charts programmatically.

```rust
use anyhow::Result;
use async_trait::async_trait;
use graph_flow::{
    Context, FlowRunner, GraphBuilder, GraphError, InMemorySessionStorage,
    NextAction, Session, SessionStorage, Task, TaskResult,
};
use rig::providers::ollama;
use serde::{Deserialize, Serialize};
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
```

---

## 🏃 Running the Application

1. **Ensure Ollama is running locally** with your preferred model pulled:
   ```bash
   ollama pull qwen3.5:4b
   ```

2. **Execute the workflow runner**:
   ```bash
   cargo run
   ```

3. **Run unit & integration tests**:
   ```bash
   cargo test
   ```

---

## 📊 Expected Output

Upon successful execution, the application prints the automatically generated architecture, the state data, the LLM response, and the full JSON audit trail:

```json
{
  "query": "Explain why stateful graphs are useful for AI agents.",
  "data": [
    "Processed Input for label [Node_A]"
  ],
  "final_answer": "Stateful graphs allow AI agents to retain memory...",
  "architecture_mermaid": "graph TD\n    START --> node_a\n    node_a --> node_b\n    node_b --> END\n",
  "audit_trail": [
    {
      "task_id": "node_a",
      "input_data": "Explain why stateful graphs are useful for AI agents.",
      "output_data": "Processed Input for label [Node_A]"
    },
    {
      "task_id": "node_b",
      "input_data": "Context Data: ...",
      "output_data": "Stateful graphs allow AI agents..."
    }
  ]
}
```
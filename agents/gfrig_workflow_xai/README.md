# 🤖 Rust Workflow Graph with Explainable AI (XAI) & RIG Orchestration

A modular, object-oriented workflow graph orchestrator built in **Rust**, mirroring LangGraph’s node-based development flow while leveraging **`rig-core`** for local LLM integration (Ollama) and **`rusqlite`** for reliable, persistent state tracking and XAI audit logs.

---

## 🏗️ Architectural Overview

The framework replicates graph-flow execution patterns safely and performantly in Rust:

* **`WorkflowGraph` & `WorkflowNode`**: A trait-based object-oriented architecture allowing sequential registration and execution of custom asynchronous nodes via `.add_node()`.
* **`AgentState`**: A strongly typed shared state container that flows through the execution graph, keeping track of tasks, queries, histories, and audit trails.
* **Explainable AI (XAI)**: Captures step-by-step reasoning (`XaiStep`), inputs, and generated outputs for full transparency.
* **SQLite Persistence**: Automatically persists workflow execution logs and XAI trails into an embedded database (`storage/workflow_xai.db`).

---

## 📂 Project Structure

```text
gfrig_workflow_xai/
├── Cargo.toml
├── src/
│   ├── lib.rs                 # Core engine: WorkflowGraph, WorkflowNode, AgentState, SQLite manager
│   └── main.rs                # Main execution flow, Rig/Ollama integration, and XAI reporting
└── tests/
    └── workflow_xai_tests.rs   # Comprehensive integration and unit tests
```

---

## 🔍 Code Review & Technical Deep Dive

### 1. Lifetime Management & Higher-Ranked Trait Bounds (HRTB)

Asynchronous closures modifying a shared mutable state (`&mut AgentState`) in Rust often suffer from complex lifetime constraints. This implementation uses Higher-Ranked Lifetime Bounds (`for<'a>`) to elegantly satisfy `Send + Sync` requirements across asynchronous boundaries:

```rust
pub trait WorkflowNode: Send + Sync {
    fn execute<'a>(&'a self, state: &'a mut AgentState) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}
```

* **Benefit**: Allows closures to accept a borrow of `AgentState` with any valid execution lifetime, preventing premature drop or lifetime mismatch errors during sequential graph invocation.

### 2. Builder Pattern & Node Registration

The graph orchestrator exposes a fluent API reminiscent of Python's LangGraph:

```rust
let mut graph = WorkflowGraph::new();
graph.add_node("prepare_query", move |state| {
    let agent = agent_clone.clone();
    Box::pin(async move {
        // Node execution logic...
        Ok(())
    })
});
```

* **Benefit**: Zero-cost abstraction, type-safe closures, and absolute compile-time safety.

### 3. Explainable AI (XAI) Auditability

Every processing node emits structured audit metrics:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiStep {
    pub node_name: String,
    pub input_data: String,
    pub generated_output: String,
    pub reasoning: String,
}
```

* **Benefit**: Full inspection capabilities allowing developers and compliance systems to trace *why* and *how* a specific model output or state transition was produced.

---

## 📦 Dependencies (`Cargo.toml`)

```toml
[package]
name = "gfrig_workflow_xai"
version = "0.1.0"
edition = "2021"

[dependencies]
rig-core = "0.4"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rusqlite = { version = "0.31", features = ["bundled"] }
anyhow = "1.0"
```

---

## 🚀 Getting Started & Running Tests

1. **Ensure Ollama is running locally** with your desired model (e.g., `glm4:9b` or `mistral:7b`):
   ```bash
   ollama serve
   ollama pull glm4:9b
   ```

2. **Run the main execution graph**:
   ```bash
   cargo run
   ```

3. **Execute the test suite**:
   ```bash
   cargo test
   ```
# Rust Agentic Engine: Parallel Fork-Join & LLM Synthesis

A high-performance, production-ready agentic workflow engine written in **Rust**. It combines structured graph state management (`graph-flow`), native asynchronous parallelism (`tokio::join!`), and local LLM semantic synthesis (`rig` with Ollama).

---

## Overview & Technical Rationale

### Why We Bypassed Graph-Flow's Native Fork-Join Using Tokio

While Python-based frameworks like LangGraph natively handle multi-branch parallel routing (Fork-Join topologies) via complex graph runners, lightweight Rust graph engines like `graph-flow` are typically optimized around sequential or single-path execution runners (`FlowRunner`).

When attempting true multi-branch graph routing ($A \rightarrow B$, $A \rightarrow C$, $B \rightarrow D$, $C \rightarrow D$), graph runners often encounter limitations:

1. **Runner State Isolation & Race Conditions:** Separate threads executing nodes concurrently can overwrite shared session states or fail to synchronize execution windows correctly, leading to missing node states (`C missing`).
2. **Deterministic Execution:** By executing parallel tasks (`tokio::join!`) inside a single robust node, we leverage Tokio's enterprise-grade work-stealing scheduler directly.
3. **Guaranteed Synchronization:** `tokio::join!` forces the workflow to wait until all parallel data-fetching operations (e.g., Cloud API + Local Metrics) finish completely before passing the accumulated context to the final LLM synthesis node. This guarantees zero race conditions and robust performance without fighting the underlying graph runner's routing constraints.

---

## Features

- **Native Asynchronous Parallelism:** Utilizes Tokio's work-stealing scheduler for concurrent task execution (`tokio::join!`).
- **Hybrid Unified Nodes:** Seamlessly switch between deterministic tools (`CustomTool`), semantic routers (`LlmRouter`), and intelligent synthesis agents (`LlmAgent`).
- **Explainable AI (XAI) Audit Trail:** Fully serializes workflow execution steps, inputs, outputs, and topology structures into JSON format for monitoring, dashboards, or graph databases like Neo4j.
- **Mermaid Architecture Generation:** Automatically builds Mermaid graph representations of the active execution flow.

---

## Project Structure

```text
gfrig_parallel_node/
├── Cargo.toml
├── src/
│   ├── lib.rs       # Core engine, UnifiedNode, workflow runner, and state definitions
│   └── main.rs      # Telemetry workflow pipeline with tokio::join! & LLM synthesis
└── tests/
    └── workflow_tests.rs # Integration test suite for workflow verification
```

---

## Getting Started

### Prerequisites

1. **Rust Toolchain:** Latest stable edition with Cargo.
2. **Ollama:** Installed and running locally with your target model:
   ```bash
   ollama serve
   ollama pull glm4:9b
   ```

### Environment Setup

Create a `.env` file in the root directory if necessary or configure environment variables for local LLM endpoints.

### Running the Project

Execute the telemetry workflow pipeline:
```bash
cargo run
```

### Running Tests

Run the integration test suite:
```bash
cargo test
```
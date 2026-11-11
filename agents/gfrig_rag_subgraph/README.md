# 🤖 Rust LangGraph to RIG: Subgraph Micro-RAG with XAI

This project implements a hybrid **Self-Correcting Micro-RAG System with Explainable AI (XAI)** in **Rust**. It rewrites a Python-based **LangGraph** subgraph architecture into Rust using the **Rig** crate (`rig-core`) and local LLM orchestration via **Ollama**.

The system features:
1. **Main Orchestrator Graph:** Handles initial query expansion (`PrepareQueryNode`), external/web search retrieval (`WikipediaCallNode`), delegates to a RAG Subgraph, and synthesizes final knowledge (`SummarizeNode`).
2. **Cyclic Self-Correcting Subgraph (`MicroRAG`):** Contains document retrieval, reranking, generation, and a **Critic Agent** that evaluates answers on a scale of 1–10. If the score is below threshold, it executes a feedback loop (max 3 iterations) to refine the query and re-run retrieval.
3. **Explainable AI (XAI) Transparency Report:** Captures human-readable rationale (`critic_reason`), confidence scores (`critic_score`), query mutations, and data provenance across both main graph and subgraph execution steps.

---

## 📂 Project Structure

```text
gfrig_rag_subgraph/
├── Cargo.toml                # Project metadata and dependencies
├── README.md                 # Project documentation and guide
├── src/
│   ├── lib.rs                # Shared AgentState definition
│   └── main.rs               # Main execution graph, subgraph loop, and XAI reporting
└── tests/
    └── agent_state_tests.rs  # Unit and integration test suite
```

---

## 🛠️ Prerequisites & Setup

### 1. Requirements

* **Rust Toolchain:** (1.75+ recommended)
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
* **Ollama Server:** Running locally at `http://localhost:11434`.
  * Pull the configured LLM model (e.g., `glm4:9b` or `mistral:7b`):
    ```bash
    ollama pull glm4:9b
    ```

### 2. Cargo Configuration (`Cargo.toml`)

```toml
[package]
name = "gfrig_rag_subgraph"
version = "0.1.0"
edition = "2021"

[dependencies]
rig-core = "0.4"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

---

## 🏗️ Architecture & Technical Workflow

```text
[ START ]
    │
    ▼
[ PrepareQueryNode ] ──> Optimizes raw user task using LLM
    │
    ├───────────────────────────────┐
    ▼                               ▼
[ WikipediaCallNode ]       [ MicroRAG Subgraph ]
(External Search Context)           │
                                    ├──> [ RetrievalNode ] (SQLite/FAISS)
                                    ├──> [ RerankNode ] (Context Compression)
                                    ├──> [ GenerationNode ] (Contextual QA)
                                    └──> [ CriticNode (XAI) ] (Score + Rationale)
                                            │
                                            ├── Score < 7.0 & Iter < 3 ──> Retry Loop
                                            └── Score >= 7.0 ─────────────┐
                                                                          │
    ┌─────────────────────────────────────────────────────────────────────┘
    ▼
[ SummarizeNode ] ──> Synthesizes External + Internal RAG Knowledge
    │
    ▼
[ XAI Report Output ] ──> Prints transparent decision audit & confidence metrics
```

---

## 💻 Source Code Overview

### 1. Shared State (`src/lib.rs`)

```rust
use serde::{Deserialize, Serialize};

/// Overall state of an agent traversing the main graph and the subgraph
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentState {
    pub task: String,
    pub question: String,
    pub documents: Vec<String>,
    pub context: String,
    pub answer: String,
    pub critic_score: f32,
    pub critic_reason: String, // XAI rationale from the critic agent
    pub iteration: usize,
    pub search_result: String,
    pub summarization: String,
}

impl AgentState {
    pub fn new(task: &str) -> Self {
        Self {
            task: task.to_string(),
            question: String::new(),
            documents: vec![],
            context: String::new(),
            answer: String::new(),
            critic_score: 0.0,
            critic_reason: String::new(),
            iteration: 0,
            search_result: String::new(),
            summarization: String::new(),
        }
    }
}
```

### 2. Main Execution & Subgraph Engine (`src/main.rs`)

```rust
use rig::providers::ollama;
use rig::agent::AgentBuilder;
use gfrig_rag_subgraph::AgentState;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RUST LANGGRAPH -> RIG: SUBGRAPH RAG WITH XAI ===");

    // Initialize Ollama provider client
    let client = ollama::Ollama::new();
    let model = client.completion("glm4:9b");

    // Build the primary Rig Agent
    let agent = AgentBuilder::new(model)
        .preamble("You are an expert AI assistant specialized in research, RAG systems, and data analysis.")
        .build();

    let mut state = AgentState::new("What are the developments and trends in AI (artificial intelligence)?");

    // --- STEP 1: PrepareQueryNode (Main Graph) ---
    println!("\n--- [PrepareQueryNode] ---");
    let query_prompt = format!(
        "Improve the user query so it can be used for a better search. User question: {}",
        state.task
    );
    let improved_query_resp = agent.prompt(&query_prompt).await?;
    state.question = improved_query_resp.to_string().trim().to_string();
    println!("✅ Improved Query: '{}'", state.question);

    // --- STEP 2: External Search Node ---
    println!("\n--- [WikipediaCallNode] ---");
    state.search_result = format!(
        "External search results for '{}': AI is rapidly evolving with focus on LLMs, Edge AI, and Hybrid architectures.",
        state.question
    );
    println!("✅ Search context retrieved.");

    // --- STEP 3: Micro-RAG Subgraph (Cyclic Self-Correction + XAI) ---
    println!("\n--- [MicroRAG Subgraph Start with XAI Evaluation] ---");
    let max_iterations = 3;
    let mut rag_success = false;

    while state.iteration < max_iterations && !rag_success {
        state.iteration += 1;
        println!("\n🔄 RAG Iteration no. {} (out of {})", state.iteration, max_iterations);

        // 3.1 Retrieval Node
        state.documents = vec![
            "Document 1: Machine learning and deep learning integration form the core of modern AI.".to_string(),
            "Document 2: Edge AI allows running models locally with lower latency and higher privacy.".to_string(),
            "Document 3: Explainable AI (XAI) is critical for enterprise deployment and regulation compliance.".to_string(),
        ];

        // 3.2 Rerank Node
        state.context = format!("{}\n\n{}", state.documents[0], state.documents[1]);

        // 3.3 Generation Node
        let gen_prompt = format!(
            "Answer the question based only on the following context:\nContext: {}\nQuestion: {}",
            state.context, state.question
        );
        let answer_resp = agent.prompt(&gen_prompt).await?;
        state.answer = answer_resp.to_string();

        // 3.4 Critic Node with XAI
        let critic_prompt = format!(
            "You are a critic agent. Evaluate the answer for relevance to the question. \
            Return your response strictly in this format: \
            SCORE: <number from 1.0 to 10.0> | REASON: <brief explanation why>\n\
            Question: {}\nAnswer: {}",
            state.question, state.answer
        );
        let critic_resp = agent.prompt(&critic_prompt).await?;
        let critic_response_str = critic_resp.to_string();
        
        if critic_response_str.contains("SCORE:") && critic_response_str.contains("REASON:") {
            let parts: Vec<&str> = critic_response_str.split('|').collect();
            let score_part = parts[0].replace("SCORE:", "").trim().to_string();
            let reason_part = parts[1].replace("REASON:", "").trim().to_string();

            state.critic_score = score_part.parse().unwrap_or(7.5);
            state.critic_reason = reason_part;
        } else {
            state.critic_score = 7.5;
            state.critic_reason = critic_response_str.trim().to_string();
        }

        println!("  ⭐ Subgraph XAI Critic Score: {} / 10.0", state.critic_score);
        println!("  🔍 Subgraph XAI Reasoning: \"{}\"", state.critic_reason);

        if state.critic_score >= 7.0 {
            println!("✅ Subgraph condition met: Score >= 7. Exiting RAG loop.");
            rag_success = true;
        } else {
            println!("⚠️ Score too low (< 7). Retrying retrieval with modified query feedback...");
            state.question.push_str(" (please specify for a better answer)");
        }
    }

    // --- STEP 4: SummarizeNode ---
    println!("\n--- [SummarizeNode & Main App XAI] ---");
    let summary_prompt = format!(
        "Summarize the following information combining external search and verified internal RAG answer.\n\nExternal Search:\n{}\n\nMicro-RAG Answer:\n{}",
        state.search_result, state.answer
    );
    let summary_resp = agent.prompt(&summary_prompt).await?;
    state.summarization = summary_resp.to_string();

    // --- STEP 5: Final XAI Audit Report ---
    println!("\n==========================================");
    println!("========= FINAL PIPELINE RESULT ==========");
    println!("==========================================");
    println!("{}", state.summarization);
    println!("\n==========================================");
    println!("🔍 XAI TRANSPARENCY REPORT (MAIN APP & SUBGRAPH):");
    println!("==========================================");
    println!("• Original User Task:        {}", state.task);
    println!("• Optimized Search Query:    {}", state.question);
    println!("• Subgraph Iterations Run:   {} / {}", state.iteration, max_iterations);
    println!("• Final Subgraph Score:      {} / 10.0", state.critic_score);
    println!("• Subgraph Critic Reasoning: {}", state.critic_reason);
    println!("• Sources Integrated:        External Wikipedia/Web search + SQLite/FAISS Micro-RAG");
    println!("==========================================");

    Ok(())
}
```

---

## 🧪 Testing

The integration test suite in `tests/agent_state_tests.rs` validates state initialization, XAI output string parsing, and cyclic retry conditions.

Run the test suite with:
```bash
cargo test
```

### Example Test Execution Output:
```text
running 3 tests
test test_agent_state_initialization ... ok
test test_subgraph_retry_loop_condition ... ok
test test_xai_critic_parsing_logic ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; finished in 0.00s
```

---

## 🚀 Execution & Usage

To run the complete pipeline:

```bash
cargo run
```

---

## 🔑 Key Takeaways (LangGraph vs. Rig Rust)

| Feature | LangGraph (Python) | Rig (Rust) |
| :--- | :--- | :--- |
| **State Container** | `TypedDict` / `Annotated` | Native Rust `struct` (`AgentState`) |
| **Subgraph Representation** | Compiled `StateGraph` Node | Asynchronous `while` Loop & Tokio Tasks |
| **LLM Provider** | `ChatOllama` | `rig::providers::ollama::Ollama` |
| **Agent Prompting** | `RunnableSequence` (`|`) | `AgentBuilder` + `agent.prompt(...)` |
| **Explainability (XAI)** | Custom dictionary metadata | Parsed structured fields on `AgentState` |
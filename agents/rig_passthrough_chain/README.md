# RunnablePassthrough RAG Pattern in Rust (Rig Framework)

This project implements a type-safe, modular **RAG (Retrieval-Augmented Generation)** execution pattern in Rust using the **Rig** framework and local LLMs via **Ollama**.

It serves as an explicit, high-performance Rust replacement for LangChain's `RunnablePassthrough` pattern. It demonstrates how to preserve original input state (identity function/pass-through) while concurrently fetching or generating contextual data, ensuring zero loss of user queries across pipeline boundaries.

## 🚀 Key Features

* **Input State Preservation**: Implements the `RunnablePassthrough` paradigm in pure Rust, carrying the original user query intact alongside newly synthesized or retrieved context.
* **XAI & Audit-Log Ready**: Every stage (context generation, passthrough mapping, and final RAG synthesis) outputs structured step logs (`RagStepRecord`) ready for serialization into JSON or graph memory databases (such as Neo4j).
* **Decoupled Configuration**: System instructions, prompts, and context parameters are injected at runtime via config parameters without hardcoded system state.

## 📁 Project Structure

```
rig_passthrough_chain/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                   # Core RAG pipeline logic & step logger
│   └── main.rs                  # Runner demonstration with context passthrough
└── tests/
    └── passthrough_chain_tests.rs # Integration test suite for RAG passthrough
```

## ⚙️ Dependencies (`Cargo.toml`)

Ensure your `Cargo.toml` contains the required dependencies:

```toml
[package]
name = "rig_passthrough_chain"
version = "0.1.0"
edition = "2021"

[dependencies]
rig = "0.45.0"
tokio = { version = "1.38", features = ["full"] }
anyhow = "1.0"
dotenvy = "0.15"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

## 💻 Architecture & Code Overview

### 1. Core RAG Pipeline (`src/lib.rs`)

Executes context retrieval/synthesis, holds the original input state, and constructs the final contextualized response:

```rust
use anyhow::Result;
use rig::providers::ollama;
use serde::{Deserialize, Serialize};

/// Single-step record for XAI / Neo4j memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagStepRecord {
    pub step_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// Result of a RAG chain with context retention
#[derive(Debug, Serialize, Deserialize)]
pub struct RagChainResult {
    pub original_question: String,
    pub generated_context: String,
    pub final_output: String,
    pub history: Vec<RagStepRecord>,
}

/// Universal runner implementing the RunnablePassthrough pattern (RAG pattern)
pub async fn run_passthrough_rag_chain(
    model_name: &str,
    question: &str,
    context_system: &str,
    context_prompt_tmpl: &str,
    final_system: &str,
    final_prompt_tmpl: &str,
) -> Result<RagChainResult> {
    let client = ollama::Ollama::new();

    // --- Step 1: Context Branch (Retriever Simulation) ---
    let context_model = client.completion(model_name);
    let context_agent = rig::agent::AgentBuilder::new(context_model)
        .preamble(context_system)
        .build();

    let context_prompt = context_prompt_tmpl.replace("{question}", question);
    let context_response = context_agent.prompt(&context_prompt).await?;
    let generated_context = context_response.output().trim().to_string();

    let mut history = vec![RagStepRecord {
        step_name: "context_retriever_simulation".to_string(),
        input_value: question.to_string(),
        generated_output: generated_context.clone(),
    }];

    // --- Step 2: RunnablePassthrough (preserving the original input) ---
    let original_question = question.to_string();

    // --- Step 3: Final RAG generation combining the context and the original question ---
    let final_model = client.completion(model_name);
    let final_agent = rig::agent::AgentBuilder::new(final_model)
        .preamble(final_system)
        .build();

    let final_prompt = final_prompt_tmpl
        .replace("{context}", &generated_context)
        .replace("{question}", &original_question);

    let final_response = final_agent.prompt(&final_prompt).await?;
    let final_output = final_response.output().trim().to_string();

    history.push(RagStepRecord {
        step_name: "final_rag_generation".to_string(),
        input_value: format!("Context: {} | Question: {}", generated_context, original_question),
        generated_output: final_output.clone(),
    });

    Ok(RagChainResult {
        original_question,
        generated_context,
        final_output,
        history,
    })
}
```

### 2. Application Runner (`src/main.rs`)

```rust
use anyhow::Result;
use dotenvy::dotenv;
use rig_passthrough_chain::run_passthrough_rag_chain;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG RUNNABLE PASSTHROUGH (RAG PATTERN) ===\n");

    let model_name = "qwen3.5:4b";
    let question = "Where is the company that made the Tesla Model Y headquartered?";

    let context_system = "You are a factual knowledge assistant. Provide concise, direct facts.";
    let context_prompt = "For the given question ({question}), write ONE sentence of relevant FACT (e.g. CEO, year of foundation, headquarters).";

    let final_system = "You are a precise assistant answering questions based on the provided context.";
    let final_prompt = "Answer the question based on the context provided.\nCONTEXT: {context}\nQUESTION: {question}\nResponse:";

    println!("User Question: '{}'\n", question);

    match run_passthrough_rag_chain(
        model_name,
        question,
        context_system,
        context_prompt,
        final_system,
        final_prompt,
    )
    .await
    {
        Ok(res) => {
            println!("--- Generated Context (Retriever simulation) ---");
            println!("{}\n", res.generated_context);

            println!("--- Final RAG Output ---");
            println!("{}\n", res.final_output);

            let json_audit = serde_json::to_string_pretty(&res.history).unwrap();
            println!("--- XAI Audit Trail (JSON) ---");
            println!("{}", json_audit);
        }
        Err(e) => eprintln!("Error during execution: {}", e),
    }

    Ok(())
}
```

## 🧪 Testing

To run the integration tests:

```bash
cargo test
```

## 🏃 Building & Running

1. Ensure your local Ollama instance is active and the model is pulled:

   ```bash
   ollama pull qwen3.5:4b
   ```

2. Execute the runner:

   ```bash
   cargo run
   ```
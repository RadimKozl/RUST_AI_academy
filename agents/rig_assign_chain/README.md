# RunnableAssign Context Enrichment Pattern in Rust (Rig Framework)

This project translates LangChain's `RunnableAssign` (`RunnablePassthrough.assign(...)`) dictionary enrichment pattern into a high-performance, type-safe Rust architecture using the **Rig** framework and local LLMs via **Ollama**.

It demonstrates how to seamlessly enrich input payloads with dynamic metadata (e.g., query length, model identifiers, timestamps) prior to prompt construction while maintaining a comprehensive audit trail for XAI (Explainable AI) and graph database storage (e.g., Neo4j).

## 🚀 Key Features

* **Context Enrichment**: Replaces Python dictionary mutation (`RunnableAssign`) with strongly typed Rust context structs (`EnrichedContext`) that calculate dynamic metadata on the fly.
* **Metadata Injection**: Dynamically injects computed metadata (timestamps, text length, model parameters) into downstream LLM prompt templates.
* **XAI & Audit Trail Ready**: Captures raw execution steps and enriched context structures into serializable logs (`AssignStepRecord`) for debugging, traceability, and graph memory integration.
* **Decoupled Architecture**: Logic, prompts, system preambles, and execution pipelines are clearly separated into modular Rust components.

## 📁 Project Structure

```
rig_assign_chain/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                   # Core enrichment logic, runner, and step logger
│   └── main.rs                  # Application entry point executing the chain
└── tests/
    └── assign_chain_tests.rs    # Unit and integration test suite
```

## ⚙️ Dependencies (`Cargo.toml`)

Ensure your `Cargo.toml` includes all required dependencies:

```toml
[package]
name = "rig_assign_chain"
version = "0.1.0"
edition = "2021"

[dependencies]
rig = "0.46.0"
tokio = { version = "1.38", features = ["full"] }
anyhow = "1.0"
dotenvy = "0.15"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = "0.4"
```

## 💻 Architecture & Code Overview

### 1. Library Logic & Enrichment Engine (`src/lib.rs`)

```rust
use anyhow::Result;
use chrono::Local;
use rig::providers::ollama;
use serde::{Deserialize, Serialize};

/// Single-step record for XAI / Neo4j memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignStepRecord {
    pub step_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// Enriched context representing the output of the RunnableAssign step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichedContext {
    pub question: String,
    pub question_length: usize,
    pub processing_model: String,
    pub timestamp: String,
}

/// The overall result of the chain, containing the final answer as well as the audit history
#[derive(Debug, Serialize, Deserialize)]
pub struct AssignChainResult {
    pub context: EnrichedContext,
    pub final_output: String,
    pub history: Vec<AssignStepRecord>,
}

/// Function performing the RunnableAssign operation (enriching the input with metadata)
pub fn assign_metadata(question: &str, model_name: &str) -> EnrichedContext {
    EnrichedContext {
        question: question.to_string(),
        question_length: question.chars().count(),
        processing_model: model_name.to_string(),
        timestamp: Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
    }
}

/// Universal runner for a RunnableAssign RAG / Enrichment chain
pub async fn run_assign_chain(
    model_name: &str,
    question: &str,
    system_instruction: &str,
    prompt_template: &str,
) -> Result<AssignChainResult> {
    let client = ollama::Ollama::new();
    
    // We can set max_tokens or other parameters if the model is limited
    let model = client.completion(model_name);

    // Step 1: RunnableAssign (Enriching the context with metadata)
    let enriched_ctx = assign_metadata(question, model_name);

    let mut history = vec![AssignStepRecord {
        step_name: "runnable_assign_enrichment".to_string(),
        input_value: question.to_string(),
        generated_output: serde_json::to_string(&enriched_ctx)?,
    }];

    // Step 2: Preparing the agent
    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .build();

    let final_prompt = prompt_template
        .replace("{timestamp}", &enriched_ctx.timestamp)
        .replace("{processing_model}", &enriched_ctx.processing_model)
        .replace("{question_length}", &enriched_ctx.question_length.to_string())
        .replace("{question}", &enriched_ctx.question);

    // Step 3: Generating the final response
    // If the agent in Rig supports max_tokens via the builder, it can be specified there,
    // It might also help if the prompt doesn't require an overly long text.
    let response = agent.prompt(&final_prompt).await?;
    let final_output = response.output().trim().to_string();

    history.push(AssignStepRecord {
        step_name: "final_llm_execution".to_string(),
        input_value: final_prompt,
        generated_output: final_output.clone(),
    });

    Ok(AssignChainResult {
        context: enriched_ctx,
        final_output,
        history,
    })
}
```

### 2. Application Runner (`src/main.rs`)

```rust
use anyhow::Result;
use dotenvy::dotenv;
use rig_assign_chain::run_assign_chain;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG RUNNABLE ASSIGN (DICTIONARY ENRICHMENT) ===\n");

    let model_name = "qwen3.5:4b";
    let question = "Write me a short haiku about programming in Python.";

    let system_instruction = "You are a creative poetic assistant.";
    let prompt_template = "Today's date: {timestamp} (processed by model: {processing_model}).\n\
                           The question is {question_length} characters long. Answer this question:\n\
                           QUESTION: {question}\n\
                           RESPONSE:";

    println!("🚀 Running Chain with RunnableAssign to add metadata...\n");

    match run_assign_chain(model_name, question, system_instruction, prompt_template).await {
        Ok(res) => {
            println!("--- Enriched Context (RunnableAssign Result) ---");
            println!("{:#?}\n", res.context);

            println!("--- Final Output ---");
            println!("{}\n", res.final_output);

            // XAI / Neo4j Audit Trail
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

Execute both the unit metadata tests and integration tests using Cargo:

```bash
cargo test
```

## 🏃 Execution Steps

1. Start your local Ollama instance and ensure the required model is pulled:
   ```bash
   ollama pull qwen3.5:4b
   ```

2. Build and run the project:
   ```bash
   cargo run
   ```
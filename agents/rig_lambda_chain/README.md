# RunnableLambda Pattern in Rust (Rig Framework)

This project provides a type-safe, modular implementation of LangChain's `RunnableLambda` pattern in Rust using the **Rig** framework and local LLMs via **Ollama**.

It demonstrates how to encapsulate arbitrary custom transformation functions (closures or named Rust functions) within an LLM chain to process raw outputs, extract structural metadata, validate payloads, or enforce custom business logic without compromising performance or auditability.

## 🚀 Key Features

* **Custom Function Encapsulation**: Replaces Python's `RunnableLambda` with native Rust functions and closures for type-safe data transformations.
* **Structured Output Generation**: Converts unstructured LLM string outputs into fully typed Rust structs containing summaries, length metadata, and provenance tracking.
* **XAI & Audit-Log Ready**: Captures raw execution steps and transformed outputs into structured logs (`LambdaStepRecord`) suitable for serialization to JSON or persistence in graph memory databases (such as Neo4j).
* **Decoupled Architecture**: Prompts, system instructions, and transformation pipelines are cleanly decoupled from low-level execution logic.

## 📁 Project Structure

```
rig_lambda_chain/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                   # Core transformation pipeline & step logger
│   └── main.rs                  # Runner application executing the lambda chain
└── tests/
    └── lambda_chain_tests.rs    # Unit and integration test suite
```

## ⚙️ Dependencies (`Cargo.toml`)

Ensure your `Cargo.toml` contains the required dependencies:

```toml
[package]
name = "rig_lambda_chain"
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

### 1. Library Logic & Custom Transformation (`src/lib.rs`)

```rust
use anyhow::Result;
use rig::providers::ollama;
use serde::{Deserialize, Serialize};

/// Single-step record for XAI / Neo4j memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaStepRecord {
    pub step_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// Structured result after transformation using RunnableLambda
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedSummaryResult {
    pub summary: String,
    pub full_length: usize,
    pub source: String,
}

/// Overall result of the chain, containing data and audit history
#[derive(Debug, Serialize, Deserialize)]
pub struct LambdaChainResult {
    pub final_data: ProcessedSummaryResult,
    pub history: Vec<LambdaStepRecord>,
}

/// Custom function (equivalent to the processor's RunnableLambda)
pub fn process_and_summarize(full_text: &str) -> ProcessedSummaryResult {
    let summary = if full_text.chars().count() > 20 {
        let truncated: String = full_text.chars().take(20).collect();
        format!("{}...", truncated)
    } else {
        full_text.to_string()
    };

    ProcessedSummaryResult {
        summary,
        full_length: full_text.chars().count(),
        source: "AI_Model_Ollama_Qwen".to_string(),
    }
}

/// A universal runner that simulates the behavior of a RunnableLambda within a chain
pub async fn run_lambda_chain(
    model_name: &str,
    tone: &str,
    system_instruction: &str,
    prompt_template: &str,
) -> Result<LambdaChainResult> {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);

    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .build();

    let prompt = prompt_template.replace("{tone}", tone);
    
    // Step 1: Launching the LLM agent
    let response = agent.prompt(&prompt).await?;
    let raw_text = response.output().trim().to_string();

    let mut history = vec![LambdaStepRecord {
        step_name: "llm_generation".to_string(),
        input_value: format!("Tone: {}", tone),
        generated_output: raw_text.clone(),
    }];

    // Step 2: Transformation using a RunnableLambda equivalent
    let processed_data = process_and_summarize(&raw_text);

    history.push(LambdaStepRecord {
        step_name: "runnable_lambda_transform".to_string(),
        input_value: raw_text.clone(),
        generated_output: format!("{:#?}", processed_data),
    });

    Ok(LambdaChainResult {
        final_data: processed_data,
        history,
    })
}
```

### 2. Application Runner (`src/main.rs`)

```rust
use anyhow::Result;
use dotenvy::dotenv;
use rig_lambda_chain::run_lambda_chain;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG RUNNABLE LAMBDA CHAIN (CUSTOM TRANSFORMATION) ===\n");

    let model_name = "qwen3.5:4b";
    let tone = "Use a friendly and simple tone";
    let system_instruction = "You are a clear and concise science educator.";
    let prompt_template = "Explain what an atom is in one paragraph. {tone}.";

    println!("🚀 Running Chain with RunnableLambda simulation...\n");

    match run_lambda_chain(model_name, tone, system_instruction, prompt_template).await {
        Ok(res) => {
            println!("-------------------------------------------");
            println!("Type of final result: Struct (ProcessedSummaryResult)");
            println!("FINAL OUTPUT:\n{:#?}\n", res.final_data);

            // XAI / Neo4j audit trail
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

Run both unit tests and integration tests with Cargo:

```bash
cargo test
```

## 🏃 Building & Running

1. Ensure Ollama is installed and running locally with the target model:

   ```bash
   ollama pull qwen3.5:4b
   ```

2. Run the application:

   ```bash
   cargo run
   ```
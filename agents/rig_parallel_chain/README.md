# Parallel Chain in Rust (LangChain `RunnableParallel` Equivalent using Rig)

This project implements an asynchronous, concurrent **Parallel Chain** execution engine using the **Rig** framework and local LLMs via **Ollama**.

It serves as a direct Rust equivalent of LangChain's `RunnableParallel`. It splits execution flow by fanning out a single input topic across multiple independent agents/prompts simultaneously, collecting all generated responses into a structured output map.

---

## 🚀 Key Features

* **True Asynchronous Concurrency**: Utilizes `tokio::spawn` and concurrent futures execution so that total execution latency is bound only by the slowest branch, not the sum of all branches.
* **Type-Safe Dynamic Branching**: Define any number of custom branches (`Vec<ParallelBranch>`) with specific system instructions and prompt templates.
* **LangChain `RunnableParallel` Parity**: Maps outputs directly to named key-value entries (`HashMap<String, String>`).
* **XAI & Audit-Log Ready**: Structured key-value results can easily be serialized into JSON for explainability logging, monitoring, or graph database ingestion (e.g., Neo4j).
* **Decoupled Architecture**: Prompts and system behavior are configurable from the runner layer (`main.rs`) without changing library logic.

---

## 📁 Project Structure

```text
rig_parallel_chain/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                   # Core parallel execution engine & branch models
│   └── main.rs                  # Application entry point demonstrating concurrent execution
└── tests/
    └── parallel_chain_tests.rs  # Integration test suite
```

---

## ⚙️ Dependencies (`Cargo.toml`)

Ensure your `Cargo.toml` contains the required crates:

```toml
[package]
name = "rig_parallel_chain"
version = "0.1.0"
edition = "2021"

[dependencies]
rig = "0.45.0"
tokio = { version = "1.38", features = ["full"] }
anyhow = "1.0"
dotenvy = "0.15"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
futures = "0.3"
```

---

## 💻 Architecture & Code Overview

### 1. Library Engine (`src/lib.rs`)

`src/lib.rs` provides data models for branch definitions and the `run_parallel_chain` function:

```rust
use anyhow::Result;
use rig::providers::ollama;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Record of a single branch result for auditing/XAI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelBranchResult {
    pub branch_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// Definition of a single parallel branch
#[derive(Clone)]
pub struct ParallelBranch {
    pub name: String,
    pub system_instruction: String,
    pub prompt_template: String, // Contains template placeholder "{topic}"
}

/// Universal parallel runner (LangChain RunnableParallel equivalent)
pub async fn run_parallel_chain(
    model_name: &str,
    input_topic: &str,
    branches: &[ParallelBranch],
) -> Result<HashMap<String, String>> {
    let mut futures = Vec::new();

    for branch in branches {
        let model_name_owned = model_name.to_string();
        let topic_owned = input_topic.to_string();
        let branch_name = branch.name.clone();
        let system_instr = branch.system_instruction.clone();
        let prompt_tmpl = branch.prompt_template.clone();

        let fut = tokio::spawn(async move {
            let model = ollama::Ollama::new().completion(&model_name_owned);
            let agent = rig::agent::AgentBuilder::new(model)
                .preamble(&system_instr)
                .build();

            let prompt = prompt_tmpl.replace("{topic}", &topic_owned);
            let response = agent.prompt(&prompt).await?;
            Ok::<(String, String), anyhow::Error>((branch_name, response.output().trim().to_string()))
        });

        futures.push(fut);
    }

    let results = futures::future::join_all(futures).await;

    let mut output_map = HashMap::new();
    for res in results {
        match res {
            Ok(Ok((name, output))) => {
                output_map.insert(name, output);
            }
            Ok(Err(e)) => return Err(e),
            Err(e) => return Err(anyhow::anyhow!("Task join error: {}", e)),
        }
    }

    Ok(output_map)
}
```

### 2. Application Runner (`src/main.rs`)

This entry point configures two parallel branches (`joke` and `poem`) for the topic `"ice cream"`:

```rust
use anyhow::Result;
use dotenvy::dotenv;
use rig_parallel_chain::{run_parallel_chain, ParallelBranch};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG PARALLEL CHAIN (RUNNABLE PARALLEL EQUIVALENT) ===\n");

    let model_name = "qwen3.5:4b";
    let topic = "ice cream";

    let branches = vec![
        ParallelBranch {
            name: "joke".to_string(),
            system_instruction: "You are a funny comedian. Tell short, witty jokes.".to_string(),
            prompt_template: "tell me a short joke about {topic}".to_string(),
        },
        ParallelBranch {
            name: "poem".to_string(),
            system_instruction: "You are a creative and lyrical poet.".to_string(),
            prompt_template: "write a 2-line poem about {topic}".to_string(),
        },
    ];

    println!("Launching parallel chain for topic: '{}'\n", topic);

    match run_parallel_chain(model_name, topic, &branches).await {
        Ok(results) => {
            if let Some(joke) = results.get("joke") {
                println!("---- Joke ----\n{}\n", joke);
            }
            if let Some(poem) = results.get("poem") {
                println!("---- Poem ----\n{}\n", poem);
            }

            let json_output = serde_json::to_string_pretty(&results).unwrap();
            println!("--- JSON Map Output ---");
            println!("{}", json_output);
        }
        Err(e) => eprintln!("Error during parallel execution: {}", e),
    }

    Ok(())
}
```

---

## 🧪 Testing

Run the integration test suite (`tests/parallel_chain_tests.rs`) to verify concurrent execution and map generation:

```bash
cargo test
```

---

## 🏃 Building & Running

1. Ensure Ollama is running locally with your desired model:
   ```bash
   ollama pull qwen3.5:4b
   ```

2. Build and run the project:
   ```bash
   cargo build
   cargo run
   ```
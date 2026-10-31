# Modular N-Step Sequential Chain in Rust (Rig Framework)

This project implements a fully **decoupled, N-step modular sequential chain** (Chain of Thought pipeline) using the **Rig** framework and local LLMs via **Ollama**.

Unlike rigid, hardcoded chains, this architecture allows you to define an arbitrary number of steps ($N$ steps) dynamically in code or configuration. It tracks every intermediate step in a strongly typed, serializable audit log—making it directly usable for **Explainable AI (XAI)** audit trails, observability logging, or direct graph insertion into graph databases like **Neo4j**.

---

## 🚀 Key Features

* **Dynamic $N$-Step Pipelines**: Define any sequence of transformation steps (`Vec<ChainStep>`) without modifying library code or re-compiling the pipeline runner.
* **Full Audit Trail for XAI & Neo4j**: Returns both the final result and a step-by-step history array (`Vec<ChainStepRecord>`) containing input, step name, and generated output.
* **Serde JSON Ready**: The history structure derives `Serialize` and `Deserialize`, making export to JSON, NoSQL, or graph stores trivial.
* **Decoupled Architecture**: Prompts, system instructions, and step definitions are supplied from the caller level (`main.rs` or API layer), maintaining a clean separation of concerns.
* **Compile-Time Safety**: Strong Rust types replace loose Python dictionary passing (`{"city": chain1}`).

---

## 📁 Project Structure

```text
rig_sequential_chain/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                 # Core generic N-step execution engine & data types
│   └── main.rs                # Example 4-step execution runner
└── tests/
    └── modular_chain_tests.rs # Integration test suite verifying context passing & history
```

---

## ⚙️ Dependencies (`Cargo.toml`)

Add the following crates to your `Cargo.toml`:

```toml
[package]
name = "rig_sequential_chain"
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

---

## 💻 Architecture & Code Overview

### 1. Data Structures & Engine (`src/lib.rs`)

`src/lib.rs` provides generic data models and the `run_modular_chain` runner:

```rust
use anyhow::Result;
use rig::providers::ollama;
use serde::{Deserialize, Serialize};

/// Record of a single executed step (useful for XAI / Neo4j memory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStepRecord {
    pub step_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// Result of the modular chain containing final answer and execution history
#[derive(Debug, Serialize, Deserialize)]
pub struct ModularChainResult {
    pub final_output: String,
    pub history: Vec<ChainStepRecord>,
}

/// Definition of a single configurable pipeline step
#[derive(Clone)]
pub struct ChainStep {
    pub name: String,
    pub prompt_template: String, // Template containing placeholder "{input}"
}

/// Universal modular runner for N steps
pub async fn run_modular_chain(
    model_name: &str,
    initial_input: &str,
    system_instruction: &str,
    steps: &[ChainStep],
) -> Result<ModularChainResult> {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);

    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .build();

    let mut current_input = initial_input.to_string();
    let mut history = Vec::with_capacity(steps.len());

    for step in steps {
        let prompt = step.prompt_template.replace("{input}", &current_input);
        
        let response = agent.prompt(&prompt).await?;
        let output = response.output().trim().to_string();

        history.push(ChainStepRecord {
            step_name: step.name.clone(),
            input_value: current_input.clone(),
            generated_output: output.clone(),
        });

        current_input = output;
    }

    Ok(ModularChainResult {
        final_output: current_input,
        history,
    })
}
```

---

### 2. Execution Entry Point (`src/main.rs`)

Demonstrates how to configure a 4-step pipeline (*Person $\rightarrow$ City $\rightarrow$ Country $\rightarrow$ Continent $\rightarrow$ Hemisphere*):

```rust
use anyhow::Result;
use dotenvy::dotenv;
use rig_sequential_chain::{run_modular_chain, ChainStep};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG MODULAR N-STEP CHAIN (XAI & NEO4J READY) ===\n");

    let model_name = "qwen3.5:4b";
    let system_instruction = "You are a precise assistant. Respond concisely and provide only the requested name.";

    let steps = vec![
        ChainStep {
            name: "find_city".to_string(),
            prompt_template: "In which city was the person {input} born or did they live? Write only the name of the city.".to_string(),
        },
        ChainStep {
            name: "find_country".to_string(),
            prompt_template: "In which country is the city of {input} located? Write only the name of the country.".to_string(),
        },
        ChainStep {
            name: "find_continent".to_string(),
            prompt_template: "On which continent is the country {input} located? Write only the name of the continent.".to_string(),
        },
        ChainStep {
            name: "find_hemisphere".to_string(),
            prompt_template: "In which hemisphere (northern/southern) does the continent {input} predominantly lie? Write only the hemisphere.".to_string(),
        },
    ];

    let initial_person = "obama";
    println!("Launching a multi-step modular chain with {} steps for input: '{}'\n", steps.len(), initial_person);

    match run_modular_chain(model_name, initial_person, system_instruction, &steps).await {
        Ok(result) => {
            println!("=== FINAL RESULT ===");
            println!("{}\n", result.final_output);

            println!("=== XAI & NEO4J REASONING AUDIT TRAIL ===");
            for (i, record) in result.history.iter().enumerate() {
                println!("Step {}: [{}]", i + 1, record.step_name);
                println!("  Input (Context In):  {}", record.input_value);
                println!("  Output (Model Out):  {}\n", record.generated_output);
            }

            let json_memory = serde_json::to_string_pretty(&result.history).unwrap();
            println!("--- JSON representation for Neo4j/XAI ---");
            println!("{}", json_memory);
        }
        Err(e) => eprintln!("Error during execution: {}", e),
    }

    Ok(())
}
```

---

## 🧪 Testing

The integration test suite (`tests/modular_chain_tests.rs`) verifies step execution count, step output propagation, and final history matching:

```bash
cargo test
```

---

## 🏃 Building and Running

1. Make sure **Ollama** is running locally:
   ```bash
   ollama pull qwen3.5:4b
   ```

2. Build and run the project:
   ```bash
   cargo build
   cargo run
   ```
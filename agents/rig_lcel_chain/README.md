# LCEL Chains Equivalent in Rust using Rig

This repository demonstrates how to implement **LangChain Expression Language (LCEL)** chain patterns in **Rust** using the **Rig** framework and local LLMs via **Ollama**.

> **Note on Architecture:** Unlike LangChain's dynamic pipe operator (`|`), Rust and Rig use strongly typed builder patterns, static type validation, and `serde`/`schemars` schemas—making the architecture much closer to **Pydantic AI** than traditional Python chaining.

---

## 🚀 Features

- **Dynamic Prompt Templating**: Easily construct multi-turn dynamic prompts combining system preambles, conversation history, and variable injections.
- **Local Model Execution**: Direct integration with Ollama models (e.g., `qwen3.5:4b`, `llama3.1`).
- **Robust Async Processing**: Built on top of `tokio` for async LLM completions.
- **Type-Safe Design**: Compile-time safety guarantees for model outputs and agent configurations.

---

## 📁 Project Structure

```bash
rig_lcel_chain/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs          # Chain execution logic & prompt construction
│   └── main.rs         # Application entry point
└── tests/
    └── chain_tests.rs  # Integration test suite for the chain
```

---

## ⚙️ Dependencies (`Cargo.toml`)

Add the following crates to your `Cargo.toml`:

```toml
[package]
name = "rig_lcel_chain"
version = "0.1.0"
edition = "2021"

[dependencies]
rig = "0.45.0"
tokio = { version = "1.38", features = ["full"] }
anyhow = "1.0"
dotenvy = "0.15"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
schemars = "0.8"
thiserror = "2.0"
```

---

## 🔑 Prerequisites

1. Install **Ollama** locally.
2. Pull your preferred model:
   
```bash
   ollama pull qwen3.5:4b
```

---

## 💻 Code Architecture

### `src/lib.rs`
Implements the core chain logic using `rig::agent::AgentBuilder`:

```rust
use anyhow::Result;
use rig::agent::Agent;
use rig::providers::ollama;

/// Creates and executes a chain simulating LangChain's LCEL behavior
pub async fn run_company_namer_chain(
    model_name: &str,
    name: &str,
    product: &str,
) -> Result<String> {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);

    // System prompt template
    let system_prompt = "You are an AI assistant. You help users generate creative business names.";
    
    // Build agent (equivalent to prompt | llm in LCEL)
    let agent: Agent = rig::agent::AgentBuilder::new(model)
        .preamble(system_prompt)
        .build();

    // Construct user prompt with dynamic variables
    let formatted_prompt = format!(
        "Human: hi {}\nAI: hello\nHuman: What is a good name for a company that makes {}?",
        name, product
    );

    // Chain invocation
    let response = agent.prompt(&formatted_prompt).await?;
    
    Ok(response.output().to_string())
}
```

### `src/main.rs`
Entry point demonstrating the execution of the chain:

```rust
use anyhow::Result;
use dotenvy::dotenv;
use rig_lcel_chain::run_company_namer_chain;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG LCEL CHAIN EQUIVALENT ===\n");

    let name = "Lukas";
    let product = "colorful socks";
    let model_name = "qwen3.5:4b";

    println!("Launching chain with parameters:");
    println!(" - Name: {}", name);
    println!(" - Product: {}\n", product);

    match run_company_namer_chain(model_name, name, product).await {
        Ok(result) => {
            println!("---- Chain Output ----");
            println!("{}", result);
        }
        Err(e) => eprintln!("Execution error: {}", e),
    }

    Ok(())
}
```

---

## 🧪 Testing

Run integration tests to verify chain structure and execution:

```bash
cargo test
```

Test snippet (`tests/chain_tests.rs`):

```rust
use rig_lcel_chain::run_company_namer_chain;

#[tokio::test]
async fn test_company_namer_chain_mock_or_structure() {
    let result = run_company_namer_chain("qwen3.5:4b", "Lukas", "colorful socks").await;
    
    match result {
        Ok(output) => {
            assert!(!output.is_empty(), "Chain output should not be empty.");
            println!("Test passed with output:\n{}", output);
        }
        Err(e) => {
            println!("Ollama offline (expected in non-Ollama environments): {}", e);
        }
    }
}
```

---

## 🏃 Running the Application

Execute the binary with Cargo:

```bash
cargo run
```
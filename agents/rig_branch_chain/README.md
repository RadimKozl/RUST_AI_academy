# RunnableBranch Conditional Routing Pattern in Rust (Rig Framework)

This project translates LangChain's `RunnableBranch` conditional routing pattern (`if/elif/else` logic over LCEL runnables) into a high-performance, type-safe Rust architecture using the **Rig** framework and local LLMs via **Ollama**.

It demonstrates how to dynamically evaluate incoming parameters (e.g., requested tone or style) to select and execute the corresponding agent branch, falling back to a default branch if no conditions are matched. The system maintains full audit logging for XAI (Explainable AI) and graph database storage (e.g., Neo4j).

## 🚀 Key Features

* **Dynamic Conditional Routing**: Replaces Python dynamic branching with type-safe Rust conditional logic evaluating incoming payload parameters.
* **Specialized Agent Branches**: Dynamically instantiates agents configured with custom preambles and prompt templates based on the matched rule (`formal_branch`, `funny_branch`, or `default_casual_branch`).
* **XAI & Audit Trail Ready**: Logged step records (`BranchStepRecord`) record the exact branch selected, the input evaluated, and the generated response for auditability and graph memory integration.
* **Modular Design**: Clean isolation between library execution logic, test suites, and executable runners.

## 📁 Project Structure

```
rig_branch_chain/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                   # Core routing logic, agent builder, and step logger
│   └── main.rs                  # Application runner executing multiple tone test cases
└── tests/
    └── branch_chain_tests.rs    # Integration test suite for branch matching
```

## ⚙️ Dependencies (`Cargo.toml`)

Ensure your `Cargo.toml` contains the following dependencies:

```toml
[package]
name = "rig_branch_chain"
version = "0.1.0"
edition = "2021"

[dependencies]
rig = "0.46.0"
tokio = { version = "1.38", features = ["full"] }
anyhow = "1.0"
dotenvy = "0.15"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

## 💻 Architecture & Code Overview

### 1. Library Logic & Branch Evaluator (`src/lib.rs`)

```rust
use anyhow::Result;
use rig::providers::ollama;
use serde::{Deserialize, Serialize};

/// Single-step record for XAI / Neo4j memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchStepRecord {
    pub step_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// Result of executing a conditional branch (RunnableBranch)
#[derive(Debug, Serialize, Deserialize)]
pub struct BranchChainResult {
    pub matched_branch: String,
    pub final_output: String,
    pub history: Vec<BranchStepRecord>,
}

/// Universal runner simulating the behavior of RunnableBranch
pub async fn run_branch_chain(
    model_name: &str,
    topic: &str,
    tone: &str,
) -> Result<BranchChainResult> {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);

    let tone_lower = tone.to_lowercase();

    // 1. Evaluation of conditions (RunnableBranch logical decision-making)
    let (matched_branch, system_instruction, prompt_template) = if tone_lower.contains("formal") {
        (
            "formal_branch",
            "You are an academic expert. Explain topics in an extremely academic, formal, and lengthy style.",
            "Explain the topic {topic} in an extremely academic, formal, and lengthy style.",
        )
    } else if tone_lower.contains("joke") || tone_lower.contains("funny") {
        (
            "funny_branch",
            "You are a friendly and humorous assistant.",
            "Explain the topic {topic} in a very informal, friendly, and brief style with a touch of humor.",
        )
    } else {
        // ELSE (Default) branch
        (
            "default_casual_branch",
            "You are a helpful general assistant.",
            "Explain the topic {topic} in a very informal, friendly, and brief style.",
        )
    };

    let mut history = vec![BranchStepRecord {
        step_name: "branch_evaluator".to_string(),
        input_value: format!("Topic: {} | Tone: {}", topic, tone),
        generated_output: matched_branch.to_string(),
    }];

    // 2. Constructing the agent with the selected system instruction
    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .build();

    let final_prompt = prompt_template.replace("{topic}", topic);

    // 3. Generating the final response
    let response = agent.prompt(&final_prompt).await?;
    let final_output = response.output().trim().to_string();

    history.push(BranchStepRecord {
        step_name: matched_branch.to_string(),
        input_value: final_prompt,
        generated_output: final_output.clone(),
    });

    Ok(BranchChainResult {
        matched_branch: matched_branch.to_string(),
        final_output,
        history,
    })
}
```

### 2. Application Runner (`src/main.rs`)

```rust
use anyhow::Result;
use dotenvy::dotenv;
use rig_branch_chain::run_branch_chain;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG RUNNABLE BRANCH (CONDITIONAL ROUTING) ===\n");

    let model_name = "qwen3.5:4b";
    let topic = "quantum mechanics";

    let test_cases = vec![
        ("formal", "--- Example 1: Formal Inquiry ---"),
        ("joke", "--- Example 2: Funny / Informal Inquiry ---"),
        ("neutral", "--- Example 3: Default (Else Branch) ---"),
    ];

    for (tone, title) in test_cases {
        println!("{}", title);
        println!("TONE: {}\n", tone);

        match run_branch_chain(model_name, topic, tone).await {
            Ok(res) => {
                println!("Matched Branch: [{}]", res.matched_branch);
                let preview: String = res.final_output.chars().take(120).collect();
                println!("START: {}...\n", preview);

                // XAI / Neo4j Audit Trail
                let json_audit = serde_json::to_string_pretty(&res.history).unwrap();
                println!("--- XAI Audit Trail ---");
                println!("{}\n", json_audit);
            }
            Err(e) => eprintln!("Error: {}", e),
        }
        println!("--------------------------------------------------\n");
    }

    Ok(())
}
```

## 🧪 Testing

Run the automated integration tests:

```bash
cargo test
```

## 🏃 Execution Steps

1. Start your local Ollama instance and ensure your model is ready:
   ```bash
   ollama pull qwen3.5:4b
   ```

2. Compile and run the application:
   ```bash
   cargo run
   ```
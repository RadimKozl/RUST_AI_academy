# Boolean Rule-Based Dynamic Router in Rust (Rig Framework)

This project implements a modular, type-safe, and rule-driven **Dynamic Router (Conditional Routing)** using the **Rig** framework and local LLMs via **Ollama**.

It serves as an advanced Rust replacement for LangChain's `RunnableLambda` routing patterns. Instead of hardcoded `if/else` statements, it evaluates a recursive boolean expression tree (`AND`, `OR`, `NOR`, `NOT`) over an extracted context map to dynamically select and execute specialized agents.

## 🚀 Key Features

* **Tree-Based Boolean Logic Engine**: Supports composite conditions (`LogicCondition`) with nested `And`, `Or`, `Nor`, and `Not` operators without hardcoded branching.
* **Context-Driven Evaluation**: Evaluates rules against an extracted context map (`HashMap<String, String>`) containing extracted topics, intent classifications, or original user inputs.
* **XAI & Audit-Log Ready**: Every decision (classification step and matched rule execution) is captured as a structured audit trail (`RouterStepRecord`) ready for graph database storage (e.g., Neo4j).
* **Decoupled Architecture**: All system instructions, prompt templates, and routing conditions are configured declaratively in the runner (`main.rs`).

## 📁 Project Structure

```bash
rig_conditions_chain/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                 # Boolean engine, evaluator, and rule-based runner
│   └── main.rs                # Application entry point demonstrating AND/OR/NOR routing
└── tests/
    └── router_rule_tests.rs   # Unit tests for logic evaluator & integration test
```

## ⚙️ Dependencies (`Cargo.toml`)

Ensure your `Cargo.toml` contains the required dependencies:

```toml
[package]
name = "rig_conditions_chain"
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

### 1. Boolean Logic Engine & Library (`src/lib.rs`)

The core engine uses a recursive enum `LogicCondition` and an `evaluate_condition` function to evaluate dynamic rules against a runtime context map:

```rust
use anyhow::Result;
use rig::providers::ollama;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Recursive boolean conditions (AND, OR, NOR, NOT)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicCondition {
    Contains { target: String, keyword: String },
    Equals { target: String, value: String },
    And(Vec<LogicCondition>),
    Or(Vec<LogicCondition>),
    Nor(Vec<LogicCondition>),
    Not(Box<LogicCondition>),
}

/// Evaluates nested boolean expressions against a context map
pub fn evaluate_condition(cond: &LogicCondition, context: &HashMap<String, String>) -> bool {
    match cond {
        LogicCondition::Contains { target, keyword } => {
            context.get(target).map_or(false, |val| val.to_lowercase().contains(&keyword.to_lowercase()))
        }
        LogicCondition::Equals { target, value } => {
            context.get(target).map_or(false, |val| val.eq_ignore_ascii_case(value))
        }
        LogicCondition::And(conditions) => conditions.iter().all(|c| evaluate_condition(c, context)),
        LogicCondition::Or(conditions) => conditions.iter().any(|c| evaluate_condition(c, context)),
        LogicCondition::Nor(conditions) => !conditions.iter().any(|c| evaluate_condition(c, context)),
        LogicCondition::Not(c) => !evaluate_condition(c, context),
    }
}
```

### 2. Application Runner (`src/main.rs`)

Configures complex conditional routing rules using `And`, `Or`, and `Nor`:

```rust
use anyhow::Result;
use dotenvy::dotenv;
use rig_conditions_chain::{run_rule_based_router, LogicCondition, RoutingRule};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG BOOLEAN RULE-BASED ROUTER (AND/OR/NOR) ===\n");

    let model_name = "qwen3.5:4b";
    let classifier_system = "You are a keyword extractor. Respond with a single category word: `LangChain`, `Anthropic`, or `Other`.";
    let classifier_prompt = "Classify this question into one category (`LangChain`, `Anthropic`, `Other`):\n<question>\n{question}\n</question>\nCategory:";

    let rules = vec![
        // Rule 1 (OR): Topic is Anthropic OR question contains "claude"
        RoutingRule {
            name: "anthropic_branch".to_string(),
            condition: LogicCondition::Or(vec![
                LogicCondition::Contains { target: "topic".to_string(), keyword: "anthropic".to_string() },
                LogicCondition::Contains { target: "question".to_string(), keyword: "claude".to_string() },
            ]),
            system_instruction: "You are an expert in Anthropic.".to_string(),
            prompt_template: "Question: {question}\nAnswer:".to_string(),
        },
        // Rule 2 (AND): Topic is LangChain AND question contains "news"
        RoutingRule {
            name: "langchain_news_branch".to_string(),
            condition: LogicCondition::And(vec![
                LogicCondition::Contains { target: "topic".to_string(), keyword: "langchain".to_string() },
                LogicCondition::Contains { target: "question".to_string(), keyword: "news".to_string() },
            ]),
            system_instruction: "You are a tech journalist reporting on LangChain updates.".to_string(),
            prompt_template: "Provide a breaking news update for: {question}\nAnswer:".to_string(),
        },
        // Rule 3 (NOR): Topic is NEITHER LangChain NOR Anthropic
        RoutingRule {
            name: "general_branch".to_string(),
            condition: LogicCondition::Nor(vec![
                LogicCondition::Contains { target: "topic".to_string(), keyword: "langchain".to_string() },
                LogicCondition::Contains { target: "topic".to_string(), keyword: "anthropic".to_string() },
            ]),
            system_instruction: "You are a helpful general assistant.".to_string(),
            prompt_template: "Question: {question}\nAnswer:".to_string(),
        },
    ];

    let default_rule = RoutingRule {
        name: "fallback_branch".to_string(),
        condition: LogicCondition::Equals { target: "topic".to_string(), value: "".to_string() },
        system_instruction: "You are a helpful assistant.".to_string(),
        prompt_template: "Question: {question}\nAnswer:".to_string(),
    };

    let test_questions = vec![
        "Tell me what Claude can do.",
        "Give me news about LangChain.",
        "Explain Newton's 1st law.",
    ];

    for q in test_questions {
        if let Ok(res) = run_rule_based_router(model_name, q, classifier_system, classifier_prompt, &rules, &default_rule).await {
            println!("Matched Rule: [{}]", res.matched_rule);
            println!("Final Answer:\n{}\n", res.final_output);
        }
    }

    Ok(())
}
```

## 🧪 Testing

Run both unit tests (verifying the boolean logic tree) and integration tests:

```bash
cargo test
```

## 🏃 Building & Running

1. Ensure Ollama is running locally:
   ```bash
   ollama pull qwen3.5:4b
   ```
2. Run the application:
   ```bash
   cargo run
   ```
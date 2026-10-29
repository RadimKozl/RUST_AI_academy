# Rust AI Agent with Tool Calling (Rig Framework v0.45+)

This project demonstrates how to build autonomous AI agents in **Rust** using the **Rig** framework (v0.45+). It showcases **Tool Calling** (Function Calling) capabilities, allowing a local LLM (via **Ollama**) to dynamically choose and execute custom Rust tools.

## 🚀 Features

1. **Stock Price Lookup Tool (`StockPriceTool`)**: A custom tool returning stock prices for specific ticker symbols.
2. **Wikipedia Search Tool (`WikiSearchTool`)**: A real-time integration with the MediaWiki HTTP API (`https://cs.wikipedia.org/w/api.php`) using `reqwest` to search and fetch summaries.
3. **Rig 0.45+ API Compatibility**: Implements the latest `Tool` trait standards (`description()`, `parameters()`, and `&mut ToolContext` support).
4. **Autonomous Execution**: Automatically handles multi-turn agent execution loops using `.max_turns(5)`.

---

## 📁 Project Structure

```bash
rig_tool_calling/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs          # Tool implementations (Stock & Wiki) and Agent factories
│   └── main.rs         # Orchestration binary testing both tools
└── tests/
    └── agent_tools_tests.rs # Integration unit tests for tools
```

---

## ⚙️ Configuration (`Cargo.toml`)

Ensure your `Cargo.toml` includes all required dependencies for Rig, Tokio, Serde, Schemars, and Reqwest:

```toml
[package]
name = "rig_tool_calling"
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
reqwest = { version = "0.12", features = ["json"] }
urlencoding = "2.1"
```

---

## 🔑 Prerequisites

Make sure you have **Ollama** installed and running locally with a model that supports tool calling (e.g., `llama3.1` or `qwen2.5`):

```bash
ollama run llama3.1
```

---

## 💻 Code Overview

### `src/lib.rs`
Contains the implementation of `StockPriceTool` and `WikiSearchTool`, along with the `create_agent_with_all_tools` factory.

```rust
use anyhow::Result;
use rig::agent::Agent;
use rig::providers::ollama;
use rig::tool::{Tool, ToolContext};
use serde::Deserialize;
use schemars::JsonSchema;

// --- Stock Price Tool ---
#[derive(Deserialize, JsonSchema)]
pub struct StockArgs {
    pub ticker: String,
}

#[derive(Debug, thiserror::Error)]
#[error("Stock tool error")]
pub struct StockError;

pub struct StockPriceTool;

impl Tool for StockPriceTool {
    const NAME: &str = "get_stock_price";
    type Error = StockError;
    type Args = StockArgs;
    type Output = String;

    fn description(&self) -> String {
        "Retrieve the current stock price for a given ticker symbol.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(StockArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let price = match args.ticker.to_uppercase().as_str() {
            "MSFT" => "$513.57",
            "AAPL" => "$225.00",
            _ => "$100.00",
        };
        Ok(format!("The current price of {} is {}", args.ticker, price))
    }
}

// --- Wikipedia Search Tool ---
#[derive(Deserialize, JsonSchema)]
pub struct WikiArgs {
    pub query: String,
}

#[derive(Debug, thiserror::Error)]
#[error("Wikipedia tool error: {0}")]
pub struct WikiError(#[from] reqwest::Error);

pub struct WikiSearchTool;

impl Tool for WikiSearchTool {
    const NAME: &str = "search_wiki";
    type Error = WikiError;
    type Args = WikiArgs;
    type Output = String;

    fn description(&self) -> String {
        "Search Wikipedia for information about a specific topic, company, or person using the MediaWiki API.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(WikiArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let url = format!(
            "https://cs.wikipedia.org/w/api.php?action=query&list=search&srsearch={}&format=json",
            urlencoding::encode(&args.query)
        );

        let client = reqwest::Client::builder()
            .user_agent("RigRustAgent/1.0 (educational project)")
            .build()?;

        let resp = client.get(&url).send().await?;
        let json: serde_json::Value = resp.json().await?;

        if let Some(search_results) = json.pointer("/query/search").and_then(|v| v.as_array()) {
            if let Some(first_result) = search_results.first() {
                let title = first_result.get("title").and_then(|v| v.as_str()).unwrap_or("Unknown");
                let snippet = first_result.get("snippet").and_then(|v| v.as_str()).unwrap_or("No snippet available.");
                
                let clean_snippet = snippet
                    .replace("<span class=\"searchmatch\">", "")
                    .replace("</span>", "")
                    .replace("&quot;", "\"")
                    .replace("&#039;", "'");

                return Ok(format!("Title: {}\nSummary: {}...", title, clean_snippet));
            }
        }

        Ok("No relevant Wikipedia article found for the given query.".to_string())
    }
}

// --- Agent Factories ---
pub fn create_agent_with_tools(model_name: &str, system_instruction: &str) -> Agent {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);

    rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .tool(StockPriceTool)
        .build()
}

pub fn create_agent_with_all_tools(model_name: &str, system_instruction: &str) -> Agent {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);

    rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .tool(StockPriceTool)
        .tool(WikiSearchTool)
        .build()
}

pub async fn run_prompt(agent: &Agent, prompt: &str) -> Result<String> {
    let response = agent.prompt(prompt).max_turns(5).await?;
    Ok(response.output().to_string())
}
```

### `src/main.rs`
Demonstrates sequential execution of stock price querying and live Wikipedia searching:

```rust
use anyhow::Result;
use dotenvy::dotenv;
use rig_tool_calling::{create_agent_with_all_tools, run_prompt};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST AI AGENT WITH ALL TOOLS (RIG FRAMEWORK) ===\n");

    let agent = create_agent_with_all_tools(
        "llama3.1", 
        "You are a helpful assistant. You MUST use the available tools (search_wiki or get_stock_price) when asked factual or financial questions.",
    );

    // 1. Stock Tool Prompt
    let stock_prompt = "What is the current stock price for MSFT?";
    println!("--- Test 1: Stock Price Tool ---");
    println!("User query: {}\n", stock_prompt);

    match run_prompt(&agent, stock_prompt).await {
        Ok(res) => println!("Agent's final response:\n{}\n", res),
        Err(e) => eprintln!("Error during agent execution: {}\n", e),
    }

    // 2. Wikipedia Tool Prompt
    let wiki_prompt = "Who is the founder of Microsoft and in what year was the company founded?";
    println!("--- Test 2: Wikipedia Tool ---");
    println!("User query: {}\n", wiki_prompt);

    match run_prompt(&agent, wiki_prompt).await {
        Ok(res) => println!("Agent's final response:\n{}\n", res),
        Err(e) => eprintln!("Error during agent execution: {}\n", e),
    }

    Ok(())
}
```

---

## 🧪 Testing & Execution

- **Run Unit Tests:**
  ```bash
  cargo test
  ```

- **Run the Agent Application:**
  ```bash
  cargo run
  ```
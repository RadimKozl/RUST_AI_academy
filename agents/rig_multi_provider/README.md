# Multi-Provider AI Application in Rust (Rig Framework)

This project serves as a practical tutorial for developing multi-provider AI applications in **Rust** using the **Rig** framework (v0.45+). It demonstrates a modular, wrapper-based architecture designed to seamlessly query both local and cloud-based Large Language Models (LLMs).

## 🚀 Supported Providers

1. **Ollama** (Local execution for models like `gemma3n:e4b` or `llama3`)
2. **Mistral AI** (Cloud API integration with rate-limiting handling)
3. **Hugging Face** (Serverless Inference Endpoint API)

## 📁 Project Structure

```bash
rig_multi_provider/
├── Cargo.toml
├── .env
├── README.md
├── src/
│   ├── lib.rs          # Prompt templates, wrappers, and agent factories
│   └── main.rs         # Main orchestration executable
└── tests/
    └── prompt_tests.rs # Unit tests for prompt templating
```

## ⚙️ 1. Configuration (`Cargo.toml`)

Ensure your `Cargo.toml` contains the required dependencies for `rig`, `tokio`, and environment management:

```toml
[package]
name = "rig_multi_provider"
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

## 🔑 2. Environment Setup (`.env`)

Create a `.env` file in the root directory of your project and populate it with your provider credentials:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
HF_TOKEN=your_huggingface_token_here
```

*(Note: Ollama runs locally by default at `http://localhost:11434` and does not require an API key).*

## 💻 3. Code Implementation

### `src/lib.rs` (Wrappers & Templates)

```rust
use anyhow::Result;
use rig::agent::Agent;
use rig::providers::{huggingface, mistral, ollama};

pub struct PromptTemplate {
    template: String,
}

impl PromptTemplate {
    pub fn new(template: &str) -> Self {
        Self {
            template: template.to_string(),
        }
    }

    pub fn format_product(&self, product: &str) -> String {
        self.template.replace("{product}", product)
    }

    pub fn format_name_and_product(&self, name: &str, product: &str) -> String {
        self.template
            .replace("{name}", name)
            .replace("{product}", product)
    }
}

pub fn create_ollama_agent(model_name: &str, system_instruction: &str) -> Agent {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);
    rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .build()
}

pub fn create_mistral_agent(api_key: &str, model_name: &str, system_instruction: &str) -> Agent {
    let client = mistral::Client::new(api_key);
    let model = client.completion(model_name);
    rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .build()
}

pub fn create_huggingface_agent(hf_token: &str, model_name: &str, system_instruction: &str) -> Agent {
    let client = huggingface::Client::new(hf_token);
    let model = client.completion(model_name);
    rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .build()
}

pub async fn run_prompt(agent: &Agent, prompt: &str) -> Result<String> {
    let response = agent.prompt(prompt).await?;
    Ok(response.output().to_string())
}
```

### `src/main.rs` (Orchestration Executable)

```rust
use anyhow::Result;
use dotenvy::dotenv;
use rig::providers::mistral;
use rig_multi_provider::{
    create_huggingface_agent, create_mistral_agent, create_ollama_agent, run_prompt,
    PromptTemplate,
};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== MULTI-PROVIDER AI TUTORIAL IN RUST ===\n");

    let template = PromptTemplate::new("What is a good name for a company that makes {product}?");
    let prompt = template.format_product("colorful socks");

    println!("Generated Prompt: {}\n", prompt);

    // 1. Ollama (Local)
    println!("--- 1. Calling Ollama ---");
    let ollama_agent = create_ollama_agent("gemma3n:e4b", "You are a creative assistant.");
    match run_prompt(&ollama_agent, &prompt).await {
        Ok(res) => println!("Ollama Response:\n{}\n", res),
        Err(e) => eprintln!("Ollama Error (check if service is running on 11434): {}\n", e),
    }

    // Delay to respect API rate limits
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // 2. Mistral AI (Cloud)
    println!("--- 2. Calling Mistral AI ---");
    let mistral_key = env::var("MISTRAL_API_KEY").unwrap_or_default();
    if !mistral_key.is_empty() {
        let mistral_agent = create_mistral_agent(
            &mistral_key,
            mistral::MISTRAL_SMALL,
            "You are a poetic assistant.",
        );
        match run_prompt(&mistral_agent, &prompt).await {
            Ok(res) => println!("Mistral AI Response:\n{}\n", res),
            Err(e) => eprintln!("Mistral AI Error (check rate limit 429): {}\n", e),
        }
    } else {
        println!("MISTRAL_API_KEY not found in .env, skipping.\n");
    }

    // Delay to respect API rate limits
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // 3. Hugging Face (Endpoint)
    println!("--- 3. Calling Hugging Face ---");
    let hf_token = env::var("HF_TOKEN").unwrap_or_default();
    if !hf_token.is_empty() {
        let hf_agent = create_huggingface_agent(
            &hf_token,
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "You are a branding assistant.",
        );
        match run_prompt(&hf_agent, &prompt).await {
            Ok(res) => println!("Hugging Face Response:\n{}\n", res),
            Err(e) => eprintln!("Hugging Face Error: {}\n", e),
        }
    } else {
        println!("HF_TOKEN not found in .env, skipping.\n");
    }

    Ok(())
}
```

## 🧪 Running Tests & Execution

* **Run unit tests:**
  ```bash
  cargo test
  ```

* **Run the application:**
  ```bash
  cargo run
  ```
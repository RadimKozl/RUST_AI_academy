# Rig Prompt Tutorial (LangChain to Rust)

A practical tutorial demonstrating how to replace **Python's LangChain** components with **Rust's Rig framework** for building AI agents and prompt engineering.

This project uses a local **Ollama** instance as the LLM provider, ensuring complete privacy, zero API costs, and native execution.

---

## 🎯 What You'll Learn

1. **Prompt Templating:** Mapping `PromptTemplate` concepts from LangChain directly into clean, idiomatic Rust.
2. **System Instruction / Persona Alignment:** Creating styled AI agents with system preambles (equivalent to `ChatPromptTemplate` / `SystemMessage`).
3. **Local LLM Integration:** Communicating with Ollama (`gemma3n:e4b` or any other local model) directly in Rust using async `tokio`.

---

## 🛠️ Prerequisites

- **Rust:** Installed via [rustup](https://rustup.rs/) (version 1.80+ recommended).
- **Ollama:** Installed and running locally ([ollama.com](https://ollama.com/)).
  - Make sure you have downloaded the model configured in `main.rs`:
    ```bash
    ollama pull gemma3n:e4b
    ```

---

## 📁 Project Structure

```bash
rig_prompt_tutorial/
├── Cargo.toml          # Project dependencies (Rig, Tokio, Anyhow)
├── src/
│   ├── lib.rs          # Core logic (PromptTemplate & create_styled_agent)
│   └── main.rs         # Application entry point
└── tests/
    └── prompt_tests.rs # Unit tests for prompt formatting
```

---

## ⚙️ Dependencies (`Cargo.toml`)

```toml
[package]
name = "rig_prompt_tutorial"
version = "0.1.0"
edition = "2021"

[dependencies]
rig = "0.45"
tokio = { version = "1.38", features = ["full"] }
anyhow = "1.0"

[dev-dependencies]
tempfile = "3.10"
```

---

## 🚀 Running the Project

### 1. Run Unit Tests
To verify that prompt templating logic works as expected:

```bash
cargo test
```

### 2. Run the Main Application
Ensure your Ollama service is running (`http://localhost:11434`), then execute:

```bash
cargo run
```

---

## 💡 Example Output

```text
=== 1. Simple Prompt & Templates (LangChain PromptTemplate -> Rust) ===
Generated prompt: What is a good name for a company that makes colorful socks?

=== 2. Chat Template and Agent (LangChain ChatPromptTemplate -> Rig Agent) ===
Sending a query to the agent with the role 'Shakespeare'...

--- Agent's response ---
Hark! A most delightful query! For a company that doth craft socks of vibrant hue, I offer these names, fit for a king's apparel, or a commoner's merriment:

* The Chromatic Foot: A name of noble sound, hinting at a spectrum of colors.
* Sockalope: A whimsical blend of "sock" and the mythical creature, promising fantastical designs.
* Hue & Sole: A playful phrase, speaking to both color and the foundation of the foot.
* Rainbow Threads: A simple, yet evocative name, conjuring images of joyful textiles.

I trust one of these doth strike thee as most pleasing!
```
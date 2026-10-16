# Rust Ollama Function Calling / Tool Use Client (`ollama_tool_use`)

This project serves as a tutorial and practical implementation of **Function Calling (Tool Use)** with local **Ollama** models in **Rust**. 

It demonstrates the complete multi-step agent loop: sending a user prompt alongside custom tool schemas, capturing model-generated tool calls (`tool_calls`), executing local functions (simulating financial data lookups), feeding the execution results back into the conversation history, and receiving the final synthesized response from the model.

---

## **Project Structure**

```bash
ollama_tool_use/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                   # Core data structures including ToolCall, FunctionCall, and updated Message types
│   └── main.rs                  # Executable application handling the full agentic tool-use loop
└── tests/
    └── integration_tests.rs     # Integration tests verifying correct serialization of tool definitions and payloads
```

## **Prerequisites**

1. Installed **Rust** (Edition 2024+, managed via `rustup`).
2. Running **Ollama** instance locally (default port `http://localhost:11434`).
3. Pulled model trained for tool use (e.g., `granite4.2:3b`):

```bash
ollama pull mistral:7b
```

## **Dependencies (`Cargo.toml`)**

The project relies on the following key libraries:

- `tokio` – Asynchronous runtime.
- `reqwest` – HTTP client with JSON support.
- `serde` & `serde_json` – Data serialization and dynamic JSON value handling.

## **How to Run the Application**

To execute the main program, which initializes tool schemas, triggers a function call from the model, executes the function locally, and resolves the final answer, run:

```bash
cargo run
```

## **How to Run Tests**

The project includes integration tests to verify proper payload formatting and tool-call schema serialization:

```bash
cargo test
```
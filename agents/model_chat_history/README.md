# Rust Ollama Chat History Client (`ollama_chat_history`)

This project serves as a tutorial and practical demonstration of how to manage **chat history (conversation context)** when communicating with local **Ollama** models in **Rust**.

It builds upon a Python Jupyter notebook to demonstrate the fundamental difference between stateless chats (where the model has no memory of previous messages) and stateful multi-turn conversations where messages accumulate dynamically in a vector (`Vec<Message>`).

---

## **Project Structure**

```bash
ollama_chat_history/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                   # Core data structures (Message, ChatRequest, ChatResponse) and API client functions
│   └── main.rs                  # Executable application handling multi-turn conversation history
└── tests/
    └── integration_tests.rs     # Integration tests verifying serialization of complete chat histories
```

## **Prerequisites**

1. Installed Rust (Edition 2024+, managed via `rustup`).
2. Running Ollama instance locally (default port `http://localhost:11434`).
3. Pulled model compatible with Ollama (e.g., `gemma3n:e4b`):

```bash
ollama pull gemma3n:e4b
```

## **Dependencies (`Cargo.toml`)**

The project relies on the following key libraries:
- `tokio` – Asynchronous runtime.
- `reqwest` – HTTP client with JSON support.
- `serde` & `serde_json` – Data serialization and deserialization.

## **How to Run the Application**

To execute the main program, which initializes conversation history, sends an initial prompt, feeds the model's response back into the history stack, and requests a contextual follow-up action, run:

```bash
cargo run
```

## **How to Run Tests**

The project includes integration tests to ensure that multi-turn conversation vectors serialize into valid JSON payloads expected by Ollama:

```bash
cargo test
```
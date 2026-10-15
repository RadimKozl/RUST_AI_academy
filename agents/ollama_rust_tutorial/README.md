# Ollama Rust Tutorial (`ollama_rust_tutorial`)

This project serves as a demonstration and tutorial for integrating the local platform **Ollama** with the **Rust** programming language. It includes both a modular structure for HTTP calls (`/api/chat` and `/api/generate`) and integration tests.

The initial inspiration was a Python-based Jupyter notebook for working with local AI models (e.g., `gemma3n:e4b`).

---

## **Project Structure**

```bash
ollama_rust_tutorial/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs       # Data structures, message models, and API functions
│   └── main.rs      # Executable binary application
└── tests/
    └── integration_tests.rs # Integration tests for serialization and deserialization

```

## **Requirements**

1. Rust installed (the latest version, managed via `rustup`, is recommended).
2. Ollama running locally on your computer (default port: `http://localhost:11434`).
3. The model used in the code (e.g., `gemma3n:e4b`) downloaded.

```bash
ollama pull gemma3n:e4b
```

## **How to launch an application**

To run the main program, which performs test calls to both Ollama endpoints (`/api/chat` and `/api/generate`), enter the following in the project's root directory:

```bash
cargo run
```

## **How to run tests**

The project includes integration tests that verify the correctness of data conversion into JSON format and the processing of responses from Ollama:

```bash
cargo test
```

## **Results of use**

```bash
PS C:\Users\kozl\ITProjects\RUST_AI_academy\agents\ollama_rust_tutorial> cargo run 
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.16s
     Running `target\debug\ollama_rust_tutorial.exe`
--- 1. Call to /api/chat (Local client) ---
Model: gemma3n:e4b
--- Response text: ---
Why don't scientists trust atoms? 

Because they make up everything! 😄



Hope that brought a smile to your face! 😊 

Do you want to hear another one? 


--- 2. Calling /api/generate (Simplified format) ---
--- Response text: ---
Why don't scientists trust atoms?

... Because they make up everything! 

😄



Hope that brought a smile to your face! 😊 
```
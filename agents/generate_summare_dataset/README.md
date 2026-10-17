# Rust Ollama Dataset Generator (`ollama_dataset_generator`)

This project serves as a tutorial and practical Rust implementation of an automated dataset generation pipeline using local **Ollama** models. A dataset suitable for training a summarization model. 

It mirrors a Python utility script by randomly selecting news topics, prompting a local model (e.g., `mistral:latest`) to generate news articles, generating summaries for each article, and appending the structured JSON results (`id`, `message`, `summarization`) to an output file.

---

## **Project Structure**

```bash
ollama_dataset_generator/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                   # Data structures (NewsItem, GenerateRequest) and API helper functions
│   └── main.rs                  # Executable pipeline looping through topic selection, generation, and JSON appending
└── tests/
    └── integration_tests.rs     # Integration tests verifying dataset serialization structures
```

## **Prerequisites**

1. Installed **Rust** (Edition 2024+, managed via `rustup`).
2. Running **Ollama** instance locally (default port `http://localhost:11434`).   
3. Pulled model compatible with Ollama (e.g., `granite4.2:3b`):
   
```bash
ollama pull granite4.2:3b
```

## **Dependencies (`Cargo.toml`)**

The project relies on the following key libraries:

- tokio – Asynchronous runtime.
- reqwest – HTTP client with JSON support.
- serde & serde_json – Data serialization and deserialization.
- rand – Random topic selection.
- uuid – Unique identifier generation for dataset items.

## **How to Run the Application**

To execute the dataset generation pipeline for a single iteration (which can be wrapped in a loop akin to the Python `tqdm` implementation), run:

```bash
cargo run
```

## **How to Run Tests**

The project includes integration tests to ensure that generated dataset elements serialize correctly into the expected JSON format:

```bash
cargo test
```

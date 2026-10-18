# Rust Ollama Dataset Translation & Summarization Pipeline (`ollama_dataset_translator`)

This project serves as a tutorial and practical Rust implementation of an automated translation and summarization pipeline using local **Ollama** models.

It mirrors a Python data pipeline that loads text entries (e.g., from a JSON dataset), translates the source text into a target language (such as Czech) using a translation model, generates a summary of the translated text, and outputs structured JSON records containing unique identifiers, translated text, and summaries.

---

## **Project Structure**

```bash
ollama_dataset_translator/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                   # Core data structures (TranslatedNewsItem, GenerateRequest) and API client helper functions
│   └── main.rs                  # Executable pipeline looping through records, translation prompts, and summarization
└── tests/
    └── integration_tests.rs     # Integration tests verifying correct serialization of translated dataset items
```

## **Prerequisites**

1. Installed **Rust** (Edition 2024+, managed via `rustup`).
2. Running **Ollama** instance locally (default port `http://localhost:11434`).   
3. Pulled models compatible with Ollama (e.g., `gemma3n:e4b and llama3.2:1b`):

```bash
ollama pull gemma3n:e4b
ollama pull llama3.2:1b
```

## **Dependencies (`Cargo.toml`)**

The project relies on the following key libraries:

- `tokio` – Asynchronous runtime.
- `reqwest` – HTTP client with JSON support.
- `serde` & `serde_json` – Data serialization and deserialization.
- `uuid` – Unique identifier generation for dataset records.

## **How to Run the Application**

To execute the translation and summarization pipeline, run:

```bash
cargo run
```

## **How to Run Tests**

The project includes integration tests to ensure that translated dataset records serialize correctly into the expected JSON format:

```bash
cargo test
```
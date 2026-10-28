# Local Micro-RAG Pipeline in Rust

A high-performance, modular, and local Retrieval-Augmented Generation (RAG) pipeline implemented entirely in **Rust**. It ingests documents and images from a local directory, generates vector embeddings, stores metadata and chunks in **SQLite**, and queries a local **Ollama** server for contextual answers.

---

## 🏗️ Architecture & Tech Stack

- **Language**: Rust (`tokio`, `reqwest`, `serde`, `rusqlite`)
- **Metadata & Vector Storage**: SQLite (`rag_database.db`) storing documents, text chunks, and serialized vector embeddings.
- **Vector Search**: Custom in-memory cosine similarity matching over stored chunks.
- **AI Models (via Ollama)**:
  - **Embeddings**: `nomic-embed-text:latest`
  - **Chat / Generation**: `gemma3n:e4b`
  - **Vision / OCR (Images)**: `qwen3-vl:2b`

---

## 📂 Project Structure

```bash
local_micro_rag/
├── Cargo.toml               # Project dependencies and configurations
├── src/
│   ├── lib.rs               # Core logic (Ollama client, SQLite database, chunking, math)
│   └── main.rs              # Document processing pipeline and interactive chat loop
├── tests/
│   └── rag_tests.rs         # Unit and integration tests (chunking, similarity, database)
├── data/
│   └── documents/           # Place your source documents (.txt, .png, .jpg) here
└── storage/                 # Automatically generated database and storage files
```

## 🚀 Getting Started

### **Prerequisites**

1. Rust toolchain installed (stable channel).
2. Ollama running locally on `http://localhost:11434` with the required models pulled:

```bash
ollama pull nomic-embed-text
ollama pull gemma3n:e4b
ollama pull qwen3-vl:2b
```

## **1. Prepare Documents**

Create a folder named `data/documents` in the root of the project and place your files inside:

- Text files (`.txt`): Read directly as raw text.
- Images (`.png`, `.jpg`): Automatically processed using the vision model (`qwen3-vl:2b`) to extract descriptive text.

## **2. Run Tests**

Verify that all unit and database integration tests pass successfully:

```bash
cargo test
```

## **3. Run the Application**

Start the pipeline. It will automatically initialize the database, scan the documents folder, chunk texts, generate embeddings, and open an interactive chat loop:

```bash
cargo run
```

## **💬 Interactive Commands**
Once the assistant is ready, you can type queries directly into the console.

- Type `/bye` to exit the conversation and clean up session states.

```bash
> What is discussed in the uploaded documents?
--- Response ---
...
> /bye
```
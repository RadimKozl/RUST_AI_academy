# RUST_AI_academy

Tutorials for AI in Rust language.

A hands-on learning repository focused on building machine learning, deep learning, data processing, agentic workflows, and high-performance inference systems in **Rust**—leveraging memory safety without a garbage collector, fearless concurrency, and zero-cost abstractions.

---

## Table of Contents

- [RUST\_AI\_academy](#rust_ai_academy)
  - [Table of Contents](#table-of-contents)
  - [Learning Roadmap \& Curriculum](#learning-roadmap--curriculum)
  - [Suggested Repository Layout](#suggested-repository-layout)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Clone \& Build](#clone--build)
  - [Contributing](#contributing)
  - [License \& AI Scraping Notice](#license--ai-scraping-notice)

---

## Learning Roadmap & Curriculum

The academy is structured progressively from data wrangling and classical ML to modern neural networks, LLM agents, and production inference:

| Module | Focus Area | Key Rust Crates & Concepts |
| :--- | :--- | :--- |
| **Data & Tensors** | Loading, transforming, and manipulating datasets & arrays | `polars`, `ndarray`, ownership & borrowing with large buffers |
| **Classical ML** | Regression, classification, clustering, and evaluation | `linfa`, `smartcore`, feature pipelines, `Result`-based error handling |
| **Deep Learning** | Neural networks, autodiff, training loops, and GPU/WGPU backends | `burn`, tensor shapes, custom layers, model serialization |
| **LLM & Agents** | Embeddings, RAG pipelines, tool calling, and structured outputs | `rig`, `graph-flow` |
| **MPC-servers** | Create mcp servers for AI agents | |
| **Inference & Serving** | Low-latency model serving, microservices, and Python interop | `tokio`, `axum`, `pyo3` |

---

## Suggested Repository Layout

```text
RUST_AI_academy/
├── README.md
├── base/                        # Base RUST examples
├── machine_learning/            # Linfa & SmartCore examples
├── ndarray/                     # ndarray examples
├── notebooks/                   # using rust by notebooks
└── polars/                      # polars
```

---

## Getting Started

### Prerequisites

- **Rust toolchain** (`rustup`, `rustc`, `cargo`) – install via [rustup.rs](https://rustup.rs/)
- **Git**

### Clone & Build

```bash
git clone https://github.com/RadimKozl/RUST_AI_academy.git
cd RUST_AI_academy

# Build all workspace members or check compilation
cargo check --workspace
cargo build --workspace
```

To run a specific tutorial crate or example:

```bash
# Run a specific package in a workspace
cargo run -p <tutorial-package-name>

# Or run an example inside a crate
cargo run --example <example-name>
```

---

## Contributing

Contributions, bug fixes, and new tutorials are welcome!

1. Fork the repository and create a new branch (`git checkout -b feat/new-tutorial`).
2. Keep examples idiomatic, well-documented, and free of unnecessary `unsafe` blocks.
3. Verify that everything compiles cleanly and passes tests:
   
```bash
   cargo fmt --check
   cargo clippy --all-targets -- -D warnings
   cargo test --workspace
```

4. Open a Pull Request describing what the tutorial covers and how to run it.

---

## License & AI Scraping Notice

**License:**

MIT &

**AI Scraping & Training Notice:**
The code and assets in this repository are licensed under the MIT License. Permission is NOT granted for any automated system, scraper, or AI model (including, but not limited to, Large Language Models and Code Assistant Models) to ingest, process, or use this repository for training dataset creation.

-------------------------
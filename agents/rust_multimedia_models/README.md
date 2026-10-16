# Rust Multimodal & VLM Ollama Client (`rust_multimedia_models`)

This project serves as a tutorial and demonstration of how to communicate with the local **Ollama** platform in **Rust** to analyze images using Vision Language Models (**VLM**).

It builds upon a Python Jupyter notebook and demonstrates loading a local image, converting it to **Base64** format, and sending an asynchronous request via the chat endpoint `/api/chat`.

---

## **Project Structure**

```bash
rust_multimedia_models/
├── Cargo.toml
├── README.md
├── images/
│   └── car_test_image.png       # Sample test image for analysis
├── src/
│   ├── lib.rs                   # Data structures with 'images' field support and API client
│   └── main.rs                  # Executable application (image encoding and model prompt)
└── tests/
    └── integration_tests.rs     # Integration tests for Base64 serialization and VLM response deserialization
```

## **Prerequisites**

1. Installed **Rust** (Edition 2024+, managed via `rustup`).
2. Running **Ollama** instance locally (default port `http://localhost:11434`).
3. Pulled multimodal model compatible with Ollama (e.g., `qwen3-vl:2b`)

    ```bash
    ollama pull qwen3-vl:2b
    ```

4. Sample image placed in the project directory as `images/car_test_image.png`.

## **Dependencies (`Cargo.toml`)**

The project relies on the following key libraries:

- `tokio` – Asynchronous runtime.
- `reqwest` – HTTP client with JSON support.
- `serde` & `serde_json` – Data serialization and deserialization.
- `base64` – Encoding binary image data into a Base64 string.

## **How to Run the Application**

To execute the main program, which loads the image, encodes it, and sends it to the multimodal model for analysis, run the following command in the root directory:

```bash
cargo run
```

## **How to Run Tests**

The project includes integration tests verifying correct multimodal request serialization (including Base64 strings) and VLM JSON response handling:

```bash
cargo test
```
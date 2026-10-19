# Rust Ollama Vision Object Detection Client (`ollama_vlm_object_detection`)

This project serves as a tutorial and practical Rust implementation of **Vision-Language Model (VLM) Object Detection** using local **Ollama** models (such as `qwen3-vl:8b`)[cite: 11].

It mirrors a Python pipeline that encodes images to Base64, enforces structured JSON responses via Ollama's API parameters (`format: "json"`), processes bounding box coordinates (`x1, y1, x2, y2`), and handles parsed detection schemas[cite: 11].

---

## **Project Structure**

```bash
ollama_vlm_object_detection/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                   # Core VLM structures (DetectionResponse, DetectedObject, ChatRequest)
│   └── main.rs                  # Application handling image encoding, object detection prompts, and JSON parsing
└── tests/
    └── integration_tests.rs     # Integration tests verifying bounding box and JSON serialization/deserialization

```

## **Prerequisites**

1. Installed Rust (Edition 2024+, managed via `rustup`).
2. Running Ollama instance locally (default port `http://localhost:11434`).
3. Pulled VLM model compatible with Ollama (e.g., `qwen3-vl:8b`):

```bash
ollama pull qwen3-vl:8b
```

## **Dependencies (`Cargo.toml`)**

The project relies on the following key libraries:

- `tokio` – Asynchronous runtime.
- `reqwest` – HTTP client with JSON support.
- `serde` & `serde_json` – Data serialization and structured parsing.
- `base64` – Image payload encoding.

## **How to Run the Application**

To execute the object detection pipeline, run:

```bash
cargo run
```

## **How to Run Tests**

To run integration tests verifying that detection payloads and bounding box responses serialize/deserialze properly:

```bash
cargo test
```

## ⚠️ Model Limitations & Notice on Vision Object Detection

During practical testing with local VLM models (such as `qwen3-vl:8b` and `qwen2.5-vl:7b` via Ollama), we observed significant limitations regarding precise spatial localization and bounding box regression:
- **Quantization & Compression**: Smaller quantized/compressed weights (GGUF formats typical for local execution) severely degrade the spatial and geometric token representations required for pixel-accurate coordinates.
- **General vs. Specialized Tasks**: General-purpose Vision-Language Models excel at captioning, OCR, and visual QA, but struggle with strict object detection without extensive task-specific fine-tuning. 

For strict, production-grade object detection, specialized architectures (such as YOLO via ONNX/C++ runtimes) are recommended instead of general VLMs.
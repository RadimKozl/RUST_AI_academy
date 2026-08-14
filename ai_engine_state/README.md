ollama run gemma3n:e4b

cargo run --release

curl -X POST http://127.0.0.1:3000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Vysvětli v jedné větě, co je to Rust Borrow Checker.",
    "system_prompt": "Jsi zkušený Rust mentor."
  }'

  curl.exe -X POST http://127.0.0.1:3000/api/v1/chat `
  -H "Content-Type: application/json" `
  -d '{\"prompt\": \"Vysvetli v jedne vete, co je to Rust Borrow Checker.\", \"system_prompt\": \"Jsi skuseny Rust mentor.\"}'
use generate_summare_dataset::{call_generate, GenerateRequest, NewsItem};
use rand::prelude::IndexedRandom;
use rand::rng;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = reqwest::Client::new();
    let generate_url = "http://localhost:11434/api/generate";
    let model_name = "granite4.2:3b";

    let topics = vec![
        "sports",
        "science and technology",
        "medicine",
        "transportation",
        "Czech National Bank",
        "weather",
        "politics",
        "economy",
    ];

    // --- 1. DEFINITION OF RANGES AND REPETITION COUNTS ---
    let min_chars = 100;
    let max_chars = 5000;
    let num_iterations = 3; // Here you set how many items you want to generate

    let mut rng = rng();
    let mut successful_items = 0;

    // Initialization of a JSON array in a file
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("output.json")?;
    writeln!(file, "[")?;

    // --- 2. REPETITION LOOP (BATCH GENERATION) ---
    for i in 0..num_iterations {
        let selected_topic = topics.choose(&mut rng).unwrap();
        println!("\n[{}/{}] Selected Topic: {}", i + 1, num_iterations, selected_topic);

        let generate_prompt = format!(
            "Generate a random news message in English about {}. The message must have a minimum of {} characters and a maximum of {} characters.",
            selected_topic, min_chars, max_chars
        );

        let gen_payload = GenerateRequest {
            model: model_name.to_string(),
            prompt: generate_prompt,
            stream: false,
        };

        println!("Generating report...");
        let gen_res = match call_generate(&client, generate_url, &gen_payload).await {
            Ok(res) => res.response,
            Err(e) => {
                eprintln!("Error generating message: {}", e);
                continue;
            }
        };

        let summary_prompt = format!("Summarize this text in English: {}", gen_res);
        let sum_payload = GenerateRequest {
            model: model_name.to_string(),
            prompt: summary_prompt,
            stream: false,
        };

        println!("Generating a summary...");
        let sum_res = match call_generate(&client, generate_url, &sum_payload).await {
            Ok(res) => res.response,
            Err(e) => {
                eprintln!("Error during summarization: {}", e);
                continue;
            }
        };

        let item = NewsItem {
            id: Uuid::new_v4().to_string(),
            message: gen_res,
            summarization: sum_res,
        };

        // Adding a comma before every item except the first
        if successful_items > 0 {
            writeln!(file, ",")?;
        }

        let json_data = serde_json::to_string_pretty(&item)?;
        write!(file, "{}", json_data)?;
        successful_items += 1;
        println!("Item saved.");
    }

    // Close the JSON field
    writeln!(file, "\n]")?;
    println!("\nDone! {} items generated and saved to output.json.", successful_items);

    Ok(())
}
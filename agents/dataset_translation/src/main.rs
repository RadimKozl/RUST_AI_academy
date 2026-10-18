use dataset_translation::{call_generate, GenerateRequest, TranslatedNewsItem};
use serde::Deserialize;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use uuid::Uuid;

// Auxiliary structure for mapping the input format from input.json
#[derive(Deserialize, Debug)]
struct InputRecord {
    #[allow(dead_code)]
    id: Option<String>,
    message: String,
    #[allow(dead_code)]
    summarization: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = reqwest::Client::new();
    let generate_url = "http://localhost:11434/api/generate";
    
    // Model and language configuration
    let model_translate = "gemma3n:e4b"; 
    let summary_model = "llama3.2:1b";
    let language = "Czech";

    // 1. Loading and parsing the input.json file
    println!("Loading input file input.json...");
    let file_content = fs::read_to_string("input.json")?;
    let records: Vec<InputRecord> = serde_json::from_str(&file_content)?;

    println!("Number of records loaded from input.json: {}", records.len());

    // 2. Initialization of the output JSON file output.json
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("output.json")?;
    writeln!(file, "[")?;

    let mut successful_items = 0;

    // 3. Processing each record in a loop
    for (i, record) in records.iter().enumerate() {
        println!("\nProcessing {}/{}", i + 1, records.len());

        let source_text = &record.message;

        // Step 1: Translation with an instruction to suppress internal thinking
        let translate_prompt = format!(
            "Translate the following text strictly to {}. Return ONLY the translation without any conversational filler or internal thinking:\n\n{}", 
            language, source_text
        );

        let translate_payload = GenerateRequest {
            model: model_translate.to_string(),
            prompt: translate_prompt,
            stream: false,
        };

        println!("Translating using model {}...", model_translate);
        let translate_res = match call_generate(&client, generate_url, &translate_payload).await {
            Ok(res) => res.response,
            Err(e) => {
                eprintln!("Error translating item {}: {}", i + 1, e);
                continue;
            }
        };

        // Step 2: Summarization with an instruction to suppress thinking
        let summary_prompt = format!(
            "Summarize this text strictly in {}. Return ONLY the summary without any conversational filler or internal thinking:\n\n{}", 
            language, translate_res
        );

        let summary_payload = GenerateRequest {
            model: summary_model.to_string(),
            prompt: summary_prompt,
            stream: false,
        };

        println!("Generating a summary using the {} model...", summary_model);
        let summary_res = match call_generate(&client, generate_url, &summary_payload).await {
            Ok(res) => res.response,
            Err(e) => {
                eprintln!("Error summarizing item {}: {}", i + 1, e);
                continue;
            }
        };

        // Creating the output structure
        let item = TranslatedNewsItem {
            id: Uuid::new_v4().to_string(),
            origin_text: translate_res,
            summary_text: summary_res,
        };

        // Adding a comma before every item except the first
        if successful_items > 0 {
            writeln!(file, ",")?;
        }

        let json_data = serde_json::to_string_pretty(&item)?;
        write!(file, "{}", json_data)?;
        successful_items += 1;
        println!("Item successfully translated, summarized, and saved.");
    }

    // Closing the JSON array in the output file
    writeln!(file, "\n]")?;
    println!("\nDone! {} items successfully processed and saved to output.json.", successful_items);

    Ok(())
}
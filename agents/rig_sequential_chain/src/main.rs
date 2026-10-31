use anyhow::Result;
use dotenvy::dotenv;
use rig_sequential_chain::{run_modular_chain, ChainStep};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG MODULAR N-STEP CHAIN (XAI & NEO4J READY) ===\n");

    let model_name = "qwen3.5:4b";
    let system_instruction = "You are a precise assistant. Respond concisely and provide only the requested name.";

    // Here we define any number of steps (N-parts)
    let steps = vec![
        ChainStep {
            name: "find_city".to_string(),
            prompt_template: "In which city was the person {input} born or did they live? Write only the name of the city.".to_string(),
        },
        ChainStep {
            name: "find_country".to_string(),
            prompt_template: "In which country is the city of {input} located? Write only the name of the country.".to_string(),
        },
        ChainStep {
            name: "find_continent".to_string(),
            prompt_template: "On which continent is the country {input} located? Write only the name of the continent.".to_string(),
        },
        ChainStep {
            name: "find_hemisphere".to_string(),
            prompt_template: "In which hemisphere (northern/southern) does the continent {input} predominantly lie? Write only the hemisphere.".to_string(),
        },
    ];

    let initial_person = "obama";
    println!("Launching a multi-step modular chain with {} steps for input: '{}'\n", steps.len(), initial_person);

    match run_modular_chain(model_name, initial_person, system_instruction, &steps).await {
        Ok(result) => {
            println!("=== FINAL RESULT ===");
            println!("{}\n", result.final_output);

            println!("=== XAI & NEO4J REASONING AUDIT TRAIL ===");
            for (i, record) in result.history.iter().enumerate() {
                println!("Step {}: [{}]", i + 1, record.step_name);
                println!("  Input (Context In):  {}", record.input_value);
                println!("  Output (Model Out):  {}\n", record.generated_output);
            }

            // Example of how history can be easily serialized (e.g., for storage in JSON / NoSQL / Neo4j)
            let json_memory = serde_json::to_string_pretty(&result.history).unwrap();
            println!("--- JSON representation for Neo4j/XAI ---");
            println!("{}", json_memory);
        }
        Err(e) => eprintln!("Error during string execution: {}", e),
    }

    Ok(())
}


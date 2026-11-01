use anyhow::Result;
use dotenvy::dotenv;
use rig_assign_chain::run_assign_chain; // Adjust the name according to your crate in Cargo.toml

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG RUNNABLE ASSIGN (DICTIONARY ENRICHMENT) ===\n");

    let model_name = "qwen3.5:4b";
    let question = "Write me a short haiku about programming in Python.";

    let system_instruction = "You are a creative poetic assistant.";
    let prompt_template = "Today's date: {timestamp} (processed by model: {processing_model}).\n\
                           The question is {question_length} characters long. Answer this question:\n\
                           QUESTION: {question}\n\
                           RESPONSE:";

    println!("🚀 Running Chain with RunnableAssign to add metadata...\n");

    match run_assign_chain(model_name, question, system_instruction, prompt_template).await {
        Ok(res) => {
            println!("--- Enriched Context (RunnableAssign Result) ---");
            println!("{:#?}\n", res.context);

            println!("--- Final Output ---");
            println!("{}\n", res.final_output);

            // XAI / Neo4j Audit Trail
            let json_audit = serde_json::to_string_pretty(&res.history).unwrap();
            println!("--- XAI Audit Trail (JSON) ---");
            println!("{}", json_audit);
        }
        Err(e) => eprintln!("Error during execution: {}", e),
    }

    Ok(())
}
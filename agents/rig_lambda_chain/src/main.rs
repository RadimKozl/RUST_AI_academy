use anyhow::Result;
use dotenvy::dotenv;
use rig_lambda_chain::run_lambda_chain;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG RUNNABLE LAMBDA CHAIN (CUSTOM TRANSFORMATION) ===\n");

    let model_name = "qwen3.5:4b";
    let tone = "Use a friendly and simple tone";
    let system_instruction = "You are a clear and concise science educator.";
    let prompt_template = "Explain what an atom is in one paragraph. {tone}.";

    println!("🚀 Running Chain with RunnableLambda simulation...\n");

    match run_lambda_chain(model_name, tone, system_instruction, prompt_template).await {
        Ok(res) => {
            println!("-------------------------------------------");
            println!("Type of final result: Struct (ProcessedSummaryResult)");
            println!("FINAL OUTPUT:\n{:#?}\n", res.final_data);

            // XAI / Neo4j audit trail
            let json_audit = serde_json::to_string_pretty(&res.history).unwrap();
            println!("--- XAI Audit Trail (JSON) ---");
            println!("{}", json_audit);
        }
        Err(e) => eprintln!("Error during execution: {}", e),
    }

    Ok(())
}

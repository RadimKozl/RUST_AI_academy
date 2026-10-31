use anyhow::Result;
use dotenvy::dotenv;
use rig_lcel_chain::run_company_namer_chain; // adjust according to the name of your crate

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG LCEL CHAIN EQUIVALENT ===\n");

    let name = "Lukas";
    let product = "colorful socks";
    
    // We will use an available Ollama model that supports output (e.g., qwen3.5:4b)
    let model_name = "qwen3.5:4b";

    println!("I am launching the chain for the specified parameters:");
    println!(" - Name: {}", name);
    println!(" - Product: {}\n", product);

    match run_company_namer_chain(model_name, name, product).await {
        Ok(result) => {
            println!("---- String Result ----");
            println!("{}", result);
        }
        Err(e) => eprintln!("Error while executing string: {}", e),
    }

    Ok(())
}
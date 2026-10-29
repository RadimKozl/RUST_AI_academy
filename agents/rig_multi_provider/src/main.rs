use anyhow::Result;
use dotenvy::dotenv;
use rig::providers::mistral;
use rig_multi_provider::{
    create_huggingface_agent, create_mistral_agent, create_ollama_agent, run_prompt, PromptTemplate,
};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    // Loading variables from the .env file
    dotenv().ok();
    println!("=== MULTI-PROVIDER AI TUTORIAL IN RUST (WRAPPER APPROACH) ===");

    let template = PromptTemplate::new("What is a good name for a company that makes {product}?");
    let formatted_prompt = template.format_product("colorful socks");

    println!("Generated prompt: {}\n", formatted_prompt);

    // ==========================================
    // 1. OLLAMA (Local model)
    // ==========================================
    println!("--- 1. Calling Ollama (locally) ---");
    let ollama_agent = create_ollama_agent("gemma3n:e4b", "You are a creative and helpful assistant.");

    match run_prompt(&ollama_agent, &formatted_prompt).await {
        Ok(res) => println!("Ollama response:\n{}\n", res),
        Err(e) => eprintln!("Ollama error (check if it is running on port 11434): {}\n", e),
    }

    // --- 2-SECOND DELAY ---
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // ==========================================
    // 2. MISTRAL AI (Cloud API)
    // ==========================================
    println!("--- 2. Calling Mistral AI ---");
    let mistral_api_key = env::var("MISTRAL_API_KEY").unwrap_or_default();

    if !mistral_api_key.is_empty() {
        let mistral_agent = create_mistral_agent(
            &mistral_api_key,
            mistral::MISTRAL_SMALL,
            "You are a poetic AI assistant.",
        );

        match run_prompt(&mistral_agent, &formatted_prompt).await {
            Ok(res) => println!("Mistral AI response:\n{}\n", res),
            Err(e) => eprintln!("Mistral AI Error: {}\n", e),
        }
    } else {
        println!("MISTRAL_API_KEY was not found in .env; skipping.\n");
    }

    // --- 2-SECOND DELAY ---
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // ==========================================
    // 3. HUGGING FACE (Endpoint API)
    // ==========================================
    println!("--- 3. Calling Hugging Face ---");
    let hf_token = env::var("HF_TOKEN").unwrap_or_default();

    if !hf_token.is_empty() {
        let hf_agent = create_huggingface_agent(
            &hf_token,
            "HuggingFaceH4/zephyr-7b-beta",
            "You are a helpful AI assistant specialized in branding.",
        );

        match run_prompt(&hf_agent, &formatted_prompt).await {
            Ok(res) => println!("Hugging Face response:\n{}\n", res),
            Err(e) => eprintln!("Hugging Face Error: {}\n", e),
        }
    } else {
        println!("HF_TOKEN not found in .env, skipping.\n");
    }

    Ok(())
}
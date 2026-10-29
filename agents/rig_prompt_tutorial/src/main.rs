use anyhow::Result;
use rig::providers::ollama::Ollama;
use rig_prompt_tutorial::{create_styled_agent, run_prompt, PromptTemplate};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== 1. Simple Prompt & Templates (LangChain PromptTemplate -> Rust) ===");

    let template = PromptTemplate::new("What is a good name for a company that makes {product}?");
    let formatted_prompt = template.format_product("colorful socks");

    println!("Generated prompt: {}\n", formatted_prompt);

    println!("=== 2. Chat Template and Agent (LangChain ChatPromptTemplate -> Rig Agent) ===");

    // Connecting to a local Ollama server
    let client = Ollama::new();

    // Equivalent to SystemMessage in LangChain (using .preamble in Rig)
    let system_prompt = "You are a helpful assistant with output as Shakespeare.";
    let model_name = "gemma3n:e4b"; // Or any other model in your Ollama

    let agent = create_styled_agent(&client, model_name, system_prompt);

    println!("Sending a query to the agent with the role 'Shakespeare'...");
    match run_prompt(&agent, &formatted_prompt).await {
        Ok(response) => {
            println!("\n--- Agent's response ---");
            println!("{}\n", response);
        }
        Err(e) => {
            eprintln!("Error communicating with Ollama: {}", e);
            eprintln!("Make sure that Ollama is running at http://localhost:11434 and the model '{}' is downloaded.", model_name);
        }
    }

    Ok(())
}
use anyhow::Result;
use dotenvy::dotenv;
use rig_tool_calling::{create_agent_with_tools, run_prompt};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST AI AGENT WITH TOOLS (RIG FRAMEWORK) ===\n");

    let agent = create_agent_with_tools(
        "gemma4:e4b", 
        "You are a helpful financial assistant. You MUST use the provided tool to get stock prices when asked.",
    );

    // ==========================================
    // 1. Test of the first instrument (Stocks)
    // ==========================================
    let stock_prompt = "What is the current stock price for MSFT?";
    println!("--- Test 1: Action Tool ---");
    println!("User query: {}\n", stock_prompt);

    match run_prompt(&agent, stock_prompt).await {
        Ok(res) => println!("Agent's final response:\n{}\n", res),
        Err(e) => eprintln!("Error during agent execution: {}\n", e),
    }

    // ==========================================
    // 2. Testing the second tool (Wikipedia)
    // ==========================================
    let wiki_prompt = "Who is the founder of Microsoft and in what year was the company founded?";
    println!("--- Test 2: Wikipedia Tool ---");
    println!("User query: {}\n", wiki_prompt);

    match run_prompt(&agent, wiki_prompt).await {
        Ok(res) => println!("Agent's final response:\n{}\n", res),
        Err(e) => eprintln!("Error during agent execution: {}\n", e),
    }

    Ok(())
}
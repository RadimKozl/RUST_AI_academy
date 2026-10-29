use anyhow::Result;
use rig::agent::Agent;
use rig::providers::ollama;
use rig::tool::{Tool, ToolContext};
use serde::Deserialize;
use schemars::JsonSchema;

// =======================================================
// 1. Stock Price Tool
// =======================================================

#[derive(Deserialize, JsonSchema)]
pub struct StockArgs {
    /// The stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
    pub ticker: String,
}

#[derive(Debug, thiserror::Error)]
#[error("Stock tool error")]
pub struct StockError;

pub struct StockPriceTool;

impl Tool for StockPriceTool {
    const NAME: &str = "get_stock_price";
    type Error = StockError;
    type Args = StockArgs;
    type Output = String;

    fn description(&self) -> String {
        "Retrieve the current stock price for a given ticker symbol.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(StockArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let price = match args.ticker.to_uppercase().as_str() {
            "MSFT" => "$513.57",
            "AAPL" => "$225.00",
            _ => "$100.00",
        };
        Ok(format!("The current price of {} is {}", args.ticker, price))
    }
}

// =======================================================
// 2. Wikipedia Search Tool (s reálným HTTP dotazem)
// =======================================================

#[derive(Deserialize, JsonSchema)]
pub struct WikiArgs {
    /// The search query or topic to look up on Wikipedia (e.g., 'Microsoft', 'Rust programming language')
    pub query: String,
}

#[derive(Debug, thiserror::Error)]
#[error("Wikipedia tool error: {0}")]
pub struct WikiError(#[from] reqwest::Error);

pub struct WikiSearchTool;

impl Tool for WikiSearchTool {
    const NAME: &str = "search_wiki";
    type Error = WikiError;
    type Args = WikiArgs;
    type Output = String;

    fn description(&self) -> String {
        "Search Wikipedia for information about a specific topic, company, or person using the MediaWiki API.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(WikiArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let url = format!(
            "https://cs.wikipedia.org/w/api.php?action=query&list=search&srsearch={}&format=json",
            urlencoding::encode(&args.query)
        );

        let client = reqwest::Client::builder()
            .user_agent("RigRustAgent/1.0 (educational project)")
            .build()?;

        let resp = client.get(&url).send().await?;
        let json: serde_json::Value = resp.json().await?;

        if let Some(search_results) = json.pointer("/query/search").and_then(|v| v.as_array()) {
            if let Some(first_result) = search_results.first() {
                let title = first_result.get("title").and_then(|v| v.as_str()).unwrap_or("Unknown");
                let snippet = first_result.get("snippet").and_then(|v| v.as_str()).unwrap_or("No snippet available.");
                
                let clean_snippet = snippet
                    .replace("<span class=\"searchmatch\">", "")
                    .replace("</span>", "")
                    .replace("&quot;", "\"")
                    .replace("&#039;", "'");

                return Ok(format!("Title: {}\nSummary: {}...", title, clean_snippet));
            }
        }

        Ok("No relevant Wikipedia article found for the given query.".to_string())
    }
}

// ===============================================
// 3. Factory methods
// ===============================================

pub fn create_agent_with_tools(model_name: &str, system_instruction: &str) -> Agent {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);

    rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .tool(StockPriceTool)
        .build()
}

pub fn create_agent_with_all_tools(model_name: &str, system_instruction: &str) -> Agent {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);

    rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .tool(StockPriceTool)
        .tool(WikiSearchTool)
        .build()
}

pub async fn run_prompt(agent: &Agent, prompt: &str) -> Result<String> {
    let response = agent.prompt(prompt).max_turns(5).await?;
    Ok(response.output().to_string())
}
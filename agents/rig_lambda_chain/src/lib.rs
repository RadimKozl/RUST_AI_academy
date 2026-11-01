use anyhow::Result;
use rig::providers::ollama;
use serde::{Deserialize, Serialize};

/// Single-step record for XAI / Neo4j memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaStepRecord {
    pub step_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// Structured result after transformation using RunnableLambda
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedSummaryResult {
    pub summary: String,
    pub full_length: usize,
    pub source: String,
}

/// Overall result of the chain, containing data and audit history
#[derive(Debug, Serialize, Deserialize)]
pub struct LambdaChainResult {
    pub final_data: ProcessedSummaryResult,
    pub history: Vec<LambdaStepRecord>,
}

/// Custom function (equivalent to the processor's RunnableLambda)
pub fn process_and_summarize(full_text: &str) -> ProcessedSummaryResult {
    let summary = if full_text.chars().count() > 20 {
        let truncated: String = full_text.chars().take(20).collect();
        format!("{}...", truncated)
    } else {
        full_text.to_string()
    };

    ProcessedSummaryResult {
        summary,
        full_length: full_text.chars().count(),
        source: "AI_Model_Ollama_Qwen".to_string(),
    }
}

/// A universal runner that simulates the behavior of a RunnableLambda within a chain
pub async fn run_lambda_chain(
    model_name: &str,
    tone: &str,
    system_instruction: &str,
    prompt_template: &str,
) -> Result<LambdaChainResult> {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);

    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .build();

    let prompt = prompt_template.replace("{tone}", tone);
    
    // Step 1: Launching the LLM agent
    let response = agent.prompt(&prompt).await?;
    let raw_text = response.output().trim().to_string();

    let mut history = vec![LambdaStepRecord {
        step_name: "llm_generation".to_string(),
        input_value: format!("Tone: {}", tone),
        generated_output: raw_text.clone(),
    }];

    // Step 2: Transformation using a RunnableLambda equivalent
    let processed_data = process_and_summarize(&raw_text);

    history.push(LambdaStepRecord {
        step_name: "runnable_lambda_transform".to_string(),
        input_value: raw_text.clone(),
        generated_output: format!("{:#?}", processed_data),
    });

    Ok(LambdaChainResult {
        final_data: processed_data,
        history,
    })
}
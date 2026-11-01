use anyhow::Result;
use chrono::Local;
use rig::providers::ollama;
use serde::{Deserialize, Serialize};

/// Single-step record for XAI / Neo4j memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignStepRecord {
    pub step_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// Enriched context representing the output of the RunnableAssign step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichedContext {
    pub question: String,
    pub question_length: usize,
    pub processing_model: String,
    pub timestamp: String,
}

/// The overall result of the chain, containing the final answer as well as the audit history
#[derive(Debug, Serialize, Deserialize)]
pub struct AssignChainResult {
    pub context: EnrichedContext,
    pub final_output: String,
    pub history: Vec<AssignStepRecord>,
}

/// Function performing the RunnableAssign operation (enriching the input with metadata)
pub fn assign_metadata(question: &str, model_name: &str) -> EnrichedContext {
    EnrichedContext {
        question: question.to_string(),
        question_length: question.chars().count(),
        processing_model: model_name.to_string(),
        timestamp: Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
    }
}

/// Universal runner for a RunnableAssign RAG / Enrichment chain
pub async fn run_assign_chain(
    model_name: &str,
    question: &str,
    system_instruction: &str,
    prompt_template: &str,
) -> Result<AssignChainResult> {
    let client = ollama::Ollama::new();
    
    // We can set max_tokens or other parameters if the model is limited
    let model = client.completion(model_name);

    // Step 1: RunnableAssign (Enriching the context with metadata)
    let enriched_ctx = assign_metadata(question, model_name);

    let mut history = vec![AssignStepRecord {
        step_name: "runnable_assign_enrichment".to_string(),
        input_value: question.to_string(),
        generated_output: serde_json::to_string(&enriched_ctx)?,
    }];

    // Step 2: Preparing the agent
    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .build();

    let final_prompt = prompt_template
        .replace("{timestamp}", &enriched_ctx.timestamp)
        .replace("{processing_model}", &enriched_ctx.processing_model)
        .replace("{question_length}", &enriched_ctx.question_length.to_string())
        .replace("{question}", &enriched_ctx.question);

    // Step 3: Generating the final response
    // If the agent in Rig supports max_tokens via the builder, it can be specified there,
    // It might also help if the prompt doesn't require an overly long text.
    let response = agent.prompt(&final_prompt).await?;
    let final_output = response.output().trim().to_string();

    history.push(AssignStepRecord {
        step_name: "final_llm_execution".to_string(),
        input_value: final_prompt,
        generated_output: final_output.clone(),
    });

    Ok(AssignChainResult {
        context: enriched_ctx,
        final_output,
        history,
    })
}
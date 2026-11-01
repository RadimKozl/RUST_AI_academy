use anyhow::Result;
use rig::providers::ollama;
use serde::{Deserialize, Serialize};

/// Single-step record for XAI / Neo4j memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagStepRecord {
    pub step_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// Result of a RAG chain with context retention
#[derive(Debug, Serialize, Deserialize)]
pub struct RagChainResult {
    pub original_question: String,
    pub generated_context: String,
    pub final_output: String,
    pub history: Vec<RagStepRecord>,
}

/// Universal runner implementing the RunnablePassthrough pattern (RAG pattern)
pub async fn run_passthrough_rag_chain(
    model_name: &str,
    question: &str,
    context_system: &str,
    context_prompt_tmpl: &str,
    final_system: &str,
    final_prompt_tmpl: &str,
) -> Result<RagChainResult> {
    let client = ollama::Ollama::new();

    // --- Step 1: Context Branch (Retriever Simulation) ---
    let context_model = client.completion(model_name);
    let context_agent = rig::agent::AgentBuilder::new(context_model)
        .preamble(context_system)
        .build();

    let context_prompt = context_prompt_tmpl.replace("{question}", question);
    let context_response = context_agent.prompt(&context_prompt).await?;
    let generated_context = context_response.output().trim().to_string();

    let mut history = vec![RagStepRecord {
        step_name: "context_retriever_simulation".to_string(),
        input_value: question.to_string(),
        generated_output: generated_context.clone(),
    }];

    // --- Step 2: RunnablePassthrough (preserving the original input) ---
    let original_question = question.to_string();

    // --- Step 3: Final RAG generation combining the context and the original question ---
    let final_model = client.completion(model_name);
    let final_agent = rig::agent::AgentBuilder::new(final_model)
        .preamble(final_system)
        .build();

    let final_prompt = final_prompt_tmpl
        .replace("{context}", &generated_context)
        .replace("{question}", &original_question);

    let final_response = final_agent.prompt(&final_prompt).await?;
    let final_output = final_response.output().trim().to_string();

    history.push(RagStepRecord {
        step_name: "final_rag_generation".to_string(),
        input_value: format!("Context: {} | Question: {}", generated_context, original_question),
        generated_output: final_output.clone(),
    });

    Ok(RagChainResult {
        original_question,
        generated_context,
        final_output,
        history,
    })
}
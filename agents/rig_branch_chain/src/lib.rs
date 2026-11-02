use anyhow::Result;
use rig::providers::ollama;
use serde::{Deserialize, Serialize};

/// Single-step record for XAI / Neo4j memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchStepRecord {
    pub step_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// Result of executing a conditional branch (RunnableBranch)
#[derive(Debug, Serialize, Deserialize)]
pub struct BranchChainResult {
    pub matched_branch: String,
    pub final_output: String,
    pub history: Vec<BranchStepRecord>,
}

/// Universal runner simulating the behavior of RunnableBranch
pub async fn run_branch_chain(
    model_name: &str,
    topic: &str,
    tone: &str,
) -> Result<BranchChainResult> {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);

    let tone_lower = tone.to_lowercase();

    // 1. Evaluation of conditions (RunnableBranch logical decision-making)
    let (matched_branch, system_instruction, prompt_template) = if tone_lower.contains("formal") {
        (
            "formal_branch",
            "You are an academic expert. Explain topics in an extremely academic, formal, and lengthy style.",
            "Explain the topic {topic} in an extremely academic, formal, and lengthy style.",
        )
    } else if tone_lower.contains("joke") || tone_lower.contains("funny") {
        (
            "funny_branch",
            "You are a friendly and humorous assistant.",
            "Explain the topic {topic} in a very informal, friendly, and brief style with a touch of humor.",
        )
    } else {
        // ELSE (Default) branch
        (
            "default_casual_branch",
            "You are a helpful general assistant.",
            "Explain the topic {topic} in a very informal, friendly, and brief style.",
        )
    };

    let mut history = vec![BranchStepRecord {
        step_name: "branch_evaluator".to_string(),
        input_value: format!("Topic: {} | Tone: {}", topic, tone),
        generated_output: matched_branch.to_string(),
    }];

    // 2. Constructing the agent with the selected system instruction
    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .build();

    let final_prompt = prompt_template.replace("{topic}", topic);

    // 3. Generating the final response
    let response = agent.prompt(&final_prompt).await?;
    let final_output = response.output().trim().to_string();

    history.push(BranchStepRecord {
        step_name: matched_branch.to_string(),
        input_value: final_prompt,
        generated_output: final_output.clone(),
    });

    Ok(BranchChainResult {
        matched_branch: matched_branch.to_string(),
        final_output,
        history,
    })
}
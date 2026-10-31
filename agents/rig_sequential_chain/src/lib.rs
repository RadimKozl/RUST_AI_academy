use anyhow::Result;
use rig::providers::ollama;
use serde::{Deserialize, Serialize};

/// Record of a single step performed (ideal for XAI logging or Neo4j reasoning memory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStepRecord {
    pub step_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// The result of the modular chain, containing the final answer and the full history
#[derive(Debug, Serialize, Deserialize)]
pub struct ModularChainResult {
    pub final_output: String,
    pub history: Vec<ChainStepRecord>,
}

/// Definition of a single configurable step in a chain
#[derive(Clone)]
pub struct ChainStep {
    pub name: String,
    pub prompt_template: String, // Template containing the placeholder "{input}"
}

/// Universal modular runner for any number of steps (N-sections)
pub async fn run_modular_chain(
    model_name: &str,
    initial_input: &str,
    system_instruction: &str,
    steps: &[ChainStep],
) -> Result<ModularChainResult> {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);

    // Agent initialization with a provided system instruction
    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .build();

    let mut current_input = initial_input.to_string();
    let mut history = Vec::with_capacity(steps.len());

    // Dynamically iterate through any number of defined steps
    for step in steps {
        // Substitute the current intermediate result into the prompt template
        let prompt = step.prompt_template.replace("{input}", &current_input);
        
        let response = agent.prompt(&prompt).await?;
        let output = response.output().trim().to_string();

        // Save a detailed step record for XAI / Neo4j
        history.push(ChainStepRecord {
            step_name: step.name.clone(),
            input_value: current_input.clone(),
            generated_output: output.clone(),
        });

        // The output of the current step becomes the input for the next step
        current_input = output;
    }

    Ok(ModularChainResult {
        final_output: current_input,
        history,
    })
}
use anyhow::Result;
use rig::agent::Agent;
use rig::providers::{huggingface, mistral, ollama};

// ==========================================
// 1. Prompt templates
// ==========================================
pub struct PromptTemplate {
    template: String,
}

impl PromptTemplate {
    pub fn new(template: &str) -> Self {
        Self {
            template: template.to_string(),
        }
    }

    pub fn format_product(&self, product: &str) -> String {
        self.template.replace("{product}", product)
    }

    pub fn format_name_and_product(&self, name: &str, product: &str) -> String {
        self.template
            .replace("{name}", name)
            .replace("{product}", product)
    }
}

// ==========================================
// 2. Factory Methods (Wrappers) for Agents
// ==========================================

/// Creates an agent for a local Ollama instance
pub fn create_ollama_agent(model_name: &str, system_instruction: &str) -> Agent {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);
    rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .build()
}

/// Creates an agent for cloud-based Mistral AI
pub fn create_mistral_agent(api_key: &str, model_name: &str, system_instruction: &str) -> Agent {
    let client = mistral::new(api_key);
    let model = client.completion(model_name);
    rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .build()
}

/// Creates an agent for a Hugging Face Endpoint
pub fn create_huggingface_agent(hf_token: &str, model_name: &str, system_instruction: &str) -> Agent {
    let client = huggingface::new(hf_token);
    let model = client.completion(model_name);
    rig::agent::AgentBuilder::new(model)
        .preamble(system_instruction)
        .build()
}

// ==========================================
// 3. Executing the prompt
// ==========================================
pub async fn run_prompt(agent: &Agent, prompt: &str) -> Result<String> {
    let response = agent.prompt(prompt).await?;
    Ok(response.output().to_string())
}
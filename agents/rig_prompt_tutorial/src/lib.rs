use anyhow::Result;
use rig::prelude::*;
use rig::providers::ollama::Ollama;
use rig::Agent;

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

pub fn create_styled_agent(
    client: &Ollama,
    model_name: &str,
    system_instruction: &str,
) -> Agent {
    AgentBuilder::new(client.completion(model_name))
        .preamble(system_instruction)
        .build()
}

pub async fn run_prompt(agent: &Agent, prompt: &str) -> Result<String> {
    let response = agent.prompt(prompt).await?;
    Ok(response.output())
}
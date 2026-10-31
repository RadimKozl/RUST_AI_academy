use anyhow::Result;
use rig::agent::Agent;
use rig::providers::ollama;

/// The function creates and executes a chain that simulates the behavior of LangChain's LCEL.
pub async fn run_company_namer_chain(
    model_name: &str,
    name: &str,
    product: &str,
) -> Result<String> {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);

    // Constructing the system prompt template and conversation context (ChatPromptTemplate)
    let system_prompt = "You are an AI assistant. You help users generate creative business names.";
    
    // Creating an Agent (equivalent to prompt | llm)
    let agent: Agent = rig::agent::AgentBuilder::new(model)
        .preamble(system_prompt)
        .build();

    // Constructing a user prompt with dynamic variables
    let formatted_prompt = format!(
        "Human: hi {}\nAI: hello\nHuman: What is a good name for a company that makes {}?",
        name, product
    );

    // Chain executor (invoke)
    let response = agent.prompt(&formatted_prompt).await?;
    
    Ok(response.output().to_string())
}
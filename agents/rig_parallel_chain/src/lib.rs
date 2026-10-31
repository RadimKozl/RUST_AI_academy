use anyhow::Result;
use rig::providers::ollama;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Record of the result of a single parallel branch (suitable for XAI / audit trail)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelBranchResult {
    pub branch_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// Definition of a single parallel branch (analogous to a key in RunnableParallel)
#[derive(Clone)]
pub struct ParallelBranch {
    pub name: String,
    pub system_instruction: String,
    pub prompt_template: String, // Template containing the placeholder "{topic}"
}

/// Universal parallel runner (equivalent to LangChain RunnableParallel)
pub async fn run_parallel_chain(
    model_name: &str,
    input_topic: &str,
    branches: &[ParallelBranch],
) -> Result<HashMap<String, String>> {
    
    // Create asynchronous tasks (futures) for each branch separately,
    // so that they can run truly in parallel.
    let mut futures = Vec::new();

    for branch in branches {
        let model_name_owned = model_name.to_string();
        let topic_owned = input_topic.to_string();
        let branch_name = branch.name.clone();
        let system_instr = branch.system_instruction.clone();
        let prompt_tmpl = branch.prompt_template.clone();

        let fut = tokio::spawn(async move {
            let model = ollama::Ollama::new().completion(&model_name_owned);
            let agent = rig::agent::AgentBuilder::new(model)
                .preamble(&system_instr)
                .build();

            let prompt = prompt_tmpl.replace("{topic}", &topic_owned);
            let response = agent.prompt(&prompt).await?;
            Ok::<(String, String), anyhow::Error>((branch_name, response.output().trim().to_string()))
        });

        futures.push(fut);
    }

    // We will wait for all parallel branches to complete simultaneously
    let results = futures::future::join_all(futures).await;

    let mut output_map = HashMap::new();
    for res in results {
        match res {
            Ok(Ok((name, output))) => {
                output_map.insert(name, output);
            }
            Ok(Err(e)) => return Err(e),
            Err(e) => return Err(anyhow::anyhow!("Task join error: {}", e)),
        }
    }

    Ok(output_map)
}
use anyhow::Result;
use dotenvy::dotenv;
use rig_parallel_chain::{run_parallel_chain, ParallelBranch}; 

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG PARALLEL CHAIN ​​(RUNNABLE PARALLEL EQUIVALENT) ===\n");

    let model_name = "qwen3.5:4b";
    let topic = "ice cream";

    // Definition of parallel branches (analogous to RunnableParallel(joke=chain1, poem=chain2))
    let branches = vec![
        ParallelBranch {
            name: "joke".to_string(),
            system_instruction: "You are a funny comedian. Tell short, witty jokes.".to_string(),
            prompt_template: "tell me a short joke about {topic}".to_string(),
        },
        ParallelBranch {
            name: "poem".to_string(),
            system_instruction: "You are a creative and lyrical poet.".to_string(),
            prompt_template: "write a 2-line poem about {topic}".to_string(),
        },
    ];

    println!("Launching parallel chain for topic: '{}'\n", topic);

    match run_parallel_chain(model_name, topic, &branches).await {
        Ok(results) => {
            if let Some(joke) = results.get("joke") {
                println!("---- Joke ----\n{}\n", joke);
            }
            if let Some(poem) = results.get("poem") {
                println!("---- Poem ----\n{}\n", poem);
            }

            // Serialization for XAI / Neo4j / audits
            let json_output = serde_json::to_string_pretty(&results).unwrap();
            println!("--- JSON Map Output ---");
            println!("{}", json_output);
        }
        Err(e) => eprintln!("Error during parallel execution: {}", e),
    }

    Ok(())
}
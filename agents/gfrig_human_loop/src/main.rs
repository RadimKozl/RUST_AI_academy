use anyhow::Result;
use dotenvy::dotenv;
use graph_flow::{GraphBuilder, Task}; // Imported Task method/trait for .run()
use std::sync::Arc;

use gfrig_human_loop::{HumanTask, JokeTask, StoryTask};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST HUMAN-IN-THE-LOOP (HITL) PIPELINE ===\n");

    let ollama_host = "http://localhost:11434";
    let chat_model = "glm4:9b";

    let theme = "universe";
    let initial_context = graph_flow::Context::new();
    let _ = initial_context.set("theme", theme.to_string());
    let _ = initial_context.set("iteration", 0);

    loop {
        let builder = GraphBuilder::new("hitl_graph");
        
        let builder = builder.add_task(Arc::new(JokeTask::new("joke", ollama_host, chat_model)));
        let builder = builder.add_task(Arc::new(HumanTask::new("human")));
        let builder = builder.add_task(Arc::new(StoryTask::new("story", chat_model)));

        let builder = builder.add_edge("joke", "human");
        let builder = builder.add_edge("human", "story");

        let _graph = builder.build()?;
        
        println!("🤖 Generating a joke...");
        let joke_task = JokeTask::new("joke", ollama_host, chat_model);
        joke_task.run(initial_context.clone()).await?;

        let human_task = HumanTask::new("human");
        human_task.run(initial_context.clone()).await?;

        let approved = initial_context.get::<bool>("approved").unwrap_or(false);
        if approved {
            println!("📖 Generating a story...");
            let story_task = StoryTask::new("story", chat_model);
            story_task.run(initial_context.clone()).await?;
            break;
        } else {
            println!("🔄 User did not approve the joke, repeating the cycle...");
        }
    }

    println!("\n✅ Workflow successfully completed!");
    Ok(())
}
use async_trait::async_trait;
use graph_flow::{
    Context as GraphContext, Task, TaskResult,
};
use rig::providers::ollama;
use serde::{Deserialize, Serialize};

/// Shared state for a Human-in-the-Loop workflow
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HitlState {
    pub theme: String,
    pub joke_text: String,
    pub story_text: String,
    pub approved: bool,
    pub iteration: i32,
}

// ===================================================
// 1. JokeNode: Generates a joke using Rig & Ollama
// ===================================================
pub struct JokeTask {
    id_str: String,
    _ollama_host: String, // The underscore suppresses the warning about an unused field
    model_name: String,
}

impl JokeTask {
    pub fn new(id: impl Into<String>, ollama_host: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            id_str: id.into(),
            _ollama_host: ollama_host.into(),
            model_name: model_name.into(),
        }
    }
}

#[async_trait]
impl Task for JokeTask {
    fn id(&self) -> &str {
        &self.id_str
    }

    async fn run(&self, context: GraphContext) -> graph_flow::Result<TaskResult> {
        let theme = context.get::<String>("theme").unwrap_or_else(|| "universe".to_string());
        let iteration = context.get::<i32>("iteration").unwrap_or(0) + 1;
        
        println!("\n🤖 -> [JokeNode] Generating a joke on the topic: '{}' (Attempt no. {})", theme, iteration);

        let client = ollama::Ollama::new();
        let model = client.completion(&self.model_name);
        let agent = rig::agent::AgentBuilder::new(model)
            .preamble("You are a hilarious person! Generate a short, clever joke suitable for telling a story.")
            .build();

        let prompt = format!("Generate a joke on theme: {}", theme);
        let response = agent.prompt(&prompt).await.map_err(|e| graph_flow::GraphError::Other(anyhow::anyhow!(e.to_string())))?;
        let joke_text = response.output().trim().to_string();

        println!("--------------------------------");
        println!("{}", joke_text);
        println!("--------------------------------");

        // Handling the Result from context.set using let _ =
        let _ = context.set("joke_text", joke_text.clone());
        let _ = context.set("iteration", iteration);

        Ok(TaskResult::move_to_next())
    }
}

// =====================================================
// 2. HumanNode: Interactive Node (Human-in-the-Loop)
// =====================================================
pub struct HumanTask {
    id_str: String,
}

impl HumanTask {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id_str: id.into() }
    }
}

#[async_trait]
impl Task for HumanTask {
    fn id(&self) -> &str {
        &self.id_str
    }

    async fn run(&self, context: GraphContext) -> graph_flow::Result<TaskResult> {
        let joke = context.get::<String>("joke_text").unwrap_or_default();
        
        println!("\n👤 --- HUMAN INTERACTION (HITL) ---");
        println!("Do you like this joke, or do you want to generate a new one?");
        println!("--------------------------------");
        println!("{}", joke);
        println!("--------------------------------");

        use std::io::{self, Write};
        print!("Enter 'yes' to continue to the story, or anything else to rewrite the joke: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let user_input = input.trim().to_lowercase();

        if user_input == "yes" {
            println!("✅ User approved the joke! Heading to StoryNode.");
            let _ = context.set("approved", true);
        } else {
            println!("🔄 The user rejected the joke. Returning to JokeNode (Cycle).");
            let _ = context.set("approved", false);
        }

        Ok(TaskResult::move_to_next())
    }
}

// ===================================================
// 3. StoryNode: Generates a story based on a joke
// ===================================================
pub struct StoryTask {
    id_str: String,
    model_name: String,
}

impl StoryTask {
    pub fn new(id: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            id_str: id.into(),
            model_name: model_name.into(),
        }
    }
}

#[async_trait]
impl Task for StoryTask {
    fn id(&self) -> &str {
        &self.id_str
    }

    async fn run(&self, context: GraphContext) -> graph_flow::Result<TaskResult> {
        let joke_text = context.get::<String>("joke_text").unwrap_or_default();
        
        println!("\n📖 -> [StoryNode] Generating a story leading up to the joke's punchline...");

        let client = ollama::Ollama::new();
        let model = client.completion(&self.model_name);
        let agent = rig::agent::AgentBuilder::new(model)
            .preamble("You are a master storyteller! Write a short story that leads up to the punchline provided by the joke.")
            .build();

        let prompt = format!("Generate a story for this joke: {}", joke_text);
        let response = agent.prompt(&prompt).await.map_err(|e| graph_flow::GraphError::Other(anyhow::anyhow!(e.to_string())))?;
        let story_text = response.output().trim().to_string();

        let _ = context.set("story_text", story_text.clone());

        println!("\n--------- FINAL RESULT ---------");
        println!("Joke:\n{}\n", joke_text);
        println!("Story:\n{}\n", story_text);

        Ok(TaskResult::move_to_next())
    }
}
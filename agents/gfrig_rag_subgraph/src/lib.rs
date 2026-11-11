use serde::{Deserialize, Serialize};

/// Overall state of an agent traversing the main graph and the subgraph
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentState {
    pub task: String,
    pub question: String,
    pub documents: Vec<String>,
    pub context: String,
    pub answer: String,
    pub critic_score: f32,
    pub critic_reason: String, // XAI output from the subgraph (critic's rationale)
    pub iteration: usize,
    pub search_result: String,
    pub summarization: String,
}

impl AgentState {
    pub fn new(task: &str) -> Self {
        Self {
            task: task.to_string(),
            question: String::new(),
            documents: vec![],
            context: String::new(),
            answer: String::new(),
            critic_score: 0.0,
            critic_reason: String::new(),
            iteration: 0,
            search_result: String::new(),
            summarization: String::new(),
        }
    }
}
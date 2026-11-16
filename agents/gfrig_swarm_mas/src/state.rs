// src/state.rs
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentRole {
    User,
    Researcher,
    Coder,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwarmMessage {
    pub sender: AgentRole,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwarmMasState {
    pub original_query: String,
    pub messages: Vec<SwarmMessage>,
    pub current_agent: AgentRole,
    pub delegation_count: usize,
    pub final_answer: Option<String>,
}

impl SwarmMasState {
    pub fn new(query: &str) -> Self {
        let mut state = Self {
            original_query: query.to_string(),
            messages: Vec::new(),
            current_agent: AgentRole::Researcher,
            delegation_count: 0,
            final_answer: None,
        };
        state.push_message(AgentRole::User, query);
        state
    }

    pub fn push_message(&mut self, sender: AgentRole, content: impl Into<String>) {
        self.messages.push(SwarmMessage {
            sender,
            content: content.into(),
        });
    }

    pub fn format_history(&self) -> String {
        self.messages
            .iter()
            .map(|m| {
                let role = match m.sender {
                    AgentRole::User => "USER",
                    AgentRole::Researcher => "RESEARCHER",
                    AgentRole::Coder => "CODER",
                };
                format!("[{role}]: {}", m.content)
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}
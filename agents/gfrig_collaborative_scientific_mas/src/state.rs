use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentRole {
    User,
    Researcher,
    Coder,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MasMessage {
    pub sender: AgentRole,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceStep {
    pub step: usize,
    pub node: String,
    pub action: String,
    pub details: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScientificMasState {
    pub original_query: String,
    pub messages: Vec<MasMessage>,
    pub loop_count: usize,
    pub last_tool_signature: Option<String>,
    pub generated_chart_path: Option<String>,
    pub generated_chart_b64: Option<String>,
    pub trace: Vec<TraceStep>,
    pub final_answer: Option<String>,
}

impl ScientificMasState {
    pub fn new(query: &str) -> Self {
        let mut state = Self {
            original_query: query.to_string(),
            ..Default::default()
        };
        state.push_message(AgentRole::User, query);
        state
    }

    pub fn push_message(&mut self, sender: AgentRole, content: impl Into<String>) {
        self.messages.push(MasMessage {
            sender,
            content: content.into(),
        });
    }

    pub fn log_trace(&mut self, node: &str, action: &str, details: impl Into<String>) {
        let step = self.trace.len() + 1;
        self.trace.push(TraceStep {
            step,
            node: node.to_string(),
            action: action.to_string(),
            details: details.into(),
        });
    }

    /// Formats message history for the LLM agent's context
    pub fn format_history_for_prompt(&self) -> String {
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

/// Equivalent to `extract_base64_from_output` from graph_2.py
pub fn extract_base64_from_output(output: &str) -> Option<String> {
    let start_marker = "---B64-START---";
    let end_marker = "---B64-END---";
    let start_idx = output.find(start_marker)? + start_marker.len();
    let end_idx = output[start_idx..].find(end_marker)? + start_idx;
    let b64 = output[start_idx..end_idx].trim();
    if b64.is_empty() {
        None
    } else {
        Some(b64.to_string())
    }
}
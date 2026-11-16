use serde::{Deserialize, Serialize};

/// Role identifier in a multi-agent system
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentRole {
    User,
    Supervisor,
    Researcher,
    Coder,
}

/// One message in the conversation history
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MasMessage {
    pub sender: AgentRole,
    pub content: String,
}

/// Structured Supervisor decision (equivalent to `supervisor_decision` in Python)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupervisorDecision {
    Delegate {
        to_agent: AgentRole,
        task: String,
    },
    Finish {
        final_answer: String,
    },
}

/// Audit trail of a single graph step (Explainable AI - XAI)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceStep {
    pub step: usize,
    pub node: String,
    pub action: String,
    pub details: String,
}

/// Main state object passed between graph nodes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SupervisoryMasState {
    pub original_query: String,
    pub messages: Vec<MasMessage>,
    pub current_agent: AgentRole,
    pub delegation_count: usize,
    pub supervisor_decision: Option<SupervisorDecision>,
    pub generated_chart_b64: Option<String>,
    pub trace: Vec<TraceStep>,
    pub final_answer: Option<String>,
}

impl SupervisoryMasState {
    pub fn new(query: &str) -> Self {
        let mut state = Self {
            original_query: query.to_string(),
            messages: Vec::new(),
            current_agent: AgentRole::Supervisor,
            delegation_count: 0,
            supervisor_decision: None,
            generated_chart_b64: None,
            trace: Vec::new(),
            final_answer: None,
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

    /// Formats history for the LLM agent's context (equivalent to MessagesPlaceholder)
    pub fn format_history_for_prompt(&self) -> String {
        self.messages
            .iter()
            .map(|m| {
                let role = match m.sender {
                    AgentRole::User => "USER",
                    AgentRole::Supervisor => "SUPERVISOR",
                    AgentRole::Researcher => "RESEARCHER",
                    AgentRole::Coder => "CODER",
                };
                format!("[{role}]: {}", m.content)
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// Safe extraction of Base64 data from the generated graph (equivalent to `extract_base64_from_output`)
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
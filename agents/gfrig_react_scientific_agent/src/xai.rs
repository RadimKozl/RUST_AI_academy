use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolExecutionRecord {
    pub step_index: usize,
    pub tool_name: String,
    pub decision_rationale: String,
    pub input_arguments: serde_json::Value,
    pub raw_output: String,
    pub execution_time_ms: u128,
    pub status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeTrace {
    pub node_name: String,
    pub timestamp_utc: String,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XaiReport {
    pub query: String,
    pub model_used: String,
    pub nodes_visited: Vec<NodeTrace>,
    pub tool_decisions: Vec<ToolExecutionRecord>,
    pub agent_reasoning_summary: String,
    pub final_answer: String,
    pub total_duration_ms: u128,
}

#[derive(Clone, Default)]
pub struct XaiTracker {
    inner: Arc<Mutex<XaiTrackerState>>,
}

#[derive(Default)]
struct XaiTrackerState {
    nodes: Vec<NodeTrace>,
    tools: Vec<ToolExecutionRecord>,
}

impl XaiTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_node(&self, node_name: &str, description: &str) {
        let mut state = self.inner.lock().unwrap();
        state.nodes.push(NodeTrace {
            node_name: node_name.to_string(),
            timestamp_utc: format!("{:?}", std::time::SystemTime::now()),
            description: description.to_string(),
        });
    }

    pub fn record_tool(
        &self,
        tool_name: &str,
        rationale: &str,
        args: serde_json::Value,
        output: &str,
        duration_ms: u128,
        is_ok: bool,
    ) {
        let mut state = self.inner.lock().unwrap();
        let step = state.tools.len() + 1;
        state.tools.push(ToolExecutionRecord {
            step_index: step,
            tool_name: tool_name.to_string(),
            decision_rationale: rationale.to_string(),
            input_arguments: args,
            raw_output: output.to_string(),
            execution_time_ms: duration_ms,
            status: if is_ok { "SUCCESS".into() } else { "ERROR".into() },
        });
    }

    pub fn build_report(
        &self,
        query: &str,
        model: &str,
        reasoning_summary: &str,
        final_answer: &str,
        start_time: Instant,
    ) -> XaiReport {
        let state = self.inner.lock().unwrap();
        XaiReport {
            query: query.to_string(),
            model_used: model.to_string(),
            nodes_visited: state.nodes.clone(),
            tool_decisions: state.tools.clone(),
            agent_reasoning_summary: reasoning_summary.to_string(),
            final_answer: final_answer.to_string(),
            total_duration_ms: start_time.elapsed().as_millis(),
        }
    }
}
use crate::tools::{ArxivTool, SymbolicaTool};
use crate::xai::{XaiReport, XaiTracker};
use rig::prelude::*;
use rig::providers::ollama::OllamaConfig;
use std::time::Instant;

#[derive(Clone, Default, Debug)]
pub struct GraphState {
    pub input_query: String,
    pub execution_trace: Vec<String>,
    pub final_output: String,
    pub xai_report: Option<XaiReport>,
}

pub struct ScientificGraphWorkflow {
    state: GraphState,
}

impl ScientificGraphWorkflow {
    pub fn new(query: &str) -> Self {
        Self {
            state: GraphState {
                input_query: query.to_string(),
                ..Default::default()
            },
        }
    }

    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let workflow_start = Instant::now();
        let tracker = XaiTracker::new();
        let model_name = "qwen3.5:9b";

        println!("--- GRAPH-FLOW START: State Initialization ---");
        self.state.execution_trace.push("Node: InitState".to_string());
        tracker.record_node("InitState", "Initialized workflow state and XAI telemetry tracker.");

        let ollama = OllamaConfig::new().client();
        let model = ollama.completion(model_name);

        let agent = AgentBuilder::new(model)
            .preamble(
                "You are a scientific assistant agent in a Rust graph-flow workflow. \
                 Use the provided tools for literature search and for math. \
                 When calling ANY tool, you MUST fill in the 'rationale' parameter explaining \
                 why that tool is needed for the user's query. \
                 Base your answer ONLY on what the tools returned: do not invent paper \
                 details and do not do the math yourself. \
                 Structure your final response into two clear sections:\n\
                 ### Reasoning Trace\n\
                 (Briefly explain step-by-step how you decomposed the query and synthesized the tool outputs)\n\
                 ### Final Answer\n\
                 (The complete answer based strictly on tool results)"
            )
            .tool(ArxivTool { tracker: tracker.clone() })
            .tool(SymbolicaTool { tracker: tracker.clone() })
            .build();

        println!("--- GRAPH-FLOW NODE: Agent Reasoning & Tool Execution ---");
        self.state.execution_trace.push("Node: ReActAgentLoop".to_string());
        tracker.record_node(
            "ReActAgentLoop",
            "Started ReAct loop with qwen3.5:9b and tools [search_arxiv, symbolica_calculate].",
        );

        let response = agent
            .prompt(&self.state.input_query)
            .max_turns(5)
            .await?;

        let raw_text = response.to_string();
        
        // Separation of the reasoning trace and the final answer for the XAI report
        let (reasoning_summary, final_answer) = if let Some((reason, ans)) = raw_text.split_once("### Final Answer") {
            (
                reason.replace("### Reasoning Trace", "").trim().to_string(),
                ans.trim().to_string(),
            )
        } else {
            ("See final_answer for integrated reasoning.".to_string(), raw_text.clone())
        };

        self.state.final_output = raw_text;
        self.state.execution_trace.push("Node: FinishGraph".to_string());
        tracker.record_node("FinishGraph", "Completed execution and compiled XAI JSON report.");

        self.state.xai_report = Some(tracker.build_report(
            &self.state.input_query,
            model_name,
            &reasoning_summary,
            &final_answer,
            workflow_start,
        ));

        println!("--- GRAPH-FLOW END: Success ---");
        Ok(())
    }

    pub fn get_state(&self) -> &GraphState {
        &self.state
    }
}
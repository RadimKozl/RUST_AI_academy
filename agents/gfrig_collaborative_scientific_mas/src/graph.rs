use crate::state::{extract_base64_from_output, AgentRole, ScientificMasState};
use crate::tools_coder::{RenderBarChartTool, SymbolicCalculateTool};
use crate::tools_researcher::{ArxivTool, DocsRagTool, WikipediaTool};
use rig::prelude::*;
use rig::providers::ollama::OllamaConfig;

#[derive(Debug, PartialEq, Eq)]
enum NextNode {
    Researcher,
    Coder,
    End,
}

pub struct CollaborativeScientificMas {
    model_name: String,
    max_graph_steps: usize,
}

impl CollaborativeScientificMas {
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            max_graph_steps: 10,
        }
    }

    pub async fn run(&self, query: &str) -> anyhow::Result<ScientificMasState> {
        let mut state = ScientificMasState::new(query);
        state.log_trace("START", "InitGraph", format!("Task: {query}"));

        let ollama = OllamaConfig::new().client();

        // 1. RESEARCHER AGENT (Facts + RAG documentation, no execution of calculations)
        let researcher = AgentBuilder::new(ollama.completion(&self.model_name))
            .preamble(
                "You are a **Scientific Researcher** in a Collaborative Multi-Agent System.\n\
                PRIMARY TASKS:\n\
                1. Fact Finding: Use 'wikipedia' or 'arxiv' exactly ONCE to gather data.\n\
                2. Handoff: Immediately summarize the findings and end your message with 'Pass to Coder.'\n\
                CRITICAL RULE:\n\
                - DO NOT loop or make multiple tool calls. Do it in 1 step and hand off.",
            )
            .tool(WikipediaTool)
            .tool(ArxivTool)
            .tool(DocsRagTool)
            .build();

        // 2. CODER AGENT (Symbolic Mathematics + Visualization + Final Synthesis)
        let coder = AgentBuilder::new(ollama.completion(&self.model_name))
            .preamble(
                "You are a **Scientific Coder and Analyst** in a Collaborative Multi-Agent System.\n\
                PRIMARY TASKS:\n\
                1. Symbolic Math: Use 'symbolic_calculate' when asked to solve equations or derivatives.\n\
                2. Visualization: When Researcher provides data for countries, cities, or statistics and asks for a bar chart, you MUST call the 'render_bar_chart' tool with the exact labels and numeric values provided.\n\
                CRITICAL RULES:\n\
                - DO NOT talk about ComfyUI, image prompts, or unrelated topics.\n\
                - Once the tool 'render_bar_chart' is executed, output 'FINAL ANSWER:' followed by a summary stating that the chart has been saved successfully.",
            )
            .tool(SymbolicCalculateTool)
            .tool(RenderBarChartTool)
            .build();

        // Entry edge: START -> Researcher (same as in the LangGraph example)
        let mut current_node = NextNode::Researcher;
        let mut step_counter = 0;

        while current_node != NextNode::End && step_counter < self.max_graph_steps {
            step_counter += 1;

            match current_node {
                NextNode::Researcher => {
                    println!("-> [Node: Researcher] (Graph Step #{step_counter})");
                    let prompt = format!(
                        "Conversation History:\n{}\n\n\
                         Perform your research task now and hand off to Coder.",
                        state.format_history_for_prompt()
                    );

                    let response = researcher.prompt(&prompt).max_turns(8).await?.to_string();
                    state.push_message(AgentRole::Researcher, &response);
                    state.log_trace("Researcher", "AgentStep", &response);

                    current_node = self.route_researcher(&mut state, &response);
                }

                NextNode::Coder => {
                    println!("-> [Node: Coder] (Graph Step #{step_counter})");
                    let prompt = format!(
                        "Conversation History:\n{}\n\n\
                         Execute the required symbolic math or visualization using your tools, or request missing info from Researcher.",
                        state.format_history_for_prompt()
                    );

                    let response = coder.prompt(&prompt).max_turns(4).await?.to_string();

                    // Extraction of any potential Base64 payload from the tool output / response
                    if let Some(b64) = extract_base64_from_output(&response) {
                        println!("-> [CoderNode] Clean Base64 chart payload extracted ({} chars).", b64.len());
                        state.generated_chart_b64 = Some(b64);
                    }

                    state.push_message(AgentRole::Coder, &response);
                    state.log_trace("Coder", "AgentStep", &response);

                    current_node = self.route_coder(&mut state, &response);
                }

                NextNode::End => break,
            }
        }

        if state.final_answer.is_none() {
            let fallback = state
                .messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_else(|| "Workflow terminated without explicit final answer.".into());
            state.final_answer = Some(fallback);
        }

        state.log_trace("END", "FinishGraph", "Collaborative MAS workflow completed.");
        Ok(state)
    }

    /// Router for Researcher (corresponds to `route_researcher` in graph_2.py)
    fn route_researcher(&self, state: &mut ScientificMasState, output: &str) -> NextNode {
        let lower = output.to_lowercase();
        state.loop_count += 1;

        // Anti-loop protection: after 3 iterations, we force a handoff to the Coder
        if state.loop_count >= 3 {
            println!("-> [Router: Researcher -> Coder] Anti-loop limit reached. Forcing handoff.");
            state.push_message(
                AgentRole::Researcher,
                "Data collection capped by anti-loop guard. Pass to Coder with available data.",
            );
            state.loop_count = 0;
            return NextNode::Coder;
        }

        // Standard handover to the Coder
        if lower.contains("pass to coder") || lower.contains("data found") || lower.contains("syntax found") || !output.trim().is_empty() {
            println!("-> [Router: Researcher -> Coder] Research complete, handing off.");
            state.loop_count = 0;
            NextNode::Coder
        } else {
            println!("-> [Router: Researcher -> END] Empty output.");
            state.final_answer = Some(output.to_string());
            NextNode::End
        }
    }

    /// Router for Coder (corresponds to `route_coder` in graph_2.py)
    fn route_coder(&self, state: &mut ScientificMasState, output: &str) -> NextNode {
        let lower = output.to_lowercase();

        // Does the Coder need additional facts or syntax from the Researcher?
        if lower.contains("need researcher")
            || lower.contains("need syntax")
            || lower.contains("missing data")
            || lower.contains("request researcher")
        {
            println!("-> [Router: Coder -> Researcher] Coder requested additional data/syntax.");
            return NextNode::Researcher;
        }

        // Do we have a final answer?
        println!("-> [Router: Coder -> END] Final result produced.");
        let clean_answer = output
            .split("FINAL ANSWER:")
            .last()
            .unwrap_or(output)
            .trim()
            .to_string();
        state.final_answer = Some(clean_answer);
        NextNode::End
    }
}
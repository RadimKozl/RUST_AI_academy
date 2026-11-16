use crate::state::{AgentRole, SupervisorDecision, SupervisoryMasState};
use crate::tools_coder::{RenderBarChartTool, SymbolicCalculateTool};
use crate::tools_researcher::{ArxivTool, DocsRagTool, WikipediaTool};
use rig::prelude::*;
use rig::providers::ollama::OllamaConfig;

#[derive(Debug, PartialEq, Eq)]
enum GraphNode {
    Supervisor,
    Researcher,
    Coder,
    End,
}

pub struct SupervisoryScientificMas {
    supervisor_model: String,
    researcher_model: String,
    coder_model: String,
    max_delegations: usize,
}

impl SupervisoryScientificMas {
    pub fn new(supervisor_model: &str, researcher_model: &str, coder_model: &str) -> Self {
        Self {
            supervisor_model: supervisor_model.to_string(),
            researcher_model: researcher_model.to_string(),
            coder_model: coder_model.to_string(),
            max_delegations: 12,
        }
    }

    /// Starts the entire Supervisory Graph-Flow (equivalent to `graph.invoke(initial_state)`)
    pub async fn run(&self, query: &str) -> anyhow::Result<SupervisoryMasState> {
        let mut state = SupervisoryMasState::new(query);
        state.log_trace("START", "InitGraph", format!("Task: {query}"));

        let ollama = OllamaConfig::new().client();

        // 1. SUPERVISOR AGENT (coordination and decision-making)
        let supervisor = AgentBuilder::new(ollama.completion(&self.supervisor_model))
            .preamble(
                "You coordinate Researcher and Coder agents.\n\
                 RESEARCHER: finds facts, data, or library syntax.\n\
                 CODER: executes symbolic math ('symbolic_calculate') or renders charts ('render_bar_chart').\n\n\
                 OUTPUT FORMAT (choose ONE, use EXACT format):\n\
                 To delegate to researcher:\n\
                 DELEGATE: RESEARCHER\n\
                 [specific task description]\n\n\
                 To delegate to coder:\n\
                 DELEGATE: CODER\n\
                 [specific task with exact data or expression to execute]\n\n\
                 To finish:\n\
                 FINAL ANSWER:\n\
                 [complete final answer to the user]\n\n\
                 CRITICAL RULES:\n\
                 - Always start with 'DELEGATE:' or 'FINAL ANSWER:'.\n\
                 - Keep delegation tasks focused and concise. Do NOT combine complex mathematical proofs and multi-curve chart rendering into a single delegation.\n\
                 - When you see 'RESEARCH COMPLETE', immediately delegate to CODER if computation or plotting is needed.\n\
                 - When you see 'EXECUTION COMPLETE', immediately output 'FINAL ANSWER:'."
            )
            .build();

        // 2. RESEARCHER AGENT (fact-finding and RAG documentation)
        let researcher = AgentBuilder::new(ollama.completion(&self.researcher_model))
            .preamble(
                "You are a Researcher agent. Your job is to find information using tools and report back.\n\
                 TOOLS: 'wikipedia', 'arxiv', 'search_library_docs'.\n\
                 WORKFLOW:\n\
                 1. Call the appropriate tool ONCE to gather facts or syntax.\n\
                 2. Immediately respond starting with 'RESEARCH COMPLETE' followed by a concise summary of findings.\n\
                 CRITICAL: Do NOT loop. Make at most 1-2 tool calls and report."
            )
            .tool(WikipediaTool)
            .tool(ArxivTool)
            .tool(DocsRagTool)
            .build();

        // 3. CODER AGENT (execution of symbolic mathematics and visualization)
        let coder = AgentBuilder::new(ollama.completion(&self.coder_model))
            .preamble(
                "You are a Scientific Coder Agent working under a Supervisor.\n\
                 TOOLS:\n\
                 - 'symbolic_calculate': for derivatives, integrals, equations, and ODEs.\n\
                 - 'render_bar_chart': for creating PNG bar charts from numeric labels and values.\n\n\
                 CRITICAL RULES:\n\
                 - Be extremely concise in your text responses. Execute the tool immediately and report the direct result.\n\
                 - Start your response with 'EXECUTION COMPLETE:' followed by the exact output."
            )
            .tool(SymbolicCalculateTool)
            .tool(RenderBarChartTool)
            .build();

        // Entry edge: START -> Supervisor
        let mut current_node = GraphNode::Supervisor;

        while current_node != GraphNode::End {
            match current_node {
                GraphNode::Supervisor => {
                    println!("\n🔍 SUPERVISOR: Analyzing situation...");
                    state.current_agent = AgentRole::Supervisor;

                    if state.delegation_count >= self.max_delegations {
                        println!("⚠️ SUPERVISOR: Max delegations reached. Forcing finish.");
                        let fallback = "Task processing limit reached. Providing available results.".to_string();
                        state.final_answer = Some(fallback.clone());
                        state.push_message(AgentRole::Supervisor, format!("FINAL ANSWER:\n{fallback}"));
                        state.log_trace("Supervisor", "ForceFinish", "Max delegations reached");
                        break;
                    }

                    let prompt = format!(
                        "Conversation History:\n{}\n\nDecide the next step using the exact required format.",
                        state.format_history_for_prompt()
                    );

                    let response = supervisor.prompt(&prompt).max_turns(2).await?.to_string();
                    let decision = Self::parse_supervisor_decision(&response);

                    state.push_message(AgentRole::Supervisor, &response);
                    state.supervisor_decision = Some(decision.clone());

                    match &decision {
                        SupervisorDecision::Delegate { to_agent, task } => {
                            state.delegation_count += 1;
                            println!("📋 SUPERVISOR DECISION: DELEGATE -> {:?} ({})", to_agent, task);
                            state.log_trace("Supervisor", "Delegate", format!("To {:?}: {task}", to_agent));
                        }
                        SupervisorDecision::Finish { final_answer } => {
                            println!("🏁 SUPERVISOR DECISION: FINISH");
                            state.final_answer = Some(final_answer.clone());
                            state.log_trace("Supervisor", "Finish", final_answer);
                        }
                    }

                    current_node = Self::route_supervisor(&decision);
                }

                GraphNode::Researcher => {
                    println!("📚 RESEARCHER: Executing assigned task...");
                    state.current_agent = AgentRole::Researcher;

                    let task = match &state.supervisor_decision {
                        Some(SupervisorDecision::Delegate { task, .. }) => task.clone(),
                        _ => state.original_query.clone(),
                    };

                    let prompt = format!(
                        "Conversation History:\n{}\n\nSUPERVISOR TASK: {task}\nUse your tools and respond with RESEARCH COMPLETE.",
                        state.format_history_for_prompt()
                    );

                    // max_turns(4) provides protection against tool loops
                    let response = researcher.prompt(&prompt).max_turns(8).await?.to_string();
                    println!("✅ RESEARCHER: Task completed. Reporting to Supervisor.");

                    state.push_message(AgentRole::Researcher, &response);
                    state.log_trace("Researcher", "ResearchComplete", &response);

                    // Edge: Researcher -> Supervisor
                    current_node = GraphNode::Supervisor;
                }

                GraphNode::Coder => {
                    println!("💻 CODER: Executing code task...");
                    state.current_agent = AgentRole::Coder;

                    let task = match &state.supervisor_decision {
                        Some(SupervisorDecision::Delegate { task, .. }) => task.clone(),
                        _ => state.original_query.clone(),
                    };

                    let prompt = format!(
                        "Conversation History:\n{}\n\nSUPERVISOR TASK: {task}\nExecute the required calculation or chart tool now.",
                        state.format_history_for_prompt()
                    );

                    let response = coder.prompt(&prompt).max_turns(4).await?.to_string();

                    // Retrieving Base64 data from thread-local storage (a pure state variable for XAI)
                    if let Some(b64) = crate::tools_coder::take_last_generated_base64() {
                        println!("📊 CODER [XAI State]: Base64 chart payload securely stored in state ({} chars).", b64.len());
                        state.generated_chart_b64 = Some(b64);
                    }

                    println!("✅ CODER: Task completed. Reporting to Supervisor.");
                    state.push_message(AgentRole::Coder, &response);
                    state.log_trace("Coder", "ExecutionComplete", &response);

                    // Edge: Coder -> Supervisor
                    current_node = GraphNode::Supervisor;
                }

                GraphNode::End => break,
            }
        }

        state.log_trace("END", "FinishGraph", "Supervisory MAS workflow completed.");
        Ok(state)
    }

    /// Deterministic parser for Supervisor decisions (equivalent to `_parse_decision` in Python)
    pub fn parse_supervisor_decision(content: &str) -> SupervisorDecision {
        let trimmed = content.trim();
        let upper = trimmed.to_uppercase();

        if let Some(idx) = upper.find("FINAL ANSWER:") {
            let answer = trimmed[idx + "FINAL ANSWER:".len()..].trim().to_string();
            return SupervisorDecision::Finish {
                final_answer: if answer.is_empty() {
                    "Task completed.".to_string()
                } else {
                    answer
                },
            };
        }

        if let Some(idx) = upper.find("DELEGATE:") {
            let rest = trimmed[idx + "DELEGATE:".len()..].trim();
            let mut lines = rest.lines();
            let target_line = lines.next().unwrap_or("").to_uppercase();
            let remaining_task = lines.collect::<Vec<_>>().join("\n").trim().to_string();

            let to_agent = if target_line.contains("CODER") {
                AgentRole::Coder
            } else {
                AgentRole::Researcher
            };

            let task = if remaining_task.is_empty() {
                rest.to_string()
            } else {
                remaining_task
            };

            return SupervisorDecision::Delegate { to_agent, task };
        }

        // Fallback heuristic if the LLM does not follow the exact prefix
        if upper.contains("EXECUTION COMPLETE") {
            SupervisorDecision::Finish {
                final_answer: trimmed.to_string(),
            }
        } else if upper.contains("RESEARCH COMPLETE") {
            SupervisorDecision::Delegate {
                to_agent: AgentRole::Coder,
                task: trimmed.to_string(),
            }
        } else {
            SupervisorDecision::Delegate {
                to_agent: AgentRole::Researcher,
                task: trimmed.to_string(),
            }
        }
    }

    /// Supervisor routing function (equivalent to `route_supervisor` in Python)
    fn route_supervisor(decision: &SupervisorDecision) -> GraphNode {
        match decision {
            SupervisorDecision::Finish { .. } => GraphNode::End,
            SupervisorDecision::Delegate { to_agent, .. } => match to_agent {
                AgentRole::Researcher => GraphNode::Researcher,
                AgentRole::Coder => GraphNode::Coder,
                _ => GraphNode::End,
            },
        }
    }
}
// src/graph_swarm.rs
use crate::state::{AgentRole, SwarmMasState};
use crate::tools_coder::{RenderBarChartTool, SymbolicCalculateTool};
use crate::tools_researcher::{ArxivTool, DocsRagTool, WikipediaTool};
use rig::prelude::*;
use rig::providers::ollama::OllamaConfig;
use rig::tool::{Tool, ToolContext};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

#[derive(Clone, Debug)]
pub enum SwarmAction {
    Delegate { target: AgentRole, task: String },
    Complete { final_answer: String },
}

thread_local! {
    static LAST_SWARM_ACTION: RefCell<Option<SwarmAction>> = RefCell::new(None);
}

pub fn take_last_swarm_action() -> Option<SwarmAction> {
    LAST_SWARM_ACTION.with(|a| a.borrow_mut().take())
}

pub fn set_swarm_action(action: SwarmAction) {
    LAST_SWARM_ACTION.with(|a| {
        *a.borrow_mut() = Some(action);
    });
}

// ---------------------------------------------------------------------------
// DELEGATOR TOOL (Nativní tool pro předávání úkolů mezi agenty)
// ---------------------------------------------------------------------------
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
pub struct DelegatorArgs {
    /// Target agent to delegate to: "coder" or "researcher"
    pub target: String,
    /// Task description or instructions for the target agent
    pub task: String,
}

pub struct DelegatorTool;

impl Tool for DelegatorTool {
    const NAME: &'static str = "delegator";
    type Error = std::io::Error;
    type Args = DelegatorArgs;
    type Output = String;

    fn description(&self) -> String {
        "Delegate a subtask to another agent in the swarm ('coder' or 'researcher').".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(DelegatorArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let target_role = match args.target.to_lowercase().as_str() {
            "coder" => AgentRole::Coder,
            _ => AgentRole::Researcher,
        };
        set_swarm_action(SwarmAction::Delegate {
            target: target_role.clone(),
            task: args.task.clone(),
        });
        Ok(format!("Successfully delegated task to {:?}.", target_role))
    }
}

// ---------------------------------------------------------------------------
// COMPLETE TASK TOOL (Nativní tool pro ukončení běhu)
// ---------------------------------------------------------------------------
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
pub struct CompleteArgs {
    /// Final answer or summary to return to the user
    pub final_answer: String,
}

pub struct CompleteTool;

impl Tool for CompleteTool {
    const NAME: &'static str = "complete_task";
    type Error = std::io::Error;
    type Args = CompleteArgs;
    type Output = String;

    fn description(&self) -> String {
        "Mark the task as successfully completed and provide the final answer to the user.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(CompleteArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        set_swarm_action(SwarmAction::Complete {
            final_answer: args.final_answer.clone(),
        });
        Ok("Task marked as completed.".to_string())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum SwarmNode {
    Researcher,
    Coder,
    End,
}

pub struct SwarmScientificMas {
    researcher_model: String,
    coder_model: String,
    max_steps: usize,
}

impl SwarmScientificMas {
    pub fn new(researcher_model: &str, coder_model: &str) -> Self {
        Self {
            researcher_model: researcher_model.to_string(),
            coder_model: coder_model.to_string(),
            max_steps: 10,
        }
    }

    /// Spuštění Swarm workflow
    pub async fn run(&self, query: &str) -> anyhow::Result<SwarmMasState> {
        let mut state = SwarmMasState::new(query);
        let ollama = OllamaConfig::new().client();

        // 1. RESEARCHER AGENT (vyhledávací nástroje + delegator + complete_task)
        let researcher = AgentBuilder::new(ollama.completion(&self.researcher_model))
            .preamble(
                "You are a Researcher agent in a swarm system. Your job is to find information using tools ('wikipedia', 'arxiv', 'search_library_docs').\n\
                 When you have the required facts or syntax, use the 'delegator' tool to hand off the task to the 'coder' agent.\n\
                 When the final answer is ready, use the 'complete_task' tool."
            )
            .tool(WikipediaTool)
            .tool(ArxivTool)
            .tool(DocsRagTool)
            .tool(DelegatorTool)
            .tool(CompleteTool)
            .build();

        // 2. CODER AGENT (výpočetní a vizualizační nástroje + delegator + complete_task)
        let coder = AgentBuilder::new(ollama.completion(&self.coder_model))
            .preamble(
                "You are a Scientific Coder Agent in a swarm system. Your job is to execute calculations or render charts ('symbolic_calculate', 'render_bar_chart').\n\
                 If you need more information or syntax, use the 'delegator' tool to send a task back to the 'researcher'.\n\
                 When finished, use the 'complete_task' tool."
            )
            .tool(SymbolicCalculateTool)
            .tool(RenderBarChartTool)
            .tool(DelegatorTool)
            .tool(CompleteTool)
            .build();

        let mut current_node = SwarmNode::Researcher;

        while current_node != SwarmNode::End {
            if state.delegation_count >= self.max_steps {
                println!("⚠️ SWARM: Max steps reached. Forcing completion.");
                state.final_answer = Some("Max execution steps reached.".to_string());
                break;
            }

            match current_node {
                SwarmNode::Researcher => {
                    println!("\n📚 SWARM [RESEARCHER]: Processing task...");
                    state.current_agent = AgentRole::Researcher;

                    let prompt = format!(
                        "Conversation History:\n{}\n\nYour turn as Researcher. Use tools or delegate to Coder.",
                        state.format_history()
                    );

                    let response = researcher.prompt(&prompt).max_turns(6).await?.to_string();
                    println!("💬 RESEARCHER RESPONSE:\n{response}");

                    state.push_message(AgentRole::Researcher, &response);
                    state.delegation_count += 1;

                    if let Some(action) = take_last_swarm_action() {
                        current_node = Self::route_from_action(&action, &mut state);
                    } else {
                        current_node = SwarmNode::Coder;
                    }
                }

                SwarmNode::Coder => {
                    println!("\n💻 SWARM [CODER]: Processing task...");
                    state.current_agent = AgentRole::Coder;

                    let prompt = format!(
                        "Conversation History:\n{}\n\nYour turn as Coder. Execute calculations/charts or delegate back.",
                        state.format_history()
                    );

                    let response = coder.prompt(&prompt).max_turns(6).await?.to_string();
                    println!("💬 CODER RESPONSE:\n{response}");

                    state.push_message(AgentRole::Coder, &response);
                    state.delegation_count += 1;

                    if let Some(action) = take_last_swarm_action() {
                        current_node = Self::route_from_action(&action, &mut state);
                    } else {
                        current_node = SwarmNode::End;
                    }
                }

                SwarmNode::End => break,
            }
        }

        if state.final_answer.is_none() {
            if let Some(last_msg) = state.messages.last() {
                state.final_answer = Some(last_msg.content.clone());
            } else {
                state.final_answer = Some("Task completed.".to_string());
            }
        }

        Ok(state)
    }

    fn route_from_action(action: &SwarmAction, state: &mut SwarmMasState) -> SwarmNode {
        match action {
            SwarmAction::Delegate { target, task } => {
                println!("🔀 SWARM DELEGATION -> {:?} | Task: {}", target, task);
                state.push_message(AgentRole::User, format!("[Delegation to {:?}]: {}", target, task));
                match target {
                    AgentRole::Researcher => SwarmNode::Researcher,
                    AgentRole::Coder => SwarmNode::Coder,
                    AgentRole::User => SwarmNode::End,
                }
            }
            SwarmAction::Complete { final_answer } => {
                println!("🏁 SWARM COMPLETED | Answer: {}", final_answer);
                state.final_answer = Some(final_answer.clone());
                SwarmNode::End
            }
        }
    }
}
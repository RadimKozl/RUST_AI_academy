use crate::marl::{MarlPolicyEngine, SwarmAction};
use crate::memory::{AgonIntentMessage, PersonalGraphRag, ReasoningStep};
use crate::tools::{ArxivArgs, ArxivSearchTool, GraphRagQueryTool};
use async_trait::async_trait;
use graph_flow::{Context, GraphError, NextAction, Task, TaskResult};
use rig::prelude::*;
use rig::providers::ollama::OllamaConfig;
use rig::tool::{Tool, ToolContext};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Shared state of the entire Swarm in graph-flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgonSwarmState {
    pub episode: usize,
    pub step: usize,
    pub task_specification: String,
    pub current_phase: String,
    /// Short-term memory: a shared message queue between agents
    pub short_term_messages: Vec<AgonIntentMessage>,
    /// Individual agent proposals in the current episode (for competition evaluation)
    pub proposals: HashMap<String, AgentProposal>,
    /// MARL policies for all 3 agents
    pub marl_engines: HashMap<String, MarlPolicyEngine>,
    pub consensus_reached: bool,
    pub final_agon_spec: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProposal {
    pub agent_name: String,
    pub action_taken: SwarmAction,
    pub content: String,
    pub confidence: f64,
    pub critique_penalties: f64,
}

/// A graph task representing a single autonomous Rig agent with its own G-RAG
pub struct AgonSwarmNode {
    pub task_id: String,
    pub agent_name: String,
    pub role_preamble: String,
    pub personal_rag: Arc<Mutex<PersonalGraphRag>>,
}

#[async_trait]
impl Task for AgonSwarmNode {
    fn id(&self) -> &str {
        &self.task_id
    }

    async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
        let mut state: AgonSwarmState = context.get("swarm_state").ok_or_else(|| {
            GraphError::TaskExecutionFailed("Missing swarm_state in context".into())
        })?;

        // 1. The agent selects an action according to its MARL policy (JAL-AM + Regret Matching)
        let action = state
            .marl_engines
            .get(&self.agent_name)
            .map(|e| e.select_action(&state.current_phase))
            .unwrap_or(SwarmAction::ProposeSpecification);

        // 2. Retrieving context from a dedicated, isolated G-RAG and shared Short-Term Memory
        let local_rag_context = {
            let rag = self.personal_rag.lock().unwrap();
            rag.query_graph_rag(&state.task_specification, 3)
        };

        let recent_intents: String = state
            .short_term_messages
            .iter()
            .rev()
            .take(4)
            .map(|m| {
                format!(
                    "[{} -> {} | Intent: {} | conf={:.2}]: {}",
                    m.sender, m.target, m.intent_type, m.confidence, m.payload
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        // 3. Constructing the prompt for the Rig agent
        let prompt_input = format!(
            "TASK (task.txt):\n{}\n\nYOUR SELECTED MARL ACTION: {}\n\nYOUR PERSONAL G-RAG CONTEXT:\n{}\n\nRECENT MESSAGES IN THE SWARM (Short-Term Memory):\n{}\n\n\
            Instructions: Execute the selected action. If you are competing (CritiqueAndCompete), identify weaknesses in your colleagues' proposals and suggest a more precise, type-safe solution in Rust/AGON. \
            Keep the answer under 250 words. \
            At the end of your response, include a line in the following format: CONFIDENCE: 0.XX",
            state.task_specification,
            action.as_str(),
            local_rag_context,
            recent_intents
        );

        // 4. Calling the local LLM (Ollama) via Rig with connected tools
        let response_text = self.execute_rig_agent(&prompt_input, action).await;

        // 5. Extraction of confidence scores
        let confidence = Self::parse_confidence(&response_text);

        // 6. Writing to Personal Reasoning Memory & Long-Term G-RAG
        {
            let mut rag = self.personal_rag.lock().unwrap();
            let step_id = rag.reasoning_trace.len() + 1;
            rag.reasoning_trace.push(ReasoningStep {
                step_id,
                agent_id: self.agent_name.clone(),
                thought: format!("Executed action {:?} in phase {}", action, state.current_phase),
                tool_called: if action == SwarmAction::DeepArxivResearch {
                    Some("search_arxiv_and_memorize".to_string())
                } else {
                    Some("query_personal_graph_rag".to_string())
                },
                confidence,
            });
            rag.upsert_entity(
                &format!("Proposal_{}_Ep{}", self.agent_name, state.episode),
                "AgonConstruct",
                &response_text.chars().take(250).collect::<String>(),
            );
        }

        // 7. Publishing the AGON Intent to the shared Short-Term Memory
        let msg_id = state.short_term_messages.len() + 1;
        state.short_term_messages.push(AgonIntentMessage {
            id: msg_id,
            sender: self.agent_name.clone(),
            target: "AllSwarm".to_string(),
            intent_type: action.as_str().to_string(),
            payload: response_text.chars().take(350).collect(),
            confidence,
        });

        // 8. Saving the design for MARL evaluation
        if action == SwarmAction::CritiqueAndCompete {
            for (other_name, prop) in state.proposals.iter_mut() {
                if other_name != &self.agent_name && prop.confidence < confidence {
                    prop.critique_penalties += 0.25;
                }
            }
        }

        state.proposals.insert(
            self.agent_name.clone(),
            AgentProposal {
                agent_name: self.agent_name.clone(),
                action_taken: action,
                content: response_text,
                confidence,
                critique_penalties: 0.0,
            },
        );

        state.step += 1;
        context.set("swarm_state", state)?;

        Ok(TaskResult::new(
            Some(format!(
                "Agent {} finished action {}",
                self.agent_name,
                action.as_str()
            )),
            NextAction::Continue,
        ))
    }
}

impl AgonSwarmNode {
    async fn try_llm(&self, prompt_input: &str) -> Option<String> {
        let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3.5:4b".to_string());
        let ollama = OllamaConfig::new().client();

        let agent = AgentBuilder::new(ollama.completion(&model))
            .preamble(&self.role_preamble)
            .tool(ArxivSearchTool {
                personal_memory: Arc::clone(&self.personal_rag),
            })
            .tool(GraphRagQueryTool {
                personal_memory: Arc::clone(&self.personal_rag),
            })
            .build();

        match agent.prompt(prompt_input).max_turns(3).await {
            Ok(resp) => Some(Self::strip_think(&resp.to_string())),
            Err(e) => {
                eprintln!(
                    "⚠️  [{}] Ollama call failed ({e}); using local fallback.",
                    self.agent_name
                );
                None
            }
        }
    }

    fn strip_think(text: &str) -> String {
        match (text.find("<think>"), text.find("</think>")) {
            (Some(s), Some(e)) if e > s => {
                format!("{}{}", &text[..s], &text[e + "</think>".len()..])
                    .trim()
                    .to_string()
            }
            _ => text.trim().to_string(),
        }
    }

    async fn execute_rig_agent(&self, prompt_input: &str, action: SwarmAction) -> String {
        let llm_enabled = std::env::var("AGON_LLM")
            .map(|v| !v.eq_ignore_ascii_case("off"))
            .unwrap_or(true);
        if llm_enabled {
            if let Some(answer) = self.try_llm(prompt_input).await {
                return answer;
            }
        }

        if action == SwarmAction::DeepArxivResearch {
            let tool = ArxivSearchTool {
                personal_memory: Arc::clone(&self.personal_rag),
            };
            let mut ctx = ToolContext::new();
            let _ = tool
                .call(
                    &mut ctx,
                    ArxivArgs {
                        query: "QMIX MARL actor model type safety".to_string(),
                        max_results: 2,
                    },
                )
                .await;
        }

        let rag_snapshot = {
            let rag = self.personal_rag.lock().unwrap();
            rag.query_graph_rag(prompt_input, 2)
        };

        let response_content = format!(
            "Synthesized solution for AGON based on local G-RAG memory & action {:?}.\nRetrieved Knowledge Context:\n{}",
            action, rag_snapshot
        );

        let memory_bonus = {
            let rag = self.personal_rag.lock().unwrap();
            (rag.entities.len() as f64 * 0.02).min(0.10)
        };
        let confidence = (0.86 + memory_bonus).clamp(0.0, 1.0);

        format!(
            "[{} | Action: {}] Proposal for AGON: Strongly typed Intent<T> structures. {}\nCONFIDENCE: {:.2}",
            self.agent_name,
            action.as_str(),
            response_content,
            confidence
        )
    }

    fn parse_confidence(text: &str) -> f64 {
        text.lines()
            .find_map(|line| {
                line.trim()
                    .trim_start_matches(['*', '#', ' '])
                    .strip_prefix("CONFIDENCE:")
                    .and_then(|val| val.trim().trim_end_matches('*').trim().parse::<f64>().ok())
            })
            .unwrap_or(0.75)
            .clamp(0.0, 1.0)
    }
}

/// Evaluation task for MARL (General-Sum: Common Reward + Competitive Bonus/Penalty)
/// With support for cross-writing (sync) to all agent G-RAG memories after each round.
pub struct MarlConsensusEvaluatorNode {
    pub architect_rag: Arc<Mutex<PersonalGraphRag>>,
    pub systems_rag: Arc<Mutex<PersonalGraphRag>>,
    pub critic_rag: Arc<Mutex<PersonalGraphRag>>,
}

#[async_trait]
impl Task for MarlConsensusEvaluatorNode {
    fn id(&self) -> &str {
        "evaluator"
    }

    async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
        let mut state: AgonSwarmState = context.get("swarm_state").ok_or_else(|| {
            GraphError::TaskExecutionFailed("Missing swarm_state in context".into())
        })?;

        let avg_confidence: f64 = state.proposals.values().map(|p| p.confidence).sum::<f64>()
            / (state.proposals.len().max(1) as f64);
        let common_reward = if avg_confidence >= 0.80 { 1.0 } else { 0.2 };

        let mut best_agent = String::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_content = String::new();

        for (name, prop) in &state.proposals {
            let net_score = prop.confidence - prop.critique_penalties;
            if net_score > best_score {
                best_score = net_score;
                best_agent = name.clone();
                best_content = prop.content.clone();
            }
        }

        // Cross-entry: All proposals from this round are stored in the memories of ALL agents.
        for (name, prop) in &state.proposals {
            let entity_name = format!("Consensus_Ep{}_{}", state.episode, name);
            let entity_desc = prop.content.chars().take(300).collect::<String>();

            for rag_mutex in [&self.architect_rag, &self.systems_rag, &self.critic_rag] {
                let mut rag = rag_mutex.lock().unwrap();
                rag.upsert_entity(&entity_name, "SwarmConsensus", &entity_desc);
            }
        }

        let joint_actions: Vec<(String, SwarmAction)> = state
            .proposals
            .iter()
            .map(|(name, p)| (name.clone(), p.action_taken))
            .collect();

        let prev_phase = state.current_phase.clone();
        let next_phase = if best_score >= 0.85 {
            state.consensus_reached = true;
            state.final_agon_spec = best_content;
            "consensus_complete".to_string()
        } else {
            "refinement_round".to_string()
        };
        state.current_phase = next_phase.clone();

        for (name, prop) in &state.proposals {
            let competitive_delta = if name == &best_agent {
                0.75
            } else if prop.critique_penalties > 0.0 {
                -0.5
            } else {
                0.0
            };

            let total_agent_reward = common_reward + competitive_delta;

            if let Some(engine) = state.marl_engines.get_mut(name) {
                engine.update_policy(
                    &prev_phase,
                    prop.action_taken,
                    &joint_actions,
                    total_agent_reward,
                    &next_phase,
                );
            }
        }

        context.set("swarm_state", state)?;

        Ok(TaskResult::new(
            Some(format!("MARL evaluation complete. Round winner: {}", best_agent)),
            NextAction::End,
        ))
    }
}
pub mod marl;
pub mod memory;
pub mod tools;
pub mod workflow;

pub use marl::{MarlPolicyEngine, SwarmAction};
pub use memory::{AgonIntentMessage, GraphEntity, GraphRelation, PersonalGraphRag, ReasoningStep};
pub use tools::{ArxivArgs, ArxivSearchTool, GraphQueryArgs, GraphRagQueryTool, SwarmToolError};
pub use workflow::{AgentProposal, AgonSwarmNode, AgonSwarmState, MarlConsensusEvaluatorNode};

use graph_flow::{
    ExecutionStatus, FlowRunner, Graph, GraphBuilder, GraphError, InMemorySessionStorage, Session,
    SessionStorage,
};
use std::sync::{Arc, Mutex};

/// Id of the first task of the swarm graph.
pub const START_TASK: &str = "architect";

/// Helper function for constructing the production workflow graph for AGON Swarm
pub fn build_agon_swarm_graph(
    architect_rag: Arc<Mutex<PersonalGraphRag>>,
    systems_rag: Arc<Mutex<PersonalGraphRag>>,
    critic_rag: Arc<Mutex<PersonalGraphRag>>,
) -> Arc<Graph> {
    let node_architect = Arc::new(AgonSwarmNode {
        task_id: "architect".to_string(),
        agent_name: "ArchitectAgent".to_string(),
        role_preamble: "You are ArchitectAgent. You are designing the type system for the AGON language (Intents, Propositions, Confidence). Make use of arXiv and your G-RAG. Call tools at most once, then immediately write your final response ending with CONFIDENCE: 0.XX".to_string(),
        personal_rag: Arc::clone(&architect_rag), // Here we use Arc::clone
    });

    let node_systems = Arc::new(AgonSwarmNode {
        task_id: "systems".to_string(),
        agent_name: "SystemsAgent".to_string(),
        role_preamble: "You are SystemsAgent. You are porting AGON to high-performance Rust (Tokio actors, zero-copy, rig-core). You are competing for the most efficient architecture. Call tools at most once, then immediately write your final response ending with CONFIDENCE: 0.XX".to_string(),
        personal_rag: Arc::clone(&systems_rag), // Arc::clone here as well
    });

    let node_critic = Arc::new(AgonSwarmNode {
        task_id: "critic".to_string(),
        agent_name: "GameTheoryCritic".to_string(),
        role_preamble: "You are a Game Theory Critic. You audit colleagues' proposals using game theory (Nash/QMIX/Regret) and consensus protocols. Look for flaws and compete for bonus points. Call tools at most once, then immediately write your final response ending with CONFIDENCE: 0.XX".to_string(),
        personal_rag: Arc::clone(&critic_rag), // Arc::clone here as well
    });

    let node_evaluator = Arc::new(MarlConsensusEvaluatorNode {
        architect_rag, // The final move into the evaluator takes place here
        systems_rag,
        critic_rag,
    });

    // Sequential pass through an episode: Architect -> Systems -> Critic -> MARL Evaluator
    let graph = GraphBuilder::new("agon_marl_swarm_graph")
        .add_task(node_architect)
        .add_task(node_systems)
        .add_task(node_critic)
        .add_task(node_evaluator)
        .add_edge("architect", "systems")
        .add_edge("systems", "critic")
        .add_edge("critic", "evaluator")
        .set_start_task(START_TASK)
        .build()
        .expect("AGON swarm graph is statically valid");

    Arc::new(graph)
}

/// Runs one full episode (architect -> systems -> critic -> evaluator) and returns the updated state.
///
/// `FlowRunner::run` executes exactly one task per call, so we loop until the workflow completes.
pub async fn run_episode(
    graph: Arc<Graph>,
    session_id: &str,
    state: AgonSwarmState,
) -> graph_flow::Result<AgonSwarmState> {
    let storage = Arc::new(InMemorySessionStorage::new());
    let runner = FlowRunner::new(graph, storage.clone());

    let session = Session::new_from_task(session_id.to_string(), START_TASK);
    session.context.set("swarm_state", state)?;
    storage.save(session).await?;

    loop {
        let result = runner.run(session_id).await?;
        match result.status {
            ExecutionStatus::Completed => break,
            ExecutionStatus::Paused { .. } => continue,
            ExecutionStatus::WaitingForInput => break,
        }
    }

    let session = storage
        .get(session_id)
        .await?
        .ok_or_else(|| GraphError::SessionNotFound(session_id.to_string()))?;
    session
        .context
        .get::<AgonSwarmState>("swarm_state")
        .ok_or_else(|| GraphError::TaskExecutionFailed("swarm_state missing after run".into()))
}
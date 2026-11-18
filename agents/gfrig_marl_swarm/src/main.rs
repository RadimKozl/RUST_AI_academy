use gfrig_marl_swarm::{
    AgonSwarmState, MarlPolicyEngine, PersonalGraphRag, build_agon_swarm_graph, run_episode,
};
use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    // 1. Loading the AGON language specification from the file task.txt
    let task_content = fs::read_to_string("task.txt").unwrap_or_else(|_| {
        "Design the AGON language (featuring Semantic Type Safety, the Actor Model, Native MARL, and Consensus) and its compilation into Rust.".to_string()
    });

    // 2. Initialization of isolated G-RAG memories for each of the 3 agents
    let architect_rag = Arc::new(Mutex::new(PersonalGraphRag::new("ArchitectAgent")));
    let systems_rag = Arc::new(Mutex::new(PersonalGraphRag::new("SystemsAgent")));
    let critic_rag = Arc::new(Mutex::new(PersonalGraphRag::new("GameTheoryCritic")));

    // Pre-loading domain knowledge into custom graphs (Long-Term Memory)
    {
        let mut r1 = architect_rag.lock().unwrap();
        r1.upsert_entity("SemanticTypeSafety", "AgonPillar", "Messages are typed Intents and Propositions with a confidence in the range [0.0, 1.0].");
        r1.upsert_entity("AgonAST", "CompilerModule", "Abstract syntax tree for type validation prior to compilation to Rust.");
        r1.add_relation("SemanticTypeSafety", "ENFORCED_BY", "AgonAST", 1.0);

        let mut r2 = systems_rag.lock().unwrap();
        r2.upsert_entity("ActorModelTokio", "RuntimePillar", "Asynchronous actors in Rust with allocation-free mpsc messaging.");
        r2.upsert_entity("RigIntegration", "BackendAdapter", "Integrating an LLM as a semantic translator into AGON structures via rig-core.");
        r2.add_relation("ActorModelTokio", "INTEGRATES_WITH", "RigIntegration", 0.95);

        let mut r3 = critic_rag.lock().unwrap();
        r3.upsert_entity("QMIX_CTDE", "MarlAlgorithm", "Monotonic decomposition of Q_tot for solving credit assignment in a swarm.");
        r3.upsert_entity("MajorityVoteConsensus", "ConsensusProtocol", "Weighted confidence score voting with a 500ms timeout.");
        r3.add_relation("QMIX_CTDE", "OPTIMIZES", "MajorityVoteConsensus", 0.9);
    }

    // 3. Workflow graph in `graph-flow`
    let graph = build_agon_swarm_graph(
        Arc::clone(&architect_rag),
        Arc::clone(&systems_rag),
        Arc::clone(&critic_rag),
    );

    // 4. Initialization of MARL policies (JAL-AM + Regret Matching)
    let mut marl_engines = HashMap::new();
    marl_engines.insert("ArchitectAgent".to_string(), MarlPolicyEngine::new("ArchitectAgent"));
    marl_engines.insert("SystemsAgent".to_string(), MarlPolicyEngine::new("SystemsAgent"));
    marl_engines.insert("GameTheoryCritic".to_string(), MarlPolicyEngine::new("GameTheoryCritic"));

    let mut shared_state = AgonSwarmState {
        episode: 0,
        step: 0,
        task_specification: task_content,
        current_phase: "init".to_string(),
        short_term_messages: Vec::new(),
        proposals: HashMap::new(),
        marl_engines,
        consensus_reached: false,
        final_agon_spec: String::new(),
    };

    // 5. MARL training loop across multiple episodes
    let total_episodes = 12;
    for ep in 1..=total_episodes {
        println!("\n==================== MARL EPISODE {} ====================", ep);
        shared_state.episode = ep;
        shared_state.proposals.clear();

        shared_state = run_episode(
            Arc::clone(&graph),
            &format!("agon_session_ep_{}", ep),
            shared_state,
        )
        .await?;

        for (agent_name, engine) in &shared_state.marl_engines {
            println!(
                "Agent: {:<18} | Cumulative MARL reward: {:>6.2} | Epsilon: {:.3}",
                agent_name, engine.total_reward, engine.epsilon
            );
        }

        if shared_state.consensus_reached {
            println!(
                "\n>>> SWARM CONSENSUS REACHED IN EPISODE {} <<<\nWinning AGON specification:\n{}",
                ep, shared_state.final_agon_spec
            );

            // Automatic writing to a file
            std::fs::write("agon_final_spec.txt", &shared_state.final_agon_spec)
                .expect("Cannot write the final specification to the file.");
        }
    }

    // 6. Dump of the final state of isolated G-RAG memories
    println!("\n==================== STATUS OF PERSONAL G-RAG MEMORIES ====================");
    for rag in [&architect_rag, &systems_rag, &critic_rag] {
        let guard = rag.lock().unwrap();
        println!(
            "Agent: {:<18} | Graph nodes: {:>2} | Edges: {:>2} | Reasoning steps: {:>2}",
            guard.owner_agent,
            guard.entities.len(),
            guard.relations.len(),
            guard.reasoning_trace.len()
        );
    }

    Ok(())
}
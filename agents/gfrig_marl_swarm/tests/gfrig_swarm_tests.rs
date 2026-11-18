use gfrig_marl_swarm::{
    build_agon_swarm_graph, run_episode, AgentProposal, AgonSwarmState, GraphQueryArgs,
    GraphRagQueryTool, MarlConsensusEvaluatorNode, MarlPolicyEngine, PersonalGraphRag, SwarmAction,
};
use graph_flow::{Context, Task};
use rig::tool::{Tool, ToolContext};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// 1. Test of isolated GraphRAG memory (vector similarity + 1-hop graph expansion)
#[test]
fn test_personal_graph_rag_retrieval_and_isolation() {
    let mut architect_rag = PersonalGraphRag::new("ArchitectAgent");
    let systems_rag = PersonalGraphRag::new("SystemsAgent");

    architect_rag.upsert_entity(
        "SemanticTypeSafety",
        "AgonPillar",
        "Strongly typed Intents and Propositions with confidence score [0.0, 1.0].",
    );
    architect_rag.upsert_entity(
        "AgonAST",
        "CompilerModule",
        "Abstract syntax tree enforcing confidence bounds before Rust codegen.",
    );
    architect_rag.add_relation("SemanticTypeSafety", "ENFORCED_BY", "AgonAST", 1.0);

    // Verification of the lookup for a node and its 1-hop neighbor
    let result = architect_rag.query_graph_rag("SemanticTypeSafety Intents confidence", 2);
    assert!(result.contains("SemanticTypeSafety"));
    assert!(result.contains("ENFORCED_BY"));
    assert!(result.contains("AgonAST"));

    // Verification of strict memory isolation between agents
    let empty_result = systems_rag.query_graph_rag("SemanticTypeSafety", 2);
    assert!(empty_result.contains("Paměť je zatím prázdná"));
}

/// 2. Testing the MARL engine (JAL-AM opponent modeling, TD Q-update, Regret Matching, and epsilon decay)
#[test]
fn test_marl_policy_engine_jal_am_and_regret_update() {
    let mut engine = MarlPolicyEngine::new("ArchitectAgent");
    let initial_epsilon = engine.epsilon;

    let joint_actions = vec![
        ("ArchitectAgent".to_string(), SwarmAction::ProposeSpecification),
        ("SystemsAgent".to_string(), SwarmAction::DeepArxivResearch),
        ("GameTheoryCritic".to_string(), SwarmAction::CritiqueAndCompete),
    ];

    engine.update_policy(
        "init",
        SwarmAction::ProposeSpecification,
        &joint_actions,
        1.75, // Common reward (1.0) + Competitive winner bonus (0.75)
        "consensus_complete",
    );

    // Verification of total reward accumulation and epsilon decay
    assert!((engine.total_reward - 1.75).abs() < 1e-6);
    assert!(engine.epsilon < initial_epsilon);

    // Q-value calculation for the action taken: Q = 0 + 0.15 * (1.75 + 0.95 * 0) = 0.2625
    let q_val = engine
        .q_values
        .get(&("init".to_string(), SwarmAction::ProposeSpecification))
        .copied()
        .unwrap_or(0.0);
    assert!((q_val - 0.2625).abs() < 1e-6);

    // Verification of JAL-AM action counters for other agents (own actions are not included in opponent_action_counts)
    assert_eq!(engine.opponent_action_counts.len(), 2);
    assert_eq!(
        engine.opponent_action_counts.get(&(
            "init".to_string(),
            "SystemsAgent".to_string(),
            SwarmAction::DeepArxivResearch
        )),
        Some(&1)
    );
    assert_eq!(
        engine.opponent_action_counts.get(&(
            "init".to_string(),
            "ArchitectAgent".to_string(),
            SwarmAction::ProposeSpecification
        )),
        None
    );

    // Verification of the selection of a valid action
    let next_action = engine.select_action("init");
    assert!(SwarmAction::all().contains(&next_action));
}

/// 3. Test Rig for the `GraphRag Query Tool` for querying an agent's private graph
#[tokio::test]
async fn test_rig_graph_rag_query_tool() {
    let rag = Arc::new(Mutex::new(PersonalGraphRag::new("GameTheoryCritic")));
    {
        let mut guard = rag.lock().unwrap();
        guard.upsert_entity(
            "QMIX_CTDE",
            "MarlAlgorithm",
            "Monotonic value decomposition Q_tot for credit assignment.",
        );
    }

    let tool = GraphRagQueryTool {
        personal_memory: Arc::clone(&rag),
    };

    let mut tool_ctx = ToolContext::new();
    let output = tool
        .call(
            &mut tool_ctx,
            GraphQueryArgs {
                concept: "QMIX credit assignment".to_string(),
            },
        )
        .await
        .expect("Tool call should succeed");

    assert!(output.contains("QMIX_CTDE"));
    assert!(output.contains("Monotonic value decomposition"));
}

/// 4. Test of the MARL Consensus Evaluator (cooperative reward + competitive bonus and a penalty of -0.5)
#[tokio::test]
async fn test_marl_consensus_evaluator_general_sum_payoffs() {
    let mut proposals = HashMap::new();
    proposals.insert(
        "ArchitectAgent".to_string(),
        AgentProposal {
            agent_name: "ArchitectAgent".to_string(),
            action_taken: SwarmAction::ProposeSpecification,
            content: "Spec A: Typed Intent<T>".to_string(),
            confidence: 0.78,
            critique_penalties: 0.25, // Attacked by a critic -> net score 0.53
        },
    );
    proposals.insert(
        "SystemsAgent".to_string(),
        AgentProposal {
            agent_name: "SystemsAgent".to_string(),
            action_taken: SwarmAction::ProposeSpecification,
            content: "Spec B: Tokio Actor Runtime".to_string(),
            confidence: 0.92,
            critique_penalties: 0.0, // Round winner -> net score 0.92
        },
    );
    proposals.insert(
        "GameTheoryCritic".to_string(),
        AgentProposal {
            agent_name: "GameTheoryCritic".to_string(),
            action_taken: SwarmAction::CritiqueAndCompete,
            content: "Critique & Nash check".to_string(),
            confidence: 0.85,
            critique_penalties: 0.0,
        },
    );

    let mut marl_engines = HashMap::new();
    marl_engines.insert("ArchitectAgent".to_string(), MarlPolicyEngine::new("ArchitectAgent"));
    marl_engines.insert("SystemsAgent".to_string(), MarlPolicyEngine::new("SystemsAgent"));
    marl_engines.insert(
        "GameTheoryCritic".to_string(),
        MarlPolicyEngine::new("GameTheoryCritic"),
    );

    let initial_state = AgonSwarmState {
        episode: 1,
        step: 3,
        task_specification: "AGON Language Spec".to_string(),
        current_phase: "init".to_string(),
        short_term_messages: Vec::new(),
        proposals,
        marl_engines,
        consensus_reached: false,
        final_agon_spec: String::new(),
    };

    let ctx = Context::new();
    ctx.set("swarm_state", initial_state).unwrap();

    // Create test RAG memories for the evaluator in the test
    let eval_architect_rag = Arc::new(Mutex::new(PersonalGraphRag::new("ArchitectAgent")));
    let eval_systems_rag = Arc::new(Mutex::new(PersonalGraphRag::new("SystemsAgent")));
    let eval_critic_rag = Arc::new(Mutex::new(PersonalGraphRag::new("GameTheoryCritic")));

    let evaluator = MarlConsensusEvaluatorNode {
        architect_rag: eval_architect_rag,
        systems_rag: eval_systems_rag,
        critic_rag: eval_critic_rag,
    };
    
    let res = evaluator
        .run(ctx.clone())
        .await
        .expect("Evaluator node should succeed");
    assert!(res.response.as_deref().unwrap_or_default().contains("SystemsAgent"));

    let updated: AgonSwarmState = ctx.get("swarm_state").unwrap();
    assert!(updated.consensus_reached);
    assert_eq!(updated.final_agon_spec, "Spec B: Tokio Actor Runtime");
    assert_eq!(updated.current_phase, "consensus_complete");

    // Average confidence = (0.78 + 0.92 + 0.85) / 3 = 0.85 >= 0.80 => common_reward = 1.0
    // SystemsAgent (winner): 1.0 + 0.75 = 1.75
    // ArchitectAgent (penalized): 1.0 - 0.50 = 0.50
    // GameTheoryCritic (neutral): 1.0 + 0.0 = 1.00
    assert!((updated.marl_engines["SystemsAgent"].total_reward - 1.75).abs() < 1e-6);
    assert!((updated.marl_engines["ArchitectAgent"].total_reward - 0.50).abs() < 1e-6);
    assert!((updated.marl_engines["GameTheoryCritic"].total_reward - 1.00).abs() < 1e-6);
}

/// 5. End-to-end integration test of the entire `graph-flow` workflow with 3 agents (without an external LLM key)
#[tokio::test]
async fn test_full_agon_swarm_graph_episode_execution() {
    unsafe { std::env::set_var("AGON_LLM", "off") };

    let architect_rag = Arc::new(Mutex::new(PersonalGraphRag::new("ArchitectAgent")));
    let systems_rag = Arc::new(Mutex::new(PersonalGraphRag::new("SystemsAgent")));
    let critic_rag = Arc::new(Mutex::new(PersonalGraphRag::new("GameTheoryCritic")));

    let graph = build_agon_swarm_graph(
        Arc::clone(&architect_rag),
        Arc::clone(&systems_rag),
        Arc::clone(&critic_rag),
    );

    let mut marl_engines = HashMap::new();
    marl_engines.insert("ArchitectAgent".to_string(), MarlPolicyEngine::new("ArchitectAgent"));
    marl_engines.insert("SystemsAgent".to_string(), MarlPolicyEngine::new("SystemsAgent"));
    marl_engines.insert(
        "GameTheoryCritic".to_string(),
        MarlPolicyEngine::new("GameTheoryCritic"),
    );

    // We pre-prepare strong proposals directly in the initial state, ensuring the evaluator can reliably reach a consensus (score >= 0.85)
    let mut proposals = HashMap::new();
    proposals.insert(
        "SystemsAgent".to_string(),
        AgentProposal {
            agent_name: "SystemsAgent".to_string(),
            action_taken: SwarmAction::ProposeSpecification,
            content: "Winning Proposal for AGON: Strongly typed Intent<T> with Tokio Actor runtime.".to_string(),
            confidence: 0.92,
            critique_penalties: 0.0,
        },
    );

    let state = AgonSwarmState {
        episode: 1,
        step: 0,
        task_specification: "Design AGON Intent<T> and Tokio Actor runtime.".to_string(),
        current_phase: "init".to_string(),
        short_term_messages: Vec::new(),
        proposals,
        marl_engines,
        consensus_reached: false,
        final_agon_spec: String::new(),
    };

    let final_state = run_episode(graph, "test_ep_1", state)
        .await
        .expect("Graph execution should complete cleanly");

    assert_eq!(final_state.step, 3);
    assert!(final_state.consensus_reached);
    assert!(final_state.final_agon_spec.contains("Intent<T>"));

    for rag in [&architect_rag, &systems_rag, &critic_rag] {
        let guard = rag.lock().unwrap();
        assert!(!guard.reasoning_trace.is_empty());
    }
}
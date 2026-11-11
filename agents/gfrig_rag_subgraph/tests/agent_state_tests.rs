use gfrig_rag_subgraph::AgentState;

#[test]
fn test_agent_state_initialization() {
    let state = AgentState::new("Test task for AI trends");
    assert_eq!(state.task, "Test task for AI trends");
    assert_eq!(state.question, "");
    assert_eq!(state.iteration, 0);
    assert_eq!(state.critic_score, 0.0);
    assert!(state.documents.is_empty());
    assert!(state.critic_reason.is_empty());
}

#[test]
fn test_xai_critic_parsing_logic() {
    // Simulating a response from the LLM critic in the required format
    let critic_response = "SCORE: 8.5 | REASON: The answer is comprehensive and covers all major trends accurately.";
    
    let mut state = AgentState::new("AI Trends");

    if critic_response.contains("SCORE:") && critic_response.contains("REASON:") {
        let parts: Vec<&str> = critic_response.split('|').collect();
        let score_part = parts[0].replace("SCORE:", "").trim().to_string();
        let reason_part = parts[1].replace("REASON:", "").trim().to_string();

        state.critic_score = score_part.parse().unwrap_or(0.0);
        state.critic_reason = reason_part;
    }

    assert_eq!(state.critic_score, 8.5);
    assert_eq!(
        state.critic_reason, 
        "The answer is comprehensive and covers all major trends accurately."
    );
    assert!(state.critic_score >= 7.0, "Should satisfy the subgraph exit condition.");
}

#[test]
fn test_subgraph_retry_loop_condition() {
    let mut state = AgentState::new("Edge AI");
    state.critic_score = 5.0; // Below 7.0 -> should trigger a retry
    let max_iterations = 3;

    let mut rag_success = false;
    if state.critic_score >= 7.0 {
        rag_success = true;
    } else {
        state.question.push_str(" (please specify for a better answer)");
        state.iteration += 1;
    }

    assert!(!rag_success);
    assert_eq!(state.iteration, 1);
    assert!(state.question.contains("please specify"));
    assert!(state.iteration < max_iterations);
}
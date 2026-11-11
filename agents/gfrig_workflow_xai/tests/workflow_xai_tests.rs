use gfrig_workflow_xai::{AgentState, SqliteStorageManager, WorkflowGraph, XaiStep};

#[tokio::test]
async fn test_workflow_graph_execution() {
    let mut graph = WorkflowGraph::new();

    // Register test node 1
    graph.add_node("node_alpha", |state| {
        Box::pin(async move {
            state.improved_query = format!("Optimized: {}", state.user_input);
            state.xai_audit_trail.push(XaiStep {
                node_name: "node_alpha".to_string(),
                input_data: state.user_input.clone(),
                generated_output: state.improved_query.clone(),
                reasoning: "Test reasoning alpha".to_string(),
            });
            Ok(())
        })
    });

    // Register test node 2
    graph.add_node("node_beta", |state| {
        Box::pin(async move {
            state.final_answer = format!("Result for [{}]", state.improved_query);
            state.xai_audit_trail.push(XaiStep {
                node_name: "node_beta".to_string(),
                input_data: state.improved_query.clone(),
                generated_output: state.final_answer.clone(),
                reasoning: "Test reasoning beta".to_string(),
            });
            Ok(())
        })
    });

    let mut state = AgentState::new("test_thread_999", "Edge AI");

    // Launching the graph
    let result = graph.invoke(&mut state).await;
    assert!(result.is_ok(), "Graph execution should succeed");

    // Verification of outputs
    assert_eq!(state.improved_query, "Optimized: Edge AI");
    assert_eq!(state.final_answer, "Result for [Optimized: Edge AI]");
    assert_eq!(state.xai_audit_trail.len(), 2);
    assert_eq!(state.xai_audit_trail[0].node_name, "node_alpha");
    assert_eq!(state.xai_audit_trail[1].node_name, "node_beta");
}

#[test]
fn test_sqlite_storage_manager_xai() {
    let db_path = "test_workflow_xai.db";
    let thread_id = "test_session_sqlite";

    let storage = SqliteStorageManager::new(db_path).unwrap();

    let step = XaiStep {
        node_name: "prepare_query".to_string(),
        input_data: "AI trends".to_string(),
        generated_output: "AI trends 2026 deployment".to_string(),
        reasoning: "Expanded query with temporal and contextual keywords.".to_string(),
    };

    // Saving the XAI step to SQLite
    let save_res = storage.save_xai_step(thread_id, &step);
    assert!(save_res.is_ok(), "Saving XAI step to SQLite must succeed");

    // Cleanup after test
    let _ = std::fs::remove_file(db_path);
}
use gfrig_router_node::{run_graph_workflow, NodeMode, UnifiedNode};
use std::sync::Arc;

#[tokio::test]
async fn test_llm_router_conditional_execution() {
    let user_question = "What is artificial intelligence?";

    // 1. Start node A
    let node_a = UnifiedNode {
        id_str: "node_a".to_string(),
        label: "Node A".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node A".to_string(),
            logic: Arc::new(|_ctx| {
                Box::pin(async move { Ok("Started execution A".to_string()) })
            }),
        },
    };

    // 2. Router (Selects path 'b')
    let node_router = UnifiedNode {
        id_str: "node_router".to_string(),
        label: "Router".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Router decision".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    ctx.set("router_decision", "b".to_string())?;
                    Ok("Router selected path 'b'".to_string())
                })
            }),
        },
    };

    // 3. Branch B (Active branch)
    let node_b = UnifiedNode {
        id_str: "node_b".to_string(),
        label: "Node B".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node B".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    let decision = ctx.get::<String>("router_decision").unwrap_or_default();
                    if decision.contains("b") {
                        Ok("Executed Branch B".to_string())
                    } else {
                        Ok("Skipped Branch B".to_string())
                    }
                })
            }),
        },
    };

    // 4. End node D
    let node_d = UnifiedNode {
        id_str: "node_d".to_string(),
        label: "Node D".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node D".to_string(),
            logic: Arc::new(|_ctx| {
                Box::pin(async move { Ok("Joined at D".to_string()) })
            }),
        },
    };

    // Topology leading through active branch B
    let nodes = vec![node_a, node_router, node_b, node_d];
    let edges = vec![
        ("node_a", "node_router"),
        ("node_router", "node_b"),
        ("node_b", "node_d"),
    ];

    // Running the workflow test
    let result = run_graph_workflow(user_question, nodes, edges, "node_a").await;
    assert!(result.is_ok(), "Conditional workflow execution failed");

    let state = result.unwrap();

    // 1. Verification that the correct branch B and node D were processed
    assert!(state.data.iter().any(|item| item.contains("Executed Branch B")), "Branch B should be executed");
    assert!(state.data.iter().any(|item| item.contains("Joined at D")), "Workflow should reach node D");

    // 2. Verification of the Mermaid diagram
    assert!(state.architecture_mermaid.contains("node_router --> node_b"));
    assert!(state.architecture_mermaid.contains("node_b --> node_d"));

    // 3. Verification of the XAI Audit Trail (A, Router, B, D = 4 steps)
    assert_eq!(state.audit_trail.len(), 4, "Audit trail must record all executed node steps");
}
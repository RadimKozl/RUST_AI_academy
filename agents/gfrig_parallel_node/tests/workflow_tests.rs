use gfrig_parallel_node::{run_graph_workflow, NodeMode, UnifiedNode};
use std::sync::Arc;

#[tokio::test]
async fn test_fork_join_parallel_execution() {
    let user_query = "Test Fork-Join synchronization.";

    let node_a = UnifiedNode {
        id_str: "node_a".to_string(),
        label: "Node A".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node A".to_string(),
            logic: Arc::new(|_ctx| {
                Box::pin(async move { Ok("Data from A".to_string()) })
            }),
        },
    };

    let node_b = UnifiedNode {
        id_str: "node_b".to_string(),
        label: "Node B".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node B".to_string(),
            logic: Arc::new(|_ctx| {
                Box::pin(async move { Ok("Data from B".to_string()) })
            }),
        },
    };

    let node_c = UnifiedNode {
        id_str: "node_c".to_string(),
        label: "Node C".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node C".to_string(),
            logic: Arc::new(|_ctx| {
                Box::pin(async move { Ok("Data from C".to_string()) })
            }),
        },
    };

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

    let nodes = vec![node_a, node_b, node_c, node_d];
    let edges = vec![
        ("node_a", "node_b"),
        ("node_a", "node_c"),
        ("node_b", "node_d"),
        ("node_c", "node_d"),
    ];

    let result = run_graph_workflow(user_query, nodes, edges, "node_a").await;
    assert!(result.is_ok(), "Fork-Join execution should succeed");
    
    let state = result.unwrap();

    // Robust verification that the graph execution took place and the data contains key nodes
    assert!(state.data.len() >= 3, "Workflow must accumulate at least 3 steps");
    assert!(state.data.iter().any(|item| item.contains("Node A")));
    assert!(state.data.iter().any(|item| item.contains("Node D")));

    // Verification of the correct topology in the Mermaid diagram
    assert!(state.architecture_mermaid.contains("node_a --> node_b"));
    assert!(state.architecture_mermaid.contains("node_a --> node_c"));
    assert!(state.architecture_mermaid.contains("node_b --> node_d"));
    assert!(state.architecture_mermaid.contains("node_c --> node_d"));
}
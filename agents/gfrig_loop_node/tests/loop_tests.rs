use gfrig_loop_node::{run_loop_workflow, NodeMode, UnifiedNode};
use std::sync::Arc;

#[tokio::test]
async fn test_loop_workflow_execution() {
    let user_question = "Test loop workflow execution.";
    let user_query = "Run test loop.";

    let node_a = UnifiedNode {
        id_str: "node_a".to_string(),
        label: "Node A".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node A".to_string(),
            logic: Arc::new(|_ctx| {
                Box::pin(async move { Ok("Started A".to_string()) })
            }),
        },
    };

    let node_b = UnifiedNode {
        id_str: "node_b".to_string(),
        label: "Node B".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node B".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    let counter = ctx.get::<i32>("counter").unwrap_or_default();
                    Ok(format!("Worker B iteration {}", counter + 1))
                })
            }),
        },
    };

    let node_c = UnifiedNode {
        id_str: "node_c".to_string(),
        label: "Node C".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node C".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    let mut counter = ctx.get::<i32>("counter").unwrap_or_default();
                    counter += 1;
                    ctx.set("counter", counter)?;
                    Ok(format!("Incremented counter to {}", counter))
                })
            }),
        },
    };

    let node_router = UnifiedNode {
        id_str: "node_router".to_string(),
        label: "Router".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Router".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    let counter = ctx.get::<i32>("counter").unwrap_or_default();
                    if counter < 3 {
                        ctx.set("router_decision", "b".to_string())?; // Fixed: added a parenthesis
                        Ok(format!("Looping back, counter: {}", counter))
                    } else {
                        ctx.set("router_decision", "d".to_string())?;
                        Ok(format!("Threshold reached, exiting, counter: {}", counter))
                    }
                })
            }),
        },
    };

    let node_d = UnifiedNode {
        id_str: "node_d".to_string(),
        label: "Node D".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Node D".to_string(),
            logic: Arc::new(|_ctx| {
                Box::pin(async move { Ok("Finished at D".to_string()) })
            }),
        },
    };

    let nodes = vec![node_a, node_b, node_c, node_router, node_d];
    let edges = vec![
        ("node_a", "node_b"),
        ("node_b", "node_c"),
        ("node_c", "node_router"),
        ("node_router", "node_d"),
    ];

    let result = run_loop_workflow(user_query, user_question, nodes, edges, "node_a", 3).await;
    assert!(result.is_ok(), "Loop workflow should execute successfully");

    let state = result.unwrap();

    // Verification that the final counter reached the value 3
    assert_eq!(state.counter, 3, "Counter must reach exactly 3 iterations");

    // Verification that the data contains the executed loop steps
    assert!(state.data.iter().any(|item| item.contains("Worker B")));
    assert!(state.data.iter().any(|item| item.contains("Finished at D")));
}
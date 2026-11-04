use gfrig_sequential_node::{run_dynamic_workflow, NodeMode};
use std::sync::Arc;

#[tokio::test]
async fn test_router_and_dispatcher_workflow() {
    let user_query = "Run calculator";

    // In the test, we can simulate the router's behavior by directly setting router_decision,
    // or we can test CustomTool, which reads the router_decision key from the context.
    let workflow_steps = vec![
        (
            "setup_decision".to_string(),
            NodeMode::CustomTool {
                action_name: "Mock Router Decision".to_string(),
                logic: Arc::new(|ctx| {
                    Box::pin(async move {
                        // We simulate that the router has decided to launch the calculator
                        ctx.set("router_decision", "run_calculator".to_string()).unwrap();
                        Ok("Router decision set to run_calculator".to_string())
                    })
                }),
            },
        ),
        (
            "dispatcher".to_string(),
            NodeMode::CustomTool {
                action_name: "Conditional Dispatcher".to_string(),
                logic: Arc::new(|ctx| {
                    Box::pin(async move {
                        // Read the decision stored in the context
                        let decision = ctx.get::<String>("router_decision").unwrap_or_default();
                        
                        match decision.as_str() {
                            "run_calculator" => {
                                Ok("Dispatcher executed: Calculator tool (2 + 2 = 4)".to_string())
                            }
                            _ => {
                                Ok("Dispatcher executed default".to_string())
                            }
                        }
                    })
                }),
            },
        ),
    ];

    let result = run_dynamic_workflow(user_query, workflow_steps).await;
    
    assert!(result.is_ok(), "Workflow execution should succeed");
    let state = result.unwrap();

    // Verification of status and dynamic routing
    assert_eq!(state.query, user_query);
    assert_eq!(state.data.len(), 2);
    assert!(state.data[1].contains("Calculator tool (2 + 2 = 4)"));

    // Verification of the Mermaid diagram and audit trail
    assert!(state.architecture_mermaid.contains("graph TD"));
    assert!(state.architecture_mermaid.contains("node_setup_decision --> node_dispatcher"));
    assert_eq!(state.audit_trail.len(), 2);
    assert_eq!(state.audit_trail[0].task_id, "node_setup_decision");
    assert_eq!(state.audit_trail[1].task_id, "node_dispatcher");
}
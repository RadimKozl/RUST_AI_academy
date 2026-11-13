use gfrig_react_scientific_agent::tools::{SymbolicaArgs, SymbolicaTool};
use gfrig_react_scientific_agent::{ScientificGraphWorkflow, XaiTracker};
use rig::tool::{Tool, ToolContext};

#[test]
fn test_graph_state_initialization() {
    let query = "Test query for scientific agent";
    let workflow = ScientificGraphWorkflow::new(query);
    let state = workflow.get_state();

    assert_eq!(state.input_query, query);
    assert!(state.execution_trace.is_empty());
    assert!(state.final_output.is_empty());
    assert!(state.xai_report.is_none());
}

#[tokio::test]
async fn test_symbolica_tool_execution() {
    let tracker = XaiTracker::new();
    let tool = SymbolicaTool { tracker: tracker.clone() };
    let args = SymbolicaArgs {
        rationale: "Testing polynomial expansion and roots".to_string(),
        expression: "(x + 1)^2".to_string(),
    };

    let result = tool.call(&mut ToolContext::new(), args).await;
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.contains("Expanded: x^2 + 2*x + 1"));

    let report = tracker.build_report("test", "mock", "none", &output, std::time::Instant::now());
    assert_eq!(report.tool_decisions.len(), 1);
    assert_eq!(report.tool_decisions[0].tool_name, "symbolica_calculate");
    assert_eq!(report.tool_decisions[0].decision_rationale, "Testing polynomial expansion and roots");
}

#[ignore = "requires running Ollama"]
#[tokio::test]
async fn test_full_scientific_workflow() {
    let mut workflow = ScientificGraphWorkflow::new("Evaluate x + 2");
    let result = workflow.run().await;

    assert!(result.is_ok());

    let state = workflow.get_state();
    assert!(!state.execution_trace.is_empty());
    assert!(state.xai_report.is_some());
}
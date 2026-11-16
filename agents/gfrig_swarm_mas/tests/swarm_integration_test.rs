// tests/swarm_integration_test.rs
use gfrig_swarm_mas::graph_swarm::{
    take_last_swarm_action, DelegatorArgs, DelegatorTool, CompleteArgs, CompleteTool, SwarmAction
};
use gfrig_swarm_mas::state::{AgentRole, SwarmMasState};
use gfrig_swarm_mas::tools_coder::{BarChartArgs, RenderBarChartTool, SymbolicArgs, SymbolicCalculateTool};
use gfrig_swarm_mas::tools_researcher::{DocsRagArgs, DocsRagTool};
use rig::tool::{Tool, ToolContext};
use tempfile::tempdir;

#[tokio::test]
async fn test_delegator_and_complete_tools() {
    let mut ctx = ToolContext::default();

    // 1. Test DelegatorTool
    let delegator = DelegatorTool;
    let res = delegator
        .call(
            &mut ctx,
            DelegatorArgs {
                target: "coder".to_string(),
                task: "Run differential equation".to_string(),
            },
        )
        .await
        .expect("Delegator tool call must succeed");

    assert!(res.contains("Successfully delegated"));

    let action = take_last_swarm_action();
    match action {
        Some(SwarmAction::Delegate { target, task }) => {
            assert_eq!(target, AgentRole::Coder);
            assert_eq!(task, "Run differential equation");
        }
        _ => panic!("Expected SwarmAction::Delegate"),
    }

    // 2. Test CompleteTool
    let completer = CompleteTool;
    let res2 = completer
        .call(
            &mut ctx,
            CompleteArgs {
                final_answer: "Task successfully resolved.".to_string(),
            },
        )
        .await
        .expect("Complete tool call must succeed");

    assert!(res2.contains("Task marked as completed"));

    let action2 = take_last_swarm_action();
    match action2 {
        Some(SwarmAction::Complete { final_answer }) => {
            assert_eq!(final_answer, "Task successfully resolved.");
        }
        _ => panic!("Expected SwarmAction::Complete"),
    }
}

#[test]
fn test_swarm_state_history() {
    let mut state = SwarmMasState::new("Solve differential equation");
    assert_eq!(state.messages.len(), 1);
    assert_eq!(state.messages[0].sender, AgentRole::User);

    state.push_message(AgentRole::Researcher, "DELEGATE: CODER\nSyntax found.");
    state.push_message(AgentRole::Coder, "COMPLETE: Solved.");

    let history = state.format_history();
    assert!(history.contains("[USER]: Solve differential equation"));
    assert!(history.contains("[RESEARCHER]: DELEGATE: CODER"));
    assert!(history.contains("[CODER]: COMPLETE: Solved."));
}

#[tokio::test]
async fn test_symbolic_calculate_tool_ode() {
    let tool = SymbolicCalculateTool;
    let mut ctx = ToolContext::default();

    let ode_res = tool
        .call(
            &mut ctx,
            SymbolicArgs {
                operation: "ode_separable".to_string(),
                expression: "y * x^2".to_string(),
            },
        )
        .await
        .expect("ODE calculation must succeed");

    assert!(ode_res.contains("C1 * exp"));
}

#[tokio::test]
async fn test_render_bar_chart_tool() {
    let dir = tempdir().expect("Failed to create tempdir");
    let chart_path = dir.path().join("swarm_chart.png");

    let tool = RenderBarChartTool;
    let mut ctx = ToolContext::default();

    let _output = tool
        .call(
            &mut ctx,
            BarChartArgs {
                title: "Test Swarm Chart".to_string(),
                y_label: "Values".to_string(),
                labels: vec!["A".into(), "B".into()],
                values: vec![10.0, 20.0],
                file_name: chart_path.to_str().unwrap().to_string(),
            },
        )
        .await
        .expect("Chart rendering must succeed");

    assert!(chart_path.exists(), "PNG chart file must be created on disk");
}

#[tokio::test]
async fn test_docs_rag_tool() {
    let tool = DocsRagTool;
    let mut ctx = ToolContext::default();

    let res = tool
        .call(
            &mut ctx,
            DocsRagArgs {
                library: "sympy".to_string(),
                query: "dsolve differential equation dy/dx".to_string(),
            },
        )
        .await
        .expect("RAG query must succeed");

    assert!(res.contains("symbolic_calculate"));
    assert!(res.contains("ode_separable"));
}
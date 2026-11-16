use gfrig_supervisory_mas::state::{extract_base64_from_output, AgentRole, SupervisorDecision, SupervisoryMasState};
use gfrig_supervisory_mas::tools_coder::{
    take_last_generated_base64, BarChartArgs, RenderBarChartTool, SymbolicArgs, SymbolicCalculateTool,
};
use gfrig_supervisory_mas::tools_researcher::{DocsRagArgs, DocsRagTool};
use gfrig_supervisory_mas::SupervisoryScientificMas;
use rig::tool::{Tool, ToolContext};
use tempfile::tempdir;

#[test]
fn test_supervisor_decision_parser() {
    // 1. Test parsing the final response
    let text_finish = "FINAL ANSWER:\nSuccessfully resolved.\n---";
    let decision_finish = SupervisoryScientificMas::parse_supervisor_decision(text_finish);
    match decision_finish {
        SupervisorDecision::Finish { final_answer } => {
            assert!(final_answer.contains("Successfully resolved."));
        }
        _ => panic!("Decision Expected: Finish"),
    }

    // 2. Test parsing of delegation to Researcher
    let text_research = "DELEGATE: RESEARCHER\nFind information about Wikipedia.";
    let decision_research = SupervisoryScientificMas::parse_supervisor_decision(text_research);
    match decision_research {
        SupervisorDecision::Delegate { to_agent, task } => {
            assert_eq!(to_agent, AgentRole::Researcher);
            assert!(task.contains("Wikipedia"));
        }
        _ => panic!("Decision by Delegate regarding Researcher expected"),
    }

    // 3. Test parsing of delegation to Coder
    let text_coder = "DELEGATE: CODER\nRun the differential equation calculation.";
    let decision_coder = SupervisoryScientificMas::parse_supervisor_decision(text_coder);
    match decision_coder {
        SupervisorDecision::Delegate { to_agent, task } => {
            assert_eq!(to_agent, AgentRole::Coder);
            assert!(task.contains("differential equation"));
        }
        _ => panic!("Delegate decision expected on Coder"),
    }
}

#[test]
fn test_state_history_and_base64_extraction() {
    let mut state = SupervisoryMasState::new("Test query");
    assert_eq!(state.messages.len(), 1);
    assert_eq!(state.messages[0].sender, AgentRole::User);

    state.push_message(AgentRole::Researcher, "RESEARCH COMPLETE: Data found.");
    state.push_message(AgentRole::Coder, "EXECUTION COMPLETE: Graph saved.");

    let history = state.format_history_for_prompt();
    assert!(history.contains("[USER]: Test query"));
    assert!(history.contains("[RESEARCHER]: RESEARCH COMPLETE"));
    assert!(history.contains("[CODER]: EXECUTION COMPLETE"));

    // Test Base64 extraction
    let raw_output = "Graph generated.\n---B64-START---\niVBORw0KGgoAAAANSUhEUg==\n---B64-END---";
    let extracted = extract_base64_from_output(raw_output);
    assert_eq!(extracted.as_deref(), Some("iVBORw0KGgoAAAANSUhEUg=="));

    // Test for missing tags
    assert_eq!(extract_base64_from_output("There are no signs here."), None);
}

#[tokio::test]
async fn test_symbolic_calculate_tool_ode_and_diff() {
    let tool = SymbolicCalculateTool;
    let mut ctx = ToolContext::default();

    // 1. Test separovatelné ODE: dy/dx = y * x^2
    let ode_res = tool
        .call(
            &mut ctx,
            SymbolicArgs {
                operation: "ode_separable".to_string(),
                expression: "y * x^2".to_string(),
            },
        )
        .await
        .expect("The ODE calculation must complete successfully.");

    assert!(ode_res.contains("C1 * exp"));

    // 2. Test symbolické derivace
    let diff_res = tool
        .call(
            &mut ctx,
            SymbolicArgs {
                operation: "differentiate".to_string(),
                expression: "x^3 + 2*x".to_string(),
            },
        )
        .await
        .expect("The derivation must be successfully completed.");

    assert!(diff_res.contains("d/dx"));
}

#[tokio::test]
async fn test_render_bar_chart_tool_creates_png_and_base64() {
    let dir = tempdir().expect("Failed to create a temporary directory");
    let chart_path = dir.path().join("test_countries.png");
    let chart_path_str = chart_path.to_str().unwrap().to_string();

    let tool = RenderBarChartTool;
    let mut ctx = ToolContext::default();

    let _output = tool
        .call(
            &mut ctx,
            BarChartArgs {
                title: "Top 5 Largest Countries".to_string(),
                y_label: "Area (km²)".to_string(),
                labels: vec![
                    "Russia".into(),
                    "Canada".into(),
                    "China".into(),
                    "USA".into(),
                    "Brazil".into(),
                ],
                values: vec![17_098_242.0, 9_984_670.0, 9_596_961.0, 9_525_067.0, 8_515_767.0],
                file_name: chart_path_str.clone(),
            },
        )
        .await
        .expect("The rendering of the graph must be successful.");

    // 1. Verification of the actual creation of the file on the disk
    assert!(chart_path.exists(), "The PNG file was not created on the disk.");

    // 2. Verification that Base64 data was safely stored in the thread-local state variable (XAI pattern)
    let b64_opt = take_last_generated_base64();
    assert!(b64_opt.is_some(), "Base64 data must be present in the state variable");
    
    let b64 = b64_opt.unwrap();
    assert!(
        b64.starts_with("iVBORw0K"),
        "The Base64 output does not begin with valid PNG magic bytes."
    );
}

#[tokio::test]
async fn test_docs_rag_tool_syntax() {
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
        .expect("The RAG query must complete successfully.");

    assert!(res.contains("symbolic_calculate"));
    assert!(res.contains("ode_separable"));
}
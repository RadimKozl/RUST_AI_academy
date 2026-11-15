use gfrig_collaborative_scientific_mas::state::{extract_base64_from_output, AgentRole, ScientificMasState};
use gfrig_collaborative_scientific_mas::tools_coder::{
    BarChartArgs, RenderBarChartTool, SymbolicArgs, SymbolicCalculateTool,
};
use gfrig_collaborative_scientific_mas::tools_researcher::{DocsRagArgs, DocsRagTool};
use rig::tool::{Tool, ToolContext};
use tempfile::tempdir;

#[test]
fn test_state_history_and_base64_extraction() {
    let mut state = ScientificMasState::new("Solve dy/dx = y * x^2");
    assert_eq!(state.messages.len(), 1);
    assert_eq!(state.messages[0].sender, AgentRole::User);

    state.push_message(
        AgentRole::Researcher,
        "Syntax found: dsolve(Eq(y(x).diff(x), y(x)*x**2)). Pass to Coder.",
    );
    state.push_message(
        AgentRole::Coder,
        "FINAL ANSWER: y(x) = C1 * exp(x^3 / 3)",
    );

    let history = state.format_history_for_prompt();
    assert!(history.contains("[USER]: Solve dy/dx = y * x^2"));
    assert!(history.contains("[RESEARCHER]: Syntax found:"));
    assert!(history.contains("[CODER]: FINAL ANSWER:"));

    // Test of extracting a Base64 payload from tool output
    let raw_tool_output =
        "Chart saved to 'chart.png'.\n---B64-START---\n  iVBORw0KGgoAAAANSUhEUg==  \n---B64-END---";
    let extracted = extract_base64_from_output(raw_tool_output);
    assert_eq!(extracted.as_deref(), Some("iVBORw0KGgoAAAANSUhEUg=="));

    // Test of an invalid / missing marker
    assert_eq!(extract_base64_from_output("No markers here"), None);
}

#[tokio::test]
async fn test_symbolic_calculate_tool_ode_and_diff() {
    let tool = SymbolicCalculateTool;
    let mut ctx = ToolContext::default();

    // 1. Test of the differential equation dy/dx = y * x^2
    let ode_res = tool
        .call(
            &mut ctx,
            SymbolicArgs {
                operation: "ode_dy_dx_eq_y_fx".to_string(),
                expression: "y * x^2".to_string(),
            },
        )
        .await
        .expect("ODE calculation should succeed");

    assert!(ode_res.contains("C1 * exp(x^3 / 3)"));

    // 2. Test of symbolic differentiation via MathCore
    let diff_res = tool
        .call(
            &mut ctx,
            SymbolicArgs {
                operation: "differentiate".to_string(),
                expression: "x^3 + 2*x".to_string(),
            },
        )
        .await
        .expect("Differentiation should succeed");

    assert!(diff_res.contains("d/dx (x^3 + 2*x) ="));
}

#[tokio::test]
async fn test_render_bar_chart_tool_creates_png_and_base64() {
    let dir = tempdir().expect("Failed to create temp dir");
    let chart_path = dir.path().join("test_countries.png");
    let chart_path_str = chart_path.to_str().unwrap().to_string();

    let tool = RenderBarChartTool;
    let mut ctx = ToolContext::default();

    let output = tool
        .call(
            &mut ctx,
            BarChartArgs {
                title: "Top 5 Largest Countries".to_string(),
                y_label: "Area (km2)".to_string(),
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
        .expect("Bar chart rendering should succeed");

    // We verify the physical existence of the file on the disk
    assert!(chart_path.exists(), "PNG file was not created on disk");

    // Verify a valid Base64 PNG header (iVBORw0K)
    let b64 = extract_base64_from_output(&output).expect("Base64 marker must be present");
    assert!(
        b64.starts_with("iVBORw0K"),
        "Base64 output does not start with valid PNG magic bytes"
    );
}

#[tokio::test]
async fn test_render_bar_chart_validation_error() {
    let tool = RenderBarChartTool;
    let mut ctx = ToolContext::default();

    let err = tool
        .call(
            &mut ctx,
            BarChartArgs {
                title: "Invalid Chart".to_string(),
                y_label: "Y".to_string(),
                labels: vec!["A".into(), "B".into()],
                values: vec![10.0], // Length does not match
                file_name: "invalid.png".to_string(),
            },
        )
        .await;

    assert!(err.is_err(), "Mismatched labels/values length must return an error");
}

#[tokio::test]
async fn test_docs_rag_tool_ode_syntax() {
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
        .expect("RAG lookup should succeed");

    assert!(res.contains("dsolve"));
    assert!(res.contains("symbolic_calculate"));
}
use crate::state::SupervisoryMasState;
use std::fs;
use std::path::Path;

/// Generates a Mermaid Supervisory Graph-Flow architecture diagram (equivalent to `draw_mermaid_png`)
pub fn generate_static_mermaid_diagram() -> String {
    r#"flowchart TD
    START((START)) --> Supervisor[Supervisor Node]
    Supervisor -->|DELEGATE: RESEARCHER| Researcher[Researcher Node]
    Supervisor -->|DELEGATE: CODER| Coder[Coder Node]
    Supervisor -->|FINAL ANSWER / Max Limit| END((END))
    Researcher -->|RESEARCH COMPLETE| Supervisor
    Coder -->|EXECUTION COMPLETE| Supervisor
"#
    .to_string()
}

/// Generates a dynamic Mermaid diagram of the actual execution flow from the `trace` state
pub fn generate_execution_trace_mermaid(state: &SupervisoryMasState) -> String {
    let mut lines = vec!["flowchart LR".to_string()];
    for window in state.trace.windows(2) {
        let a = &window[0];
        let b = &window[1];
        let label = b.action.replace('"', "'");
        lines.push(format!(
            "    S{}[\"{}: {}\"] -->|\"{}\"| S{}[\"{}: {}\"]",
            a.step, a.node, a.action, label, b.step, b.node, b.action
        ));
    }
    lines.join("\n")
}

/// Saves the Mermaid diagram to a file (equivalent to `visualizer.visualize(...)`)
pub fn save_mermaid_visualization(output_file_path: &str, state: Option<&SupervisoryMasState>) -> anyhow::Result<()> {
    if let Some(parent) = Path::new(output_file_path).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let content = match state {
        Some(s) => format!(
            "# Supervisory MAS Execution Graph\n\n## Static Topology\n```mermaid\n{}\n```\n\n## Runtime Execution Trace\n```mermaid\n{}\n```\n",
            generate_static_mermaid_diagram(),
            generate_execution_trace_mermaid(s)
        ),
        None => generate_static_mermaid_diagram(),
    };

    fs::write(output_file_path, content)?;
    println!("✅ Graph visualization saved to: {output_file_path}");
    Ok(())
}
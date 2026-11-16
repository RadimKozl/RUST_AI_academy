// src/visualizer.rs
use crate::state::SwarmMasState;
use std::fs;
use std::path::Path;

pub fn generate_static_mermaid_diagram() -> String {
    r#"flowchart TD
    START((START)) --> Researcher[Researcher Agent]
    Researcher -->|DELEGATE: CODER| Coder[Coder Agent]
    Coder -->|DELEGATE: RESEARCHER| Researcher
    Researcher -->|COMPLETE| END((END))
    Coder -->|COMPLETE| END((END))
"#
    .to_string()
}

pub fn save_mermaid_visualization(output_file_path: &str, _state: Option<&SwarmMasState>) -> anyhow::Result<()> {
    if let Some(parent) = Path::new(output_file_path).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let content = format!(
        "# Swarm MAS Execution Graph\n\n```mermaid\n{}\n```\n",
        generate_static_mermaid_diagram()
    );

    fs::write(output_file_path, content)?;
    println!("✅ Swarm graph visualization saved to: {output_file_path}");
    Ok(())
}
use gfrig_hybrid_agent::{DynamicWorkflowBuilder, NodeMode, UnifiedNode};
use graph_flow::{InMemorySessionStorage, Session, SessionStorage, Task};
use std::sync::Arc;

#[tokio::test]
async fn test_hybrid_rag_workflow_execution() {
    let topic = "Special relativity";

    // 1. Definition of nodes for testing
    let llm_search_node = UnifiedNode {
        id_str: "llm_search".to_string(),
        label: "LLM Search Agent".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Mock Search".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    let mock_context = "Special relativity was proposed by Albert Einstein in 1905.".to_string();
                    ctx.set("context", mock_context.clone())?;
                    Ok(mock_context)
                })
            }),
        },
    };

    let theory_node = UnifiedNode {
        id_str: "theory_node".to_string(),
        label: "Theory Expert Node".to_string(),
        mode: NodeMode::CustomTool {
            action_name: "Mock Theory".to_string(),
            logic: Arc::new(|ctx| {
                Box::pin(async move {
                    let mock_theory = "The theory states that the laws of physics are the same for all non-accelerating observers.".to_string();
                    ctx.set("theory", mock_theory.clone())?;
                    Ok(mock_theory)
                })
            }),
        },
    };

    let nodes = vec![llm_search_node, theory_node];
    let edges = vec![
        ("llm_search", "theory_node"),
    ];

    let mut workflow_builder = DynamicWorkflowBuilder::new("test_hybrid_workflow");
    let mut task_map = std::collections::HashMap::new();

    for node in nodes {
        let task_arc = Arc::new(node);
        task_map.insert(task_arc.id().to_string(), task_arc.clone());
        workflow_builder = workflow_builder.add_task(task_arc);
    }

    for (from, to) in edges {
        workflow_builder = workflow_builder.add_edge(from, to);
    }

    let mermaid_code = workflow_builder.generate_mermaid();
    let _graph = workflow_builder.build().unwrap();

    let storage = Arc::new(InMemorySessionStorage::new());
    let session_id = "test_session_1".to_string();
    let session = Session::new_from_task(session_id.clone(), "llm_search");

    session.context.set("topic", topic.to_string()).unwrap();
    session.context.set("context", "".to_string()).unwrap();
    session.context.set("theory", "".to_string()).unwrap();
    session.context.set("architecture_mermaid", mermaid_code).unwrap();

    SessionStorage::save(&*storage, session).await.unwrap();

    // Executing nodes in sequence
    let sess = SessionStorage::get(&*storage, &session_id).await.unwrap().unwrap();
    if let Some(search_node) = task_map.get("llm_search") {
        search_node.run(sess.context.clone()).await.unwrap();
    }
    if let Some(theory_n) = task_map.get("theory_node") {
        theory_n.run(sess.context.clone()).await.unwrap();
    }
    SessionStorage::save(&*storage, sess).await.unwrap();

    let final_sess = SessionStorage::get(&*storage, &session_id).await.unwrap().unwrap();
    let ctx = &final_sess.context;

    // Verification of results in context
    let context_val = ctx.get::<String>("context").unwrap_or_default();
    let theory_val = ctx.get::<String>("theory").unwrap_or_default();

    assert!(!context_val.is_empty(), "Context must be populated");
    assert!(!theory_val.is_empty(), "Theory must be populated");
    assert!(context_val.contains("Albert Einstein"));
    assert!(theory_val.contains("laws of physics"));
}
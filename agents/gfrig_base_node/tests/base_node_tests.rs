use gfrig_base_node::{InputTask, GraphStepRecord};
use graph_flow::{Context, Task};

#[tokio::test]
async fn test_input_task_execution() {
    let context = Context::new();
    context.set("query", "Test Query".to_string()).unwrap();
    context.set("data", Vec::<String>::new()).unwrap();
    context.set("audit_trail", Vec::<GraphStepRecord>::new()).unwrap();

    let task = InputTask {
        id_str: "test_node".to_string(),
        label: "TestLabel".to_string(),
    };

    let result = task.run(context.clone()).await;
    assert!(result.is_ok());

    let data = context.get::<Vec<String>>("data").unwrap();
    assert_eq!(data.len(), 1);
    assert!(data[0].contains("TestLabel"));

    let audit = context.get::<Vec<GraphStepRecord>>("audit_trail").unwrap();
    assert_eq!(audit.len(), 1);
    assert_eq!(audit[0].task_id, "test_node");
}
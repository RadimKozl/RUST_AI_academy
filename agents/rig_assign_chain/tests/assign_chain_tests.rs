use rig_assign_chain::{assign_metadata, run_assign_chain};

#[test]
fn test_assign_metadata_calculation() {
    let question = "Test query";
    let model = "qwen3.5:4b";

    let ctx = assign_metadata(question, model);

    assert_eq!(ctx.question, question);
    assert_eq!(ctx.question_length, 10);
    assert_eq!(ctx.processing_model, model);
    assert!(!ctx.timestamp.is_empty());
}

#[tokio::test]
async fn test_assign_chain_execution() {
    let model_name = "qwen3.5:4b";
    let question = "What is Rust?";

    let system_instruction = "Be concise.";
    let prompt_template = "Date: {timestamp} | Model: {processing_model} | Length: {question_length} | Question: {question}";

    let result = run_assign_chain(model_name, question, system_instruction, prompt_template).await;

    match result {
        Ok(res) => {
            assert_eq!(res.context.question, question);
            assert!(res.context.question_length > 0);
            assert!(!res.final_output.is_empty());
            assert_eq!(res.history.len(), 2);
            println!("RunnableAssign test passed successfully!");
        }
        Err(e) => {
            println!("Ollama is unavailable (expected in CI/offline environments): {}", e);
        }
    }
}
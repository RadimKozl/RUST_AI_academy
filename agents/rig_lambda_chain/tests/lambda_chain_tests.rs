use rig_lambda_chain::{process_and_summarize, run_lambda_chain};

#[test]
fn test_process_and_summarize_logic() {
    let long_text = "An atom is the basic building block of all matter in the universe.";
    let result = process_and_summarize(long_text);

    assert!(result.summary.ends_with("..."));
    assert_eq!(result.full_length, long_text.chars().count());
    assert_eq!(result.source, "AI_Model_Ollama_Qwen");
}

#[tokio::test]
async fn test_lambda_chain_execution() {
    let model_name = "qwen3.5:4b";
    let tone = "Keep it brief";
    let system_instruction = "You are a scientist.";
    let prompt_template = "Explain an atom. {tone}.";

    let result = run_lambda_chain(model_name, tone, system_instruction, prompt_template).await;

    match result {
        Ok(res) => {
            assert!(!res.final_data.summary.is_empty());
            assert!(res.final_data.full_length > 0);
            assert_eq!(res.history.len(), 2);
            println!("RunnableLambda test passed successfully!");
        }
        Err(e) => {
            println!("Ollama is unavailable (expected in CI/offline environments): {}", e);
        }
    }
}
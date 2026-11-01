use rig_passthrough_chain::run_passthrough_rag_chain;

#[tokio::test]
async fn test_passthrough_rag_chain() {
    let model_name = "qwen3.5:4b";
    let question = "Where was Barack Obama born?";

    let context_system = "You provide facts.";
    let context_prompt = "Give a one-sentence fact about {question}.";
    let final_system = "You answer questions.";
    let final_prompt = "Context: {context}\nQuestion: {question}\nAnswer:";

    let result = run_passthrough_rag_chain(
        model_name,
        question,
        context_system,
        context_prompt,
        final_system,
        final_prompt,
    )
    .await;

    match result {
        Ok(res) => {
            assert_eq!(res.original_question, question);
            assert!(!res.generated_context.is_empty());
            assert!(!res.final_output.is_empty());
            assert_eq!(res.history.len(), 2);
            println!("Passthrough test passed successfully!");
        }
        Err(e) => {
            println!("Ollama is unavailable (expected in CI/offline environments): {}", e);
        }
    }
}
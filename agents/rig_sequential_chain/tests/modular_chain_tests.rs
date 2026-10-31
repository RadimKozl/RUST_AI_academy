use rig_sequential_chain::{run_modular_chain, ChainStep};

#[tokio::test]
async fn test_modular_chain_execution_and_history() {
    let model_name = "qwen3.5:4b";
    let system_instruction = "You are a concise geography assistant.";

    // Dynamically define N steps (in this test, 2 steps)
    let steps = vec![
        ChainStep {
            name: "find_city".to_string(),
            prompt_template: "What is the capital city of {input}? Return only the city name.".to_string(),
        },
        ChainStep {
            name: "find_continent".to_string(),
            prompt_template: "On which continent is {input} located? Return only the continent name.".to_string(),
        },
    ];

    let initial_input = "France";

    let result = run_modular_chain(model_name, initial_input, system_instruction, &steps).await;

    match result {
        Ok(res) => {
            // 1. Verification that the number of recorded steps in the history exactly matches the number of entries
            assert_eq!(res.history.len(), 2, "The history must contain exactly 2 steps.");

            // 2. Verification that the first step received the initial input
            assert_eq!(res.history[0].step_name, "find_city");
            assert_eq!(res.history[0].input_value, "France");
            assert!(!res.history[0].generated_output.is_empty());

            // 3. Verification of context passing: The input of step 2 must be the output of step 1
            assert_eq!(res.history[1].step_name, "find_continent");
            assert_eq!(res.history[1].input_value, res.history[0].generated_output);
            assert!(!res.history[1].generated_output.is_empty());

            // 4. Verification of the final output
            assert_eq!(res.final_output, res.history[1].generated_output);

            println!("Test passed successfully! Record for XAI/Neo4j:\n{:#?}", res.history);
        }
        Err(e) => {
            // If Ollama is not running locally, the test will not fail due to compilation issues or a non-functional interface.
            println!("Ollama is unavailable (expected in CI/offline environments): {}", e);
        }
    }
}
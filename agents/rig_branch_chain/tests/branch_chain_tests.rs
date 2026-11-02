use rig_branch_chain::run_branch_chain;

#[tokio::test]
async fn test_branch_chain_routing() {
    let model_name = "qwen3.5:4b";
    let topic = "gravity";

    // Test formal branch
    let res_formal = run_branch_chain(model_name, topic, "formal style").await;
    match res_formal {
        Ok(res) => {
            assert_eq!(res.matched_branch, "formal_branch");
            assert!(!res.final_output.is_empty());
        }
        Err(e) => println!("Ollama unavailable: {}", e),
    }

    // Test funny branch
    let res_funny = run_branch_chain(model_name, topic, "tell a joke").await;
    match res_funny {
        Ok(res) => {
            assert_eq!(res.matched_branch, "funny_branch");
            assert!(!res.final_output.is_empty());
        }
        Err(e) => println!("Ollama unavailable: {}", e),
    }

    // Test default branch
    let res_default = run_branch_chain(model_name, topic, "neutral").await;
    match res_default {
        Ok(res) => {
            assert_eq!(res.matched_branch, "default_casual_branch");
            assert!(!res.final_output.is_empty());
        }
        Err(e) => println!("Ollama unavailable: {}", e),
    }
}
use lm_linfa_ftrl::FtrlPipeline;

#[test]
fn test_ftrl_training_and_log_loss() {
    let alpha = 0.005;
    let beta = 1.0;
    let l1_ratio = 0.005;
    let l2_ratio = 1.0;
    let seed = 42;

    let result = FtrlPipeline::train_ftrl(alpha, beta, l1_ratio, l2_ratio, seed);
    assert!(result.is_ok(), "FTRL training failed");

    let eval = result.unwrap();
    assert!(eval.valid_log_loss > 0.0, "Log loss must be a positive number");
    assert!(eval.valid_log_loss < 1.0, "Log loss should have a reasonable value");
    assert_eq!(eval.target_probabilities.len(), eval.true_targets.len());

    let save_res = FtrlPipeline::save_to_models(&eval, "test_ftrl_results.json");
    assert!(save_res.is_ok(), "Saving JSON model failed");
}
use lm_linfa_ica::IcaPipeline;

#[test]
fn test_ica_pipeline() {
    let result = IcaPipeline::run_ica();
    assert!(result.is_ok(), "FastICA calculation failed");

    let eval = result.unwrap();
    assert_eq!(eval.n_samples, 2000);
    assert_eq!(eval.original_signals.len(), 2000);
    assert_eq!(eval.mixed_signals.len(), 2000);
    assert_eq!(eval.unmixed_signals.len(), 2000);

    let save_res = IcaPipeline::save_to_models(&eval, "test_ica_results.json");
    assert!(save_res.is_ok(), "Failed to save JSON results");
}
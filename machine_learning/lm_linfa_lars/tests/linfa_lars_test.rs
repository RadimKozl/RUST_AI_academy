use linfa::dataset::Records;
use lm_linfa_lars::LarsPipeline;
use std::path::Path;

#[test]
fn test_lars_pipeline_flow() {
    let (train, test) = LarsPipeline::load_and_split_data(0.8);

    assert!(train.nsamples() > 0);
    assert!(test.nsamples() > 0);

    let model = LarsPipeline::train_model(&train).expect("Training failed");
    let (predictions, _metrics) = LarsPipeline::evaluate(&model, &test);

    assert_eq!(predictions.len(), test.nsamples());

    // Save and load
    let test_path = Path::new("target/test_model.json");
    LarsPipeline::save_model(&model, test_path).expect("Save failed");
    let loaded_model = LarsPipeline::load_model(test_path).expect("Load failed");

    // Cleanup after test
    let _ = std::fs::remove_file(test_path);

    // Instead of assert_eq!(model, loaded_model) we compare values ​​with tolerance (eps)
    let eps = 1e-10;
    
    assert!((model.intercept - loaded_model.intercept).abs() < eps);
    assert_eq!(model.params.len(), loaded_model.params.len());
    
    for (p1, p2) in model.params.iter().zip(loaded_model.params.iter()) {
        assert!((p1 - p2).abs() < eps, "Parameters differ by more than a tolerance: {} vs {}", p1, p2);
    }
}
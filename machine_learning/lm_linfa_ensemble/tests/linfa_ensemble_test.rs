use lm_linfa_ensemble::{run_adaboost, run_random_forest};
use std::path::Path;

#[test]
fn test_adaboost_execution_and_saving() {
    let result = run_adaboost(10, 0.1, 3).expect("AdaBoost execution failed");
    
    assert!(result.accuracy > 0.0);
    assert!(!result.predictions.is_empty());
    assert!(Path::new("models/adaboost_result.json").exists());
}

#[test]
fn test_random_forest_execution_and_saving() {
    let result = run_random_forest(10, 0.8, 0.8).expect("RandomForest execution failed");
    
    assert!(result.accuracy > 0.0);
    assert!(!result.predictions.is_empty());
    assert!(Path::new("models/random_forest_result.json").exists());
}
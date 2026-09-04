use lm_smartcore_logistic_regression::{load_model, run_logistic_regression};
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::path::PathBuf;

#[test]
fn test_logistic_regression_workflow() {
    // If training fails, unwrap() prints the exact error to the console
    let result = run_logistic_regression().expect("TRAINING OR SAVE FAILED");

    assert!(result.accuracy > 80.0, "Accuracy should be above 80%");
    assert!(!result.test_predictions.is_empty());

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let model_path = PathBuf::from(manifest_dir).join("models").join("logistic_regression.bin");

    assert!(
        model_path.exists(),
        "The model file does not exist at the path: {:?}",
        model_path
    );

    let loaded_model = load_model("logistic_regression.bin").expect("Loading binary model failed");

    let sample = DenseMatrix::from_2d_array(&[&[5.1, 3.5, 1.4, 0.2]]).unwrap();
    let prediction = loaded_model.predict(&sample).expect("Prediction by loaded model failed");

    assert_eq!(prediction.len(), 1);
    assert_eq!(prediction[0], 0);
}
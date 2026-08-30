use lm_linfa_svm::SvmPipeline;

#[test]
fn test_binary_svm_training() {
    let (train, valid) = SvmPipeline::load_binary_wine(42);
    let result = SvmPipeline::train_binary_svm(&train, &valid, 80.0, 50000.0, 5000.0);

    assert!(result.is_ok(), "Binary SVM training failed!");
    let (eval, model) = result.unwrap();
    assert!(eval.accuracy_or_mse > 50.0, "Accuracy should be above 50%");

    let save_res = SvmPipeline::save_model(&model, "test_binary_svm.json");
    assert!(save_res.is_ok(), "Saving the model failed!");
}

#[test]
fn test_multiclass_svm_training() {
    let (train, valid) = SvmPipeline::load_multiclass_wine(42);
    let result = SvmPipeline::train_multiclass_svm(&train, &valid, 30.0);

    assert!(result.is_ok(), "Multi-class SVM training failed!");
    let (eval, _) = result.unwrap();
    assert!(eval.accuracy_or_mse > 30.0);
}

#[test]
fn test_svr_regression() {
    let result = SvmPipeline::train_svr_regression(100.0, 0.1, 10.0);
    assert!(result.is_ok(), "SVR regression training failed!");
    let (mse, _, _) = result.unwrap();
    assert!(mse < 1.0, "MSE of regression should be low");
}
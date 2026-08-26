use linfa::dataset::Records;
use lm_linfa_linear::RegressionPipeline;
use tempfile::NamedTempFile;

#[test]
fn test_pipeline_flow() {
    let (train, test) = RegressionPipeline::load_and_split_data(0.8);
    assert!(train.nsamples() > 0);
    assert!(test.nsamples() > 0);

    let model = RegressionPipeline::train_model(&train).unwrap();

    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    RegressionPipeline::save_model(&model, path).unwrap();
    let loaded_model = RegressionPipeline::load_model(path).unwrap();

    let (predictions, metrics) = RegressionPipeline::evaluate(&loaded_model, &test);
    assert_eq!(predictions.len(), test.nsamples());
    assert!(metrics.r2_score <= 1.0);
}
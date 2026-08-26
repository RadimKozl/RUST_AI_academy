use linfa::dataset::Records;
use lm_linfa_logistic::LogisticPipeline;
use tempfile::NamedTempFile;

#[test]
fn test_logistic_pipeline_flow() {
    let (train, test) = LogisticPipeline::load_and_split_data(0.9);
    assert!(train.nsamples() > 0);
    assert!(test.nsamples() > 0);

    let model = LogisticPipeline::train_model(&train).unwrap();

    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    LogisticPipeline::save_model(&model, path).unwrap();
    let loaded_model = LogisticPipeline::load_model(path).unwrap();

    let (predictions, metrics) = LogisticPipeline::evaluate(&loaded_model, &test);
    assert_eq!(predictions.len(), test.nsamples());
    assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
}
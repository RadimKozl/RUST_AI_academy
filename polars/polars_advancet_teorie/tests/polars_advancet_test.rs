use polars_advancet_teorie::{execute_etl_pipeline, PipelineConfig};
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn test_etl_pipeline_execution() {
    let dir = tempdir().expect("Failed to create temp dir");
    let input_path = dir.path().join("test_data.json");
    
    let sample_json = r#"[
        {"id": 1, "value": 10.5},
        {"id": 2, "value": 20.0},
        {"id": 3, "value": 30.2}
    ]"#;

    let mut file = File::create(&input_path).expect("Failed to create test JSON");
    file.write_all(sample_json.as_bytes()).expect("Failed to write JSON");

    let output_dir = dir.path().join("output");
    let config = PipelineConfig {
        input_files: vec![input_path],
        output_dir: output_dir.clone(),
        train_ratio: 0.66,
    };

    let result = execute_etl_pipeline(config);
    assert!(result.is_ok(), "Pipeline execution failed!");
    assert!(output_dir.join("train.parquet").exists());
    assert!(output_dir.join("test.parquet").exists());
}
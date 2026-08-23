use lm_linfa_market_pipeline::MarketMLPipeline;
use polars::prelude::*;
use std::fs::File;
use tempfile::tempdir;

fn create_mock_parquet(path: &std::path::Path) {
    let prices = vec![95000.0, 95010.0, 95020.0, 95015.0, 95030.0, 95025.0, 95040.0, 95050.0, 95045.0, 95060.0];
    let smas = vec![94990.0, 95000.0, 95010.0, 95012.0, 95020.0, 95022.0, 95030.0, 95035.0, 95040.0, 95048.0];
    let b_upper = vec![95020.0, 95030.0, 95040.0, 95035.0, 95050.0, 95045.0, 95060.0, 95070.0, 95065.0, 95080.0];
    let b_lower = vec![94960.0, 94970.0, 94980.0, 94989.0, 94990.0, 94999.0, 95000.0, 95000.0, 95015.0, 95016.0];

    let s_price = Series::new("last_price".into(), prices);
    let s_sma = Series::new("sma_5".into(), smas);
    let s_bupper = Series::new("bband_upper".into(), b_upper);
    let s_blower = Series::new("bband_lower".into(), b_lower);

    let mut df = DataFrame::new_infer_height(vec![
        s_price.into(),
        s_sma.into(),
        s_bupper.into(),
        s_blower.into(),
    ])
    .unwrap();

    let file = File::create(path).unwrap();
    ParquetWriter::new(file).finish(&mut df).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_full_ml_pipeline_lifecycle() {
    let dir = tempdir().unwrap();
    let parquet_path = dir.path().join("test_features.parquet");
    let model_path = dir.path().join("test_model.bin");

    create_mock_parquet(&parquet_path);

    let train_result = MarketMLPipeline::train_and_save_model(&parquet_path, &model_path);
    assert!(train_result.is_ok(), "Training failed");
    assert!(model_path.exists(), "Model file not created");

    let sample = vec![95050.0, 95010.0, 95100.0, 94920.0];
    let pred_result = MarketMLPipeline::predict_live_sample(&model_path, sample);

    assert!(pred_result.is_ok(), "Inference failed");
    let prediction = pred_result.unwrap();
    assert!(prediction == 0 || prediction == 1, "Prediction must be 0 or 1");
}
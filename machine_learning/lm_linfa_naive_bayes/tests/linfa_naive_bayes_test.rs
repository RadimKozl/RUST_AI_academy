use linfa::dataset::Records; // Přidán import traitu Records
use lm_linfa_naive_bayes::{BayesType, WineBayesPipeline};
use polars::prelude::*;
use std::fs::File;
use tempfile::tempdir;

fn create_mock_wine_csv(path: &std::path::Path) {
    let s_fa = Series::new("fixed acidity".into(), vec![7.4, 7.8, 11.2, 7.4, 7.9]);
    let s_va = Series::new("volatile acidity".into(), vec![0.70, 0.88, 0.28, 0.70, 0.32]);
    let s_ca = Series::new("citric acid".into(), vec![0.00, 0.00, 0.56, 0.00, 0.51]);
    let s_rs = Series::new("residual sugar".into(), vec![1.9, 2.6, 1.9, 1.9, 1.8]);
    let s_cl = Series::new("chlorides".into(), vec![0.076, 0.098, 0.075, 0.076, 0.070]);
    let s_fso = Series::new("free sulfur dioxide".into(), vec![11.0, 25.0, 17.0, 11.0, 15.0]);
    let s_tso = Series::new("total sulfur dioxide".into(), vec![34.0, 67.0, 60.0, 34.0, 56.0]);
    let s_den = Series::new("density".into(), vec![0.9978, 0.9968, 0.9980, 0.9978, 0.9969]);
    let s_ph = Series::new("pH".into(), vec![3.51, 3.20, 3.16, 3.51, 3.04]);
    let s_sul = Series::new("sulphates".into(), vec![0.56, 0.68, 0.58, 0.56, 1.08]);
    let s_alc = Series::new("alcohol".into(), vec![9.4, 9.8, 9.8, 9.4, 10.5]);
    let s_qty = Series::new("quality".into(), vec![5, 5, 6, 5, 7]);

    let columns = vec![
        s_fa.into(), s_va.into(), s_ca.into(), s_rs.into(),
        s_cl.into(), s_fso.into(), s_tso.into(), s_den.into(),
        s_ph.into(), s_sul.into(), s_alc.into(), s_qty.into(),
    ];

    // DataFrame::new accepts height (5 rows) and columns
    let mut df = DataFrame::new(5, columns).unwrap();

    let file = File::create(path).unwrap();
    CsvWriter::new(file).finish(&mut df).unwrap();
}

#[test]
fn test_bayes_pipeline_lifecycle() {
    let dir = tempdir().unwrap();
    let csv_path = dir.path().join("test_WineQT.csv");
    let model_path = dir.path().join("test_gaussian.bin");

    create_mock_wine_csv(&csv_path);

    let (train, valid) = WineBayesPipeline::load_dataset(&csv_path).expect("Dataset loading failed");
    
    // Thanks to "use linfa::dataset::Records;" nsamples() is available
    assert!(train.nsamples() > 0);

    let res = WineBayesPipeline::train_and_eval(&train, &valid, BayesType::Gaussian);
    assert!(res.is_ok(), "Training failed");

    let (model, metrics) = res.unwrap();
    let save_res = WineBayesPipeline::save_metrics(&metrics, &model_path);
    assert!(save_res.is_ok(), "Saving metrics failed");
    assert!(model_path.exists(), "Metrics file does not exist");

    let sample = vec![7.4, 0.3, 0.4, 2.0, 0.07, 15.0, 40.0, 0.995, 3.3, 0.7, 11.5];
    let pred = WineBayesPipeline::predict_model(&model, &sample);

    assert!(pred.is_ok(), "Inference failed");
    let val = pred.unwrap();
    assert!(val == 0 || val == 1);
}
use lm_linfa_clustering::ClusteringPipeline;
use polars::prelude::*;
use std::fs::File;
use tempfile::tempdir;

fn create_mock_mall_csv(path: &std::path::Path) {
    let s_id = Series::new("CustomerID".into(), vec![1, 2, 3, 4, 5]);
    let s_gender = Series::new("Gender".into(), vec!["Male", "Male", "Female", "Female", "Female"]);
    let s_age = Series::new("Age".into(), vec![19, 21, 20, 23, 31]);
    let s_income = Series::new("Annual Income (k$)".into(), vec![15, 15, 16, 16, 17]);
    let s_score = Series::new("Spending Score (1-100)".into(), vec![39, 81, 6, 77, 40]);

    let columns = vec![
        s_id.into(), s_gender.into(), s_age.into(),
        s_income.into(), s_score.into(),
    ];

    let mut df = DataFrame::new(5, columns).unwrap();
    let file = File::create(path).unwrap();
    CsvWriter::new(file).finish(&mut df).unwrap();
}

#[test]
fn test_clustering_pipeline_lifecycle() {
    let dir = tempdir().unwrap();
    let csv_path = dir.path().join("test_mall.csv");
    let png_path = dir.path().join("test_optics.png");

    create_mock_mall_csv(&csv_path);

    let records = ClusteringPipeline::load_dataset(&csv_path).expect("Load failed");
    assert_eq!(records.shape(), &[5, 2]);

    let kmeans_res = ClusteringPipeline::run_kmeans(&records, 2);
    assert!(kmeans_res.is_ok());

    let dbscan_res = ClusteringPipeline::run_dbscan(&records, 2, 10.0);
    assert!(dbscan_res.is_ok());

    let optics_res = ClusteringPipeline::run_optics(&records, 2, 10.0);
    assert!(optics_res.is_ok());

    let reachability = optics_res.unwrap();
    let render_res = ClusteringPipeline::render_reachability_plot(&reachability, &png_path);
    assert!(render_res.is_ok());
    assert!(png_path.exists());
}
use lm_linfa_elasticnet::ElasticNetPipeline;

#[test]
fn test_elasticnet_cross_validation() {
    let mut dataset = ElasticNetPipeline::load_dataset();
    let ratios = vec![0.1, 0.5, 1.0];
    let penalty = 0.3;

    let results = ElasticNetPipeline::run_cross_validation(&mut dataset, &ratios, penalty)
        .expect("Cross validation failed");

    assert_eq!(results.len(), 3);
    for res in results {
        assert!(res.r2_score >= -1.0 && res.r2_score <= 1.0);
    }
}
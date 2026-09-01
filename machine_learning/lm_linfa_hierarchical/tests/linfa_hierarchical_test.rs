use lm_linfa_hierarchical::HierarchicalPipeline;

#[test]
fn test_hierarchical_clustering() {
    let k = 3;
    let result = HierarchicalPipeline::train_hierarchical(k);
    assert!(result.is_ok(), "Hierarchical clustering training failed");

    let eval = result.unwrap();
    assert_eq!(eval.num_clusters, k);
    assert!(eval.total_samples > 0);
    assert_eq!(eval.cluster_assignments.len(), eval.total_samples);

    // Test saving to the models/ folder
    let save_res = HierarchicalPipeline::save_to_models(&eval, "test_hierarchical_results.json");
    assert!(save_res.is_ok(), "Saving JSON results failed");
}
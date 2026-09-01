use lm_linfa_reduction::ReductionPipeline;

#[test]
fn test_pca_pipeline() {
    let pca_res = ReductionPipeline::run_pca();
    assert!(pca_res.is_ok(), "PCA pipeline failed");
    assert_eq!(pca_res.unwrap().total_samples, 80);
}

#[test]
fn test_diffusion_map_pipeline() {
    let diff_res = ReductionPipeline::run_diffusion_map();
    assert!(diff_res.is_ok(), "Diffusion map pipeline failed");
    assert_eq!(diff_res.unwrap().total_samples, 102);
}

#[test]
fn test_gaussian_projection_pipeline() {
    let res = ReductionPipeline::run_gaussian_projection();
    assert!(res.is_ok(), "Gaussian Projection failed");
    assert_eq!(res.unwrap().points_2d.len(), 80);
}

#[test]
fn test_sparse_projection_pipeline() {
    let res = ReductionPipeline::run_sparse_projection();
    assert!(res.is_ok(), "Sparse Projection failed");
    assert_eq!(res.unwrap().points_2d.len(), 80);
}
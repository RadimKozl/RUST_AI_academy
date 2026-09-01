#[macro_use]
extern crate ndarray;

use anyhow::Result;
use linfa::dataset::Dataset;
use linfa::traits::{Fit, Transformer};
use linfa_datasets::generate;
use linfa_kernel::{Kernel, KernelMethod, KernelType};
use linfa_reduction::random_projection::{GaussianRandomProjection, SparseRandomProjection};
use linfa_reduction::utils::generate_convoluted_rings2d;
use linfa_reduction::{DiffusionMap, Pca};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcaEvaluation {
    pub total_samples: usize,
    pub original_dim: usize,
    pub reduced_dim: usize,
    pub points_2d: Vec<(f64, f64)>,
    pub pca_projections: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionMapEvaluation {
    pub total_samples: usize,
    pub points_2d: Vec<(f64, f64)>,
    pub embedded_2d: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionEvaluation {
    pub method_name: String,
    pub total_samples: usize,
    pub original_dim: usize,
    pub reduced_dim: usize,
    pub points_2d: Vec<(f64, f64)>,
}

pub struct ReductionPipeline;

impl ReductionPipeline {
    /// PCA on synthetic blob groups
    pub fn run_pca() -> Result<PcaEvaluation> {
        let mut rng = StdRng::seed_from_u64(42);
        let expected_centroids = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.]];
        let n = 20;

        let dataset = Dataset::from(generate::blobs(n, &expected_centroids, &mut rng));
        let records = dataset.records().to_owned();
        let total_samples = records.nrows();

        let embedding_model: Pca<f64> = Pca::params(1)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("PCA Fit failed: {:?}", e))?;

        let transformed_dataset = embedding_model.transform(dataset);
        let embedding = transformed_dataset.records();

        let mut points_2d = Vec::with_capacity(total_samples);
        let mut pca_projections = Vec::with_capacity(total_samples);

        for i in 0..total_samples {
            points_2d.push((records[[i, 0]], records[[i, 1]]));
            pca_projections.push(embedding[[i, 0]]);
        }

        Ok(PcaEvaluation {
            total_samples,
            original_dim: 2,
            reduced_dim: 1,
            points_2d,
            pca_projections,
        })
    }

    /// Diffusion Map on rings
    pub fn run_diffusion_map() -> Result<DiffusionMapEvaluation> {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 100;

        let dataset = generate_convoluted_rings2d(
            &[(0.0, 3.0), (10.0, 13.0), (20.0, 23.0)],
            n,
            &mut rng,
        );

        let kernel = Kernel::params()
            .kind(KernelType::Sparse(15))
            .method(KernelMethod::Gaussian(2.0))
            .transform(dataset.view());

        let embedding = DiffusionMap::<f64>::params(2)
            .steps(1)
            .transform(&kernel)
            .map_err(|e| anyhow::anyhow!("DiffusionMap Transform failed: {:?}", e))?;

        let emb_matrix = embedding.embedding();
        let total_samples = dataset.nrows();

        let mut points_2d = Vec::with_capacity(total_samples);
        let mut embedded_2d = Vec::with_capacity(total_samples);

        for i in 0..total_samples {
            points_2d.push((dataset[[i, 0]], dataset[[i, 1]]));
            embedded_2d.push((emb_matrix[[i, 0]], emb_matrix[[i, 1]]));
        }

        Ok(DiffusionMapEvaluation {
            total_samples,
            points_2d,
            embedded_2d,
        })
    }

    /// Gaussian Random Projection
    pub fn run_gaussian_projection() -> Result<ProjectionEvaluation> {
        let mut rng = StdRng::seed_from_u64(42);
        let expected_centroids = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.]];
        let dataset = Dataset::from(generate::blobs(20, &expected_centroids, &mut rng));

        let projection = GaussianRandomProjection::<f64>::params()
            .target_dim(2)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("Gaussian Projection Fit failed: {:?}", e))?;

        let reduced = projection.transform(dataset);
        let records = reduced.records();
        let total_samples = records.nrows();

        let mut points_2d = Vec::with_capacity(total_samples);
        for i in 0..total_samples {
            points_2d.push((records[[i, 0]], records[[i, 1]]));
        }

        Ok(ProjectionEvaluation {
            method_name: "Gaussian Random Projection".into(),
            total_samples,
            original_dim: 2,
            reduced_dim: 2,
            points_2d,
        })
    }

    /// Sparse Random Projection
    pub fn run_sparse_projection() -> Result<ProjectionEvaluation> {
        let mut rng = StdRng::seed_from_u64(42);
        let expected_centroids = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.]];
        let dataset = Dataset::from(generate::blobs(20, &expected_centroids, &mut rng));

        let projection = SparseRandomProjection::<f64>::params()
            .target_dim(2)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("Sparse Projection Fit failed: {:?}", e))?;

        let reduced = projection.transform(dataset);
        let records = reduced.records();
        let total_samples = records.nrows();

        let mut points_2d = Vec::with_capacity(total_samples);
        for i in 0..total_samples {
            points_2d.push((records[[i, 0]], records[[i, 1]]));
        }

        Ok(ProjectionEvaluation {
            method_name: "Sparse Random Projection".into(),
            total_samples,
            original_dim: 2,
            reduced_dim: 2,
            points_2d,
        })
    }

    pub fn save_to_models<T: Serialize>(data: &T, filename: &str) -> Result<()> {
        let dir = Path::new("models");
        if !dir.exists() {
            fs::create_dir_all(dir)?;
        }
        let file_path = dir.join(filename);
        let json_data = serde_json::to_string_pretty(data)?;
        fs::write(file_path, json_data)?;
        Ok(())
    }
}
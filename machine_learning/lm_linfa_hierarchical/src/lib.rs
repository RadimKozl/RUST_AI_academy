use anyhow::Result;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalEvaluation {
    pub num_clusters: usize,
    pub total_samples: usize,
    pub num_features: usize,
    pub cluster_assignments: Vec<usize>,
    pub features_2d: Vec<(f64, f64)>,
}

pub struct HierarchicalPipeline;

impl HierarchicalPipeline {
    pub fn train_hierarchical(num_clusters: usize) -> Result<HierarchicalEvaluation> {
        let dataset = linfa_datasets::winequality();
        let (total_samples, num_features) = dataset.records().dim();

        let records = dataset.records();
        let mut features_2d = Vec::with_capacity(total_samples);
        for row in records.rows() {
            features_2d.push((row[0] as f64, row[1] as f64));
        }

        // Create and fit a KMeans model without having to pass an RNG
        let model = KMeans::params(num_clusters)
            .max_n_iterations(100)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("Error training clustering: {:?}", e))?;

        let cluster_dataset = model.predict(dataset);
        let cluster_assignments: Vec<usize> = cluster_dataset.targets().to_vec();

        Ok(HierarchicalEvaluation {
            num_clusters,
            total_samples,
            num_features,
            cluster_assignments,
            features_2d,
        })
    }

    pub fn save_to_models(eval: &HierarchicalEvaluation, filename: &str) -> Result<()> {
        let dir = Path::new("models");
        if !dir.exists() {
            fs::create_dir_all(dir)?;
        }
        let file_path = dir.join(filename);
        let json_data = serde_json::to_string_pretty(eval)?;
        fs::write(file_path, json_data)?;
        Ok(())
    }
}
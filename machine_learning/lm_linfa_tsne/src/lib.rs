use anyhow::Result;
use linfa::traits::{Fit, Transformer};
use linfa_datasets::iris;
use linfa_reduction::Pca;
use linfa_tsne::TSneParams;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
    pub label: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsneResult {
    pub points: Vec<Point2D>,
    pub perplexity: f64,
    pub approx_threshold: f64,
}

pub struct TsnePipeline;

impl TsnePipeline {
    /// Runs PCA (in 3 dimensions) and then t-SNE (in 2 dimensions) on the Iris dataset
    pub fn run_iris_tsne(perplexity: f64, approx_threshold: f64) -> Result<TsneResult> {
        let ds = iris();

        // 1. Dimensionality reduction using PCA
        let pca_ds = Pca::params(3)
            .whiten(true)
            .fit(&ds)
            .map_err(|e| anyhow::anyhow!("PCA error: {}", e))?
            .transform(ds);

        // 2. Calculation of two-dimensional t-SNE nesting
        let tsne_ds = TSneParams::embedding_size(2)
            .perplexity(perplexity)
            .approx_threshold(approx_threshold)
            .transform(pca_ds)
            .map_err(|e| anyhow::anyhow!("t-SNE error: {}", e))?;

        // 3. Extraction of the resulting 2D coordinates and labels
        let mut points = Vec::new();
        for (x, y) in tsne_ds.sample_iter() {
            points.push(Point2D {
                x: x[0],
                y: x[1],
                label: y.into_scalar().clone(), // Convert 0D ArrayView to usize value
            });
        }

        Ok(TsneResult {
            points,
            perplexity,
            approx_threshold,
        })
    }

    /// Save the resulting t-SNE embedding to a JSON file
    pub fn save_result(result: &TsneResult, path: impl AsRef<Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(result)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }
}
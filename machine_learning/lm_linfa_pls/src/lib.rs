use anyhow::Result;
use linfa::prelude::*;
use linfa_pls::PlsRegression;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlsEvaluation {
    pub n_samples: usize,
    pub n_features: usize,
    pub n_targets: usize,
    pub true_b: Vec<Vec<f64>>,
    pub estimated_b: Vec<Vec<f64>>,
}

/// Simple deterministic generator (LCG) avoiding version-mismatch in `rand_core`
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns a float in the range <-1.0, 1.0>
    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let val = (self.state >> 33) as f64 / (u32::MAX as f64);
        (val * 2.0) - 1.0
    }
}

pub struct PlsPipeline;

impl PlsPipeline {
    /// Generate synthetic data and train PLS regression
    pub fn train_pls(
        n: usize,
        p: usize,
        q: usize,
        seed: u64,
        n_components: usize,
    ) -> Result<(PlsRegression<f64>, PlsEvaluation)> {
        let mut rng = SimpleRng::new(seed);

        // Generate the feature matrix X(n, p)
        let x_vec: Vec<f64> = (0..(n * p)).map(|_| rng.next_f64()).collect();
        let x: Array2<f64> = Array2::from_shape_vec((n, p), x_vec)?;

        // Matrix of real coefficients B (p, q)
        let mut b: Array2<f64> = Array2::zeros((p, q));
        b.row_mut(0).assign(&Array1::ones(q));
        b.row_mut(1).assign(&Array1::from_elem(q, 2.0));

        // 1. Generate noise without offset (+5.0 removed, reduced variance to 0.1)
        let noise_vec: Vec<f64> = (0..(n * q)).map(|_| rng.next_f64() * 0.1).collect();
        let noise: Array2<f64> = Array2::from_shape_vec((n, q), noise_vec)?;
        let y = x.dot(&b) + noise;

        let ds = Dataset::new(x, y);

        // 2. Turn off automatic scaling to preserve the original scale of coefficients
        let fitted_model = PlsRegression::params(n_components)
            .scale(false)
            .max_iterations(200)
            .fit(&ds)?;

        // Convert matrices for serialization
        let true_b_vec = b.outer_iter().map(|row| row.to_vec()).collect();
        let est_b_vec = fitted_model
            .coefficients()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();

        let eval = PlsEvaluation {
            n_samples: n,
            n_features: p,
            n_targets: q,
            true_b: true_b_vec,
            estimated_b: est_b_vec,
        };

        Ok((fitted_model, eval))
    }

    /// Saves data to `models/` folder
    pub fn save_to_models<T: Serialize>(data: &T, filename: &str) -> Result<()> {
        let dir = Path::new("models");
        if !dir.exists() {
            create_dir_all(dir)?;
        }
        let path = dir.join(filename);
        let json = serde_json::to_string_pretty(data)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }
}
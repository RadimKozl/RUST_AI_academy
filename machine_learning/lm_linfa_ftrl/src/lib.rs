use anyhow::Result;
use linfa::dataset::AsSingleTargets;
use linfa::prelude::*;
use linfa_ftrl::Ftrl;
use rand::{rngs::SmallRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FtrlEvaluation {
    pub alpha: f64,
    pub beta: f64,
    pub l1_ratio: f64,
    pub l2_ratio: f64,
    pub train_samples: usize,
    pub valid_samples: usize,
    pub valid_log_loss: f64,
    pub target_probabilities: Vec<f64>,
    pub true_targets: Vec<bool>,
}

pub struct FtrlPipeline;

impl FtrlPipeline {
    pub fn train_ftrl(
        alpha: f64,
        beta: f64,
        l1_ratio: f64,
        l2_ratio: f64,
        seed: u64,
    ) -> Result<FtrlEvaluation> {
        let (train, valid) = linfa_datasets::winequality()
            .map_targets(|v| *v > 6)
            .split_with_ratio(0.9);

        let params = Ftrl::params()
            .alpha(alpha)
            .beta(beta)
            .l1_ratio(l1_ratio)
            .l2_ratio(l2_ratio);

        let valid_params = params.clone().check_unwrap();
        let mut model = Ftrl::new(valid_params, train.nfeatures());

        // Bootstrap by rows (online data flow simulation)
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut row_iter = train.bootstrap_samples(1, &mut rng);

        for _ in 0..train.nsamples() {
            let b_dataset = row_iter.next().ok_or_else(|| anyhow::anyhow!("Iteration failed"))?;
            model = params.fit_with(Some(model), &b_dataset)
                .map_err(|e| anyhow::anyhow!("FTLR fit error: {:?}", e))?;
        }

        let val_predictions = model.predict(&valid);
        let true_targets_vec = valid.as_single_targets().to_vec();

        // Calculating Log Loss
        let log_loss = val_predictions
            .log_loss(&true_targets_vec)
            .map_err(|e| anyhow::anyhow!("Error calculating Log Loss: {:?}", e))?;

        // Convert Pr values ​​to f64 using dereference *p
        let probabilities: Vec<f64> = val_predictions
            .iter()
            .map(|&p| f64::from(*p))
            .collect();

        Ok(FtrlEvaluation {
            alpha,
            beta,
            l1_ratio,
            l2_ratio,
            train_samples: train.nsamples(),
            valid_samples: valid.nsamples(),
            valid_log_loss: log_loss as f64,
            target_probabilities: probabilities,
            true_targets: true_targets_vec,
        })
    }

    pub fn save_to_models(eval: &FtrlEvaluation, filename: &str) -> Result<()> {
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
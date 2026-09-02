use anyhow::Result;
use linfa::prelude::*;
use linfa_datasets::iris;
use linfa_ensemble::{AdaBoostParams, RandomForestParams};
use linfa_trees::DecisionTree;
use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct EnsembleResult {
    pub algorithm_name: String,
    pub accuracy: f64,
    pub predictions: Vec<usize>,
}

const MODELS_DIR: &str = "models";

pub fn run_adaboost(n_estimators: usize, learning_rate: f64, max_depth: usize) -> Result<EnsembleResult> {
    let dataset = iris();

    let model = AdaBoostParams::new(DecisionTree::params().max_depth(Some(max_depth)))
        .n_estimators(n_estimators)
        .learning_rate(learning_rate)
        .fit(&dataset)
        .map_err(|e| anyhow::anyhow!("AdaBoost Fit Error: {:?}", e))?;

    let predictions = model.predict(&dataset);
    let cm = predictions.confusion_matrix(&dataset)?;
    let accuracy = cm.accuracy() * 100.0;

    let result = EnsembleResult {
        algorithm_name: format!("AdaBoost (depth={}, n_est={})", max_depth, n_estimators),
        accuracy: f64::from(accuracy),
        predictions: predictions.to_vec(),
    };

    save_result(&result, "adaboost_result.json")?;

    Ok(result)
}

pub fn run_random_forest(ensemble_size: usize, bootstrap_prop: f64, feature_prop: f64) -> Result<EnsembleResult> {
    let dataset = iris();

    let model = RandomForestParams::new(DecisionTree::params())
        .ensemble_size(ensemble_size)
        .bootstrap_proportion(bootstrap_prop)
        .feature_proportion(feature_prop)
        .fit(&dataset)
        .map_err(|e| anyhow::anyhow!("RandomForest Fit Error: {:?}", e))?;

    let predictions = model.predict(&dataset);
    let cm = predictions.confusion_matrix(&dataset)?;
    let accuracy = cm.accuracy() * 100.0;

    let result = EnsembleResult {
        algorithm_name: format!("RandomForest (size={})", ensemble_size),
        accuracy: f64::from(accuracy),
        predictions: predictions.to_vec(),
    };

    save_result(&result, "random_forest_result.json")?;

    Ok(result)
}

fn save_result(result: &EnsembleResult, filename: &str) -> Result<()> {
    create_dir_all(MODELS_DIR)?;
    let path = Path::new(MODELS_DIR).join(filename);
    let json_data = serde_json::to_string_pretty(result)?;
    let mut file = File::create(path)?;
    file.write_all(json_data.as_bytes())?;
    Ok(())
}
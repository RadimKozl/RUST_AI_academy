use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use smartcore::dataset::iris;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters};
use smartcore::metrics::accuracy;
use smartcore::model_selection::train_test_split;
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::PathBuf;

pub type ModelType = LogisticRegression<f64, usize, DenseMatrix<f64>, Vec<usize>>;

#[derive(Debug, Serialize, Deserialize)]
pub struct LogisticRegressionResult {
    pub algorithm_name: String,
    pub accuracy: f64,
    pub test_predictions: Vec<usize>,
}

pub fn get_model_path(filename: &str) -> Result<PathBuf> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map_err(|_| anyhow!("CARGO_MANIFEST_DIR is not set"))?;
    
    let path = PathBuf::from(manifest_dir).join("models").join(filename);
    Ok(path)
}

pub fn run_logistic_regression() -> Result<LogisticRegressionResult> {
    let dataset = iris::load_dataset();

    let x_f64: Vec<f64> = dataset.data.iter().map(|&v| v as f64).collect();
    let x = DenseMatrix::new(dataset.num_samples, dataset.num_features, x_f64, false)
        .map_err(|e| anyhow!("Matrix creation error: {:?}", e))?;

    let y: Vec<usize> = dataset.target.iter().map(|&val| val as usize).collect();

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(42));

    let params = LogisticRegressionParameters::default();
    let model = LogisticRegression::fit(&x_train, &y_train, params)
        .map_err(|e| anyhow!("Fit Error: {:?}", e))?;

    let y_pred = model.predict(&x_test)
        .map_err(|e| anyhow!("Predict Error: {:?}", e))?;

    let acc = accuracy(&y_test, &y_pred) * 100.0;

    save_model(&model, "logistic_regression.bin")?;

    Ok(LogisticRegressionResult {
        algorithm_name: "Logistic Regression".to_string(),
        accuracy: acc,
        test_predictions: y_pred,
    })
}

fn save_model(model: &ModelType, filename: &str) -> Result<()> {
    let path = get_model_path(filename)?;

    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }

    let bytes = bincode::serialize(model)
        .map_err(|e| anyhow!("Bincode serialization failed: {:?}", e))?;

    let mut file = File::create(&path)
        .map_err(|e| anyhow!("Failed to create file at path {:?}: {:?}", path, e))?;

    file.write_all(&bytes)
        .map_err(|e| anyhow!("Writing to file failed: {:?}", e))?;

    file.sync_all()
        .map_err(|e| anyhow!("Disk sync failed:: {:?}", e))?;

    Ok(())
}

pub fn load_model(filename: &str) -> Result<ModelType> {
    let path = get_model_path(filename)?;
    
    let mut file = File::open(&path)
        .map_err(|e| anyhow!("Opening file at path {:?} failed: {:?}", path, e))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let model: ModelType = bincode::deserialize(&buffer)
        .map_err(|e| anyhow!("Bincode deserialization failed: {:?}", e))?;

    Ok(model)
}
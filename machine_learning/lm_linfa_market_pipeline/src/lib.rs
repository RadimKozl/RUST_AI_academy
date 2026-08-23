use anyhow::{Context, Result};
use linfa::prelude::*;
use linfa_trees::DecisionTree;
use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

pub struct MarketMLPipeline;

impl MarketMLPipeline {
    pub fn train_and_save_model(parquet_path: &Path, model_output_path: &Path) -> Result<f64> {
        let ref_path = PlRefPath::try_from_path(parquet_path)?;
        let df = LazyFrame::scan_parquet(ref_path, Default::default())?
            .collect()?;

        let price_col = df.column("last_price")?;
        let sma_col = df.column("sma_5")?;
        let mask = price_col.gt(sma_col)?;

        // Use .iter() to iterate through BooleanChunked elements
        let targets_vec: Vec<usize> = mask
            .iter()
            .map(|val| if val.unwrap_or(false) { 1 } else { 0 })
            .collect();

        let rows = df.height();
        let cols = 4;
        let mut flat_data = Vec::with_capacity(rows * cols);

        let col_names = ["last_price", "sma_5", "bband_upper", "bband_lower"];
        for i in 0..rows {
            for col_name in &col_names {
                let series = df.column(col_name)?;
                let val = series.f64()?.get(i).unwrap_or(0.0);
                flat_data.push(val);
            }
        }

        let features_matrix: Array2<f64> = Array2::from_shape_vec((rows, cols), flat_data)?;
        let targets_array: Array1<usize> = Array1::from(targets_vec);

        let dataset = Dataset::new(features_matrix, targets_array)
            .with_feature_names(vec!["last_price", "sma_5", "bband_upper", "bband_lower"]);

        let (train, test) = dataset.split_with_ratio(0.8);

        let model = DecisionTree::params()
            .max_depth(Some(5))
            .fit(&train)
            .map_err(|e| anyhow::anyhow!("Training failed: {}", e))?;

        let predictions = model.predict(&test);
        let cm = predictions.confusion_matrix(&test)?;
        let accuracy: f32 = cm.accuracy();

        if let Some(parent) = model_output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(model_output_path)
            .with_context(|| format!("Cannot create file {:?}", model_output_path))?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &model)?;

        Ok(accuracy as f64)
    }

    pub fn predict_live_sample(model_path: &Path, features: Vec<f64>) -> Result<usize> {
        if features.len() != 4 {
            anyhow::bail!("Expected exactly 4 flags [last_price, sma_5, bband_upper, bband_lower].");
        }

        let file = File::open(model_path)
            .with_context(|| format!("Cannot open file with model {:?}", model_path))?;
        let reader = BufReader::new(file);
        let model: DecisionTree<f64, usize> = bincode::deserialize_from(reader)?;

        let input_matrix: Array2<f64> = Array2::from_shape_vec((1, 4), features)?;
        let prediction = model.predict(&input_matrix);

        Ok(prediction[0])
    }
}
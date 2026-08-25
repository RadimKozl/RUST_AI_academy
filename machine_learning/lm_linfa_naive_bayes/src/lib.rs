use anyhow::{Context, Result};
use linfa::metrics::ToConfusionMatrix;
use linfa::prelude::*;
use linfa_bayes::{BernoulliNb, GaussianNb, MultinomialNb};
use ndarray::{Array1, Array2};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BayesType {
    Gaussian,
    Bernoulli,
    Multinomial,
}

pub enum TrainedBayesModel {
    Gaussian(GaussianNb<f64, usize>),
    Bernoulli(BernoulliNb<f64, usize>),
    Multinomial(MultinomialNb<f64, usize>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedModel {
    pub model_type: BayesType,
    pub accuracy: f32,
    pub mcc: f32,
}

#[derive(Debug, Clone)]
pub struct ModelMetrics {
    pub model_type: BayesType,
    pub accuracy: f32,
    pub mcc: f32,
}

pub struct WineBayesPipeline;

impl WineBayesPipeline {
    pub fn load_dataset(
        csv_path: &Path,
    ) -> Result<(
        Dataset<f64, usize, ndarray::Dim<[usize; 1]>>,
        Dataset<f64, usize, ndarray::Dim<[usize; 1]>>,
    )> {
        let file = File::open(csv_path)
            .with_context(|| format!("Unable to open CSV file {:?}", csv_path))?;

        let df = CsvReader::new(file).finish()?;

        let rows = df.height();
        let cols = 11;

        let feature_cols = [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        ];

        let mut flat_data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for col_name in &feature_cols {
                let col = df.column(col_name)?;
                let series = col.as_materialized_series();
                let val = series.f64()?.get(i).unwrap_or(0.0);
                flat_data.push(val);
            }
        }

        let quality_col = df.column("quality")?;
        let quality_series = quality_col.as_materialized_series();
        let quality_ca = quality_series.i64()?;

        let targets_vec: Vec<usize> = quality_ca
            .iter()
            .map(|val| if val.unwrap_or(0) > 6 { 1 } else { 0 })
            .collect();

        let features_matrix = Array2::from_shape_vec((rows, cols), flat_data)?;
        let targets_array = Array1::from(targets_vec);

        let dataset = Dataset::new(features_matrix, targets_array)
            .with_feature_names(feature_cols.to_vec());

        Ok(dataset.split_with_ratio(0.9))
    }

    pub fn train_and_eval(
        train: &Dataset<f64, usize, ndarray::Dim<[usize; 1]>>,
        valid: &Dataset<f64, usize, ndarray::Dim<[usize; 1]>>,
        bayes_type: BayesType,
    ) -> Result<(TrainedBayesModel, ModelMetrics)> {
        let (model, accuracy, mcc) = match bayes_type {
            BayesType::Gaussian => {
                let model = GaussianNb::params().fit(train)?;
                let pred = model.predict(valid);
                let cm = pred.confusion_matrix(valid)?;
                (TrainedBayesModel::Gaussian(model), cm.accuracy(), cm.mcc())
            }
            BayesType::Bernoulli => {
                let model = BernoulliNb::params().fit(train)?;
                let pred = model.predict(valid);
                let cm = pred.confusion_matrix(valid)?;
                (TrainedBayesModel::Bernoulli(model), cm.accuracy(), cm.mcc())
            }
            BayesType::Multinomial => {
                let model = MultinomialNb::params().fit(train)?;
                let pred = model.predict(valid);
                let cm = pred.confusion_matrix(valid)?;
                (TrainedBayesModel::Multinomial(model), cm.accuracy(), cm.mcc())
            }
        };

        let metrics = ModelMetrics {
            model_type: bayes_type,
            accuracy,
            mcc,
        };

        Ok((model, metrics))
    }

    pub fn save_metrics(metrics: &ModelMetrics, output_path: &Path) -> Result<()> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let saved = SavedModel {
            model_type: metrics.model_type,
            accuracy: metrics.accuracy,
            mcc: metrics.mcc,
        };
        let file = File::create(output_path)
            .with_context(|| format!("Cannot create file {:?}", output_path))?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &saved)?;
        Ok(())
    }

    pub fn predict_model(
        model: &TrainedBayesModel,
        features: &[f64],
    ) -> Result<usize> {
        if features.len() != 11 {
            anyhow::bail!("Exactly 11 symptoms expected.");
        }

        let input_matrix = Array2::from_shape_vec((1, 11), features.to_vec())?;

        let prediction = match model {
            TrainedBayesModel::Gaussian(m) => m.predict(&input_matrix),
            TrainedBayesModel::Bernoulli(m) => m.predict(&input_matrix),
            TrainedBayesModel::Multinomial(m) => m.predict(&input_matrix),
        };

        Ok(prediction[0])
    }
}
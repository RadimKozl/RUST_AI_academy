use anyhow::{Context, Result};
use linfa::dataset::Dataset;
use linfa::prelude::*;
use linfa_lars::Lars;
use ndarray::{Array1, Array2};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub struct LarsPipeline;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CustomLarsModel {
    pub params: Vec<f64>,
    pub intercept: f64,
}

impl CustomLarsModel {
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let weights = Array1::from_vec(self.params.clone());
        x.dot(&weights) + self.intercept
    }
}

pub struct RegressionMetrics {
    pub r2: f64,
    pub mse: f64,
}

impl LarsPipeline {
    pub fn load_and_split_data(
        ratio: f32,
    ) -> (
        Dataset<f64, f64, ndarray::Dim<[usize; 1]>>,
        Dataset<f64, f64, ndarray::Dim<[usize; 1]>>,
    ) {
        linfa_datasets::diabetes().split_with_ratio(ratio)
    }

    pub fn train_model(
        train_data: &Dataset<f64, f64, ndarray::Dim<[usize; 1]>>,
    ) -> Result<CustomLarsModel> {
        let model = Lars::params()
            .fit_intercept(true)
            .fit(train_data)
            .map_err(|e| anyhow::anyhow!("Error while training LARS: {:?}", e))?;

        Ok(CustomLarsModel {
            params: model.hyperplane().to_vec(),
            intercept: model.intercept(),
        })
    }

    pub fn evaluate(
        model: &CustomLarsModel,
        test_data: &Dataset<f64, f64, ndarray::Dim<[usize; 1]>>,
    ) -> (Array1<f64>, RegressionMetrics) {
        let predictions = model.predict(test_data.records());
        let targets = test_data.targets();

        let r2 = test_data.r2(&predictions).unwrap_or(0.0);
        let mse = targets
            .iter()
            .zip(predictions.iter())
            .map(|(y, y_hat)| (y - y_hat).powi(2))
            .sum::<f64>()
            / targets.len() as f64;

        (predictions, RegressionMetrics { r2, mse })
    }

    pub fn save_model(model: &CustomLarsModel, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json_data = serde_json::to_string_pretty(model)?;
        let mut file = File::create(path)?;
        file.write_all(json_data.as_bytes())?;
        Ok(())
    }

    pub fn load_model(path: &Path) -> Result<CustomLarsModel> {
        let mut file = File::open(path)
            .with_context(|| format!("Unable to open file with model {:?}", path))?;
        let mut json_str = String::new();
        file.read_to_string(&mut json_str)?;
        let model: CustomLarsModel = serde_json::from_str(&json_str)?;
        Ok(model)
    }

    pub fn render_regression_plot(
        targets: &Array1<f64>,
        predictions: &Array1<f64>,
        output_file: &Path,
    ) -> Result<()> {
        if let Some(parent) = output_file.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let root = BitMapBackend::new(output_file, (600, 450)).into_drawing_area();
        root.fill(&WHITE)?;

        let min_val = targets
            .fold(f64::INFINITY, |a, &b| a.min(b))
            .min(predictions.fold(f64::INFINITY, |a, &b| a.min(b)));
        let max_val = targets
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            .max(predictions.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

        let mut chart = ChartBuilder::on(&root)
            .caption("Actual vs. Predicted Values ​​(LARS)", ("sans-serif", 20).into_font())
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(min_val..max_val, min_val..max_val)?;

        chart
            .configure_mesh()
            .x_desc("Real progression")
            .y_desc("Predicted progression")
            .draw()?;

        chart.draw_series(LineSeries::new(
            vec![(min_val, min_val), (max_val, max_val)],
            &RED,
        ))?;

        chart.draw_series(
            targets
                .iter()
                .zip(predictions.iter())
                .map(|(&y, &y_hat)| Circle::new((y, y_hat), 3, BLUE.filled())),
        )?;

        root.present()?;
        Ok(())
    }
}
use anyhow::{Context, Result};
use linfa::dataset::Dataset;
use linfa::prelude::*;
use linfa_linear::{FittedLinearRegression, LinearRegression};
use ndarray::{Array1, Array2};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub struct RegressionPipeline;

#[derive(Serialize, Deserialize, Clone)]
pub struct CustomLinearModel {
    pub params: Vec<f64>,
    pub intercept: f64,
}

impl CustomLinearModel {
    /// Calculates predictions for the specified feature matrix (X * w + intercept)
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let weights = Array1::from_vec(self.params.clone());
        x.dot(&weights) + self.intercept
    }
}

pub struct RegressionMetrics {
    pub r2_score: f64,
    pub mae: f64,
}

impl RegressionPipeline {
    /// Loads the Diabetes dataset and splits it into training and testing parts
    pub fn load_and_split_data(
        ratio: f32,
    ) -> (
        Dataset<f64, f64, ndarray::Dim<[usize; 1]>>,
        Dataset<f64, f64, ndarray::Dim<[usize; 1]>>,
    ) {
        let dataset = linfa_datasets::diabetes();
        dataset.split_with_ratio(ratio)
    }

    /// Trains a linear regression model using the linfa-linear library
    pub fn train_model(
        train_data: &Dataset<f64, f64, ndarray::Dim<[usize; 1]>>,
    ) -> Result<FittedLinearRegression<f64>> {
        let lin_reg = LinearRegression::new();
        let model = lin_reg.fit(train_data)?;
        Ok(model)
    }

    /// Calculates predictions and evaluates metrics (MAE and R²)
    pub fn evaluate(
        model: &CustomLinearModel,
        test_data: &Dataset<f64, f64, ndarray::Dim<[usize; 1]>>,
    ) -> (Array1<f64>, RegressionMetrics) {
        let predictions = model.predict(test_data.records());
        let targets = test_data.targets();

        let mae = (&predictions - targets).mapv(|x| x.abs()).mean().unwrap_or(0.0);

        let target_mean = targets.mean().unwrap_or(0.0);
        let ss_tot: f64 = targets.iter().map(|y| (y - target_mean).powi(2)).sum();
        let ss_res: f64 = targets
            .iter()
            .zip(predictions.iter())
            .map(|(y, y_pred)| (y - y_pred).powi(2))
            .sum();
        let r2_score = 1.0 - (ss_res / ss_tot);

        (predictions, RegressionMetrics { r2_score, mae })
    }

    /// Saves model parameters to JSON file
    pub fn save_model(model: &FittedLinearRegression<f64>, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let saved = CustomLinearModel {
            params: model.params().to_vec(),
            intercept: model.intercept(),
        };
        let json_data = serde_json::to_string_pretty(&saved)?;
        let mut file = File::create(path)?;
        file.write_all(json_data.as_bytes())?;
        Ok(())
    }

    /// Loads the model from a JSON file
    pub fn load_model(path: &Path) -> Result<CustomLinearModel> {
        let mut file = File::open(path)
            .with_context(|| format!("Unable to open file with model {:?}", path))?;
        let mut json_str = String::new();
        file.read_to_string(&mut json_str)?;
        let model: CustomLinearModel = serde_json::from_str(&json_str)?;
        Ok(model)
    }

    /// Draws a scatter plot (actual vs. predicted values)
    pub fn render_scatter_plot(
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
            .iter()
            .chain(predictions.iter())
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_val = targets
            .iter()
            .chain(predictions.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Diabetes: Actual vs Predicted", ("sans-serif", 20).into_font())
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(min_val..max_val, min_val..max_val)?;

        chart
            .configure_mesh()
            .x_desc("Actual value")
            .y_desc("Predicted Value (Predicted)")
            .draw()?;

        chart.draw_series(LineSeries::new(
            vec![(min_val, min_val), (max_val, max_val)],
            &RED,
        ))?;

        for (y_actual, y_pred) in targets.iter().zip(predictions.iter()) {
            chart.draw_series(std::iter::once(Circle::new(
                (*y_actual, *y_pred),
                3,
                ShapeStyle::from(&BLUE).filled(),
            )))?;
        }

        root.present()?;
        Ok(())
    }
}
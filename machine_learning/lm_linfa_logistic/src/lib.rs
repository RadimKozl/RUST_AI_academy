use anyhow::{Context, Result};
use linfa::dataset::Dataset;
use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{Array1, Array2};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub struct LogisticPipeline;

#[derive(Serialize, Deserialize, Clone)]
pub struct CustomLogisticModel {
    pub params: Vec<f64>,
    pub intercept: f64,
}

impl CustomLogisticModel {
    pub fn predict_probabilities(&self, x: &Array2<f64>) -> Array1<f64> {
        let weights = Array1::from_vec(self.params.clone());
        // Sign change: Linfa returns coefficients in opposite polarity
        let z = -(x.dot(&weights) + self.intercept);
        z.mapv(|val| 1.0 / (1.0 + (-val).exp()))
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<bool> {
        self.predict_probabilities(x).mapv(|prob| prob >= 0.5)
    }
}

pub struct ClassificationMetrics {
    pub accuracy: f32,
    pub mcc: f32,
}

impl LogisticPipeline {
    pub fn load_and_split_data(
        ratio: f32,
    ) -> (
        Dataset<f64, usize, ndarray::Dim<[usize; 1]>>,
        Dataset<f64, usize, ndarray::Dim<[usize; 1]>>,
    ) {
        // Convert bool target explicitly to usize: 1 = good wine (>6), 0 = others
        let raw_dataset = linfa_datasets::winequality();
        let records = raw_dataset.records().to_owned();
        let targets = raw_dataset.targets().mapv(|x| if x > 6 { 1usize } else { 0usize });

        // Standardization of symptoms (z-score scaling)
        let mean = records.mean_axis(ndarray::Axis(0)).unwrap();
        let mut std = records.std_axis(ndarray::Axis(0), 0.0);
        std.mapv_inplace(|v| if v == 0.0 { 1.0 } else { v });

        let normalized_records = (&records - &mean) / &std;
        let dataset = Dataset::new(normalized_records, targets);

        dataset.split_with_ratio(ratio)
    }

    pub fn train_model(
        train_data: &Dataset<f64, usize, ndarray::Dim<[usize; 1]>>,
    ) -> Result<CustomLogisticModel> {
        let model = LogisticRegression::default()
            .max_iterations(500)
            .alpha(0.01)
            .fit(train_data)?;

        Ok(CustomLogisticModel {
            params: model.params().to_vec(),
            intercept: model.intercept(),
        })
    }

    pub fn evaluate(
        model: &CustomLogisticModel,
        test_data: &Dataset<f64, usize, ndarray::Dim<[usize; 1]>>,
    ) -> (Array1<bool>, ClassificationMetrics) {
        let probabilities = model.predict_probabilities(test_data.records());
        let predictions = probabilities.mapv(|p| p >= 0.5);
        let targets = test_data.targets();

        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut fn_val = 0;

        for (pred, &target) in predictions.iter().zip(targets.iter()) {
            let target_bool = target == 1;
            match (pred, target_bool) {
                (true, true) => tp += 1,
                (false, false) => tn += 1,
                (true, false) => fp += 1,
                (false, true) => fn_val += 1,
            }
        }

        let total = (tp + tn + fp + fn_val) as f32;
        let accuracy = (tp + tn) as f32 / total;

        let numerator = (tp * tn) as f64 - (fp * fn_val) as f64;
        let denominator = (((tp + fp) * (tp + fn_val) * (tn + fp) * (tn + fn_val)) as f64).sqrt();

        let mcc = if denominator == 0.0 {
            0.0
        } else {
            (numerator / denominator) as f32
        };

        (predictions, ClassificationMetrics { accuracy, mcc })
    }

    pub fn save_model(model: &CustomLogisticModel, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json_data = serde_json::to_string_pretty(model)?;
        let mut file = File::create(path)?;
        file.write_all(json_data.as_bytes())?;
        Ok(())
    }

    pub fn load_model(path: &Path) -> Result<CustomLogisticModel> {
        let mut file = File::open(path)
            .with_context(|| format!("Unable to open file with model {:?}", path))?;
        let mut json_str = String::new();
        file.read_to_string(&mut json_str)?;
        let model: CustomLogisticModel = serde_json::from_str(&json_str)?;
        Ok(model)
    }

    pub fn render_probability_plot(
        probabilities: &Array1<f64>,
        output_file: &Path,
    ) -> Result<()> {
        if let Some(parent) = output_file.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let root = BitMapBackend::new(output_file, (600, 450)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Predicted probability distribution", ("sans-serif", 20).into_font())
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..1.0, 0..100)?;

        chart
            .configure_mesh()
            .x_desc("Probability (Good Wine)")
            .y_desc("Number of samples")
            .draw()?;

        let mut bins = vec![0; 10];
        for &prob in probabilities.iter() {
            let idx = ((prob * 10.0).floor() as usize).min(9);
            bins[idx] += 1;
        }

        chart.draw_series(
            bins.into_iter().enumerate().map(|(i, count)| {
                let x0 = i as f64 / 10.0;
                let x1 = (i + 1) as f64 / 10.0;
                Rectangle::new([(x0, 0), (x1, count)], BLUE.filled())
            }),
        )?;

        root.present()?;
        Ok(())
    }
}
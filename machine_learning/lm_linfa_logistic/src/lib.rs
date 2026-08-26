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
    /// Manual calculation of the sigmoidal function for probability prediction
    pub fn predict_probabilities(&self, x: &Array2<f64>) -> Array1<f64> {
        let weights = Array1::from_vec(self.params.clone());
        let z = x.dot(&weights) + self.intercept;
        z.mapv(|val| 1.0 / (1.0 + (-val).exp()))
    }

    /// Classification based on threshold 0.5 (true = "good", false = "bad")
    pub fn predict(&self, x: &Array2<f64>) -> Array1<bool> {
        self.predict_probabilities(x).mapv(|prob| prob >= 0.5)
    }
}

pub struct ClassificationMetrics {
    pub accuracy: f32,
    pub mcc: f32,
}

impl LogisticPipeline {
    /// Loads the WineQuality dataset and converts scores > 6 to "good" (true) and the rest to "bad" (false)
    pub fn load_and_split_data(
        ratio: f32,
    ) -> (
        Dataset<f64, bool, ndarray::Dim<[usize; 1]>>,
        Dataset<f64, bool, ndarray::Dim<[usize; 1]>>,
    ) {
        let dataset = linfa_datasets::winequality()
            .map_targets(|x| *x > 6); // Binary target: true = good wine, false = bad

        dataset.split_with_ratio(ratio)
    }

    /// Trains logistic regression
    pub fn train_model(
        train_data: &Dataset<f64, bool, ndarray::Dim<[usize; 1]>>,
    ) -> Result<CustomLogisticModel> {
        let model = LogisticRegression::default()
            .max_iterations(150)
            .fit(train_data)?;

        // Convert the trained weights into our serializable structure
        let params = model.params().to_vec();
        let intercept = model.intercept();

        Ok(CustomLogisticModel { params, intercept })
    }

    /// Evaluates the model and returns the Confusion Matrix metrics
    pub fn evaluate(
        model: &CustomLogisticModel,
        test_data: &Dataset<f64, bool, ndarray::Dim<[usize; 1]>>,
    ) -> (Array1<bool>, ClassificationMetrics) {
        let predictions = model.predict(test_data.records());
        
        // We will create a Dataset with predictions for calculating the Confusion Matrix via Linfa
        let pred_dataset = Dataset::new(test_data.records().clone(), predictions.clone());
        let cm = pred_dataset.confusion_matrix(test_data).unwrap();

        let metrics = ClassificationMetrics {
            accuracy: cm.accuracy(),
            mcc: cm.mcc(),
        };

        (predictions, metrics)
    }

    /// Saves model parameters to JSON file
    pub fn save_model(model: &CustomLogisticModel, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json_data = serde_json::to_string_pretty(model)?;
        let mut file = File::create(path)?;
        file.write_all(json_data.as_bytes())?;
        Ok(())
    }

    /// Loads the model from a JSON file
    pub fn load_model(path: &Path) -> Result<CustomLogisticModel> {
        let mut file = File::open(path)
            .with_context(|| format!("Unable to open file with model {:?}", path))?;
        let mut json_str = String::new();
        file.read_to_string(&mut json_str)?;
        let model: CustomLogisticModel = serde_json::from_str(&json_str)?;
        Ok(model)
    }

    /// Plots a histogram of the calculated probabilities
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
            .build_cartesian_2d(0.0..1.0, 0..50)?;

        chart
            .configure_mesh()
            .x_desc("Probability (Classification as Good Wine)")
            .y_desc("Number of samples")
            .draw()?;

        // Create a histogram of 10 bins
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
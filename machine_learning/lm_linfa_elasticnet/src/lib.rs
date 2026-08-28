use anyhow::Result;
use linfa::dataset::DatasetBase;
use linfa::prelude::*;
use linfa_elasticnet::{ElasticNet, ElasticNetParams};
use ndarray::{ArrayBase, Dim, OwnedRepr};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub type DiabetesDataset = DatasetBase<
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CvResult {
    pub l1_ratio: f64,
    pub r2_score: f64,
}

/// Custom structure for serializing trained model weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedElasticNetModel {
    pub intercept: f64,
    pub params: Vec<f64>,
}

pub struct ElasticNetPipeline;

impl ElasticNetPipeline {
    /// Loads the Diabetes dataset from linfa-datasets
    pub fn load_dataset() -> DiabetesDataset {
        linfa_datasets::diabetes()
    }

    /// Runs 5-fold cross validation for the specified L1 ratios
    pub fn run_cross_validation(
        dataset: &mut DiabetesDataset,
        ratios: &[f64],
        penalty: f64,
    ) -> Result<Vec<CvResult>> {
        let models: Vec<ElasticNetParams<f64>> = ratios
            .iter()
            .map(|&ratio| ElasticNet::params().penalty(penalty).l1_ratio(ratio))
            .collect();

        let r2_values = dataset.cross_validate_single(5, &models, |prediction, truth| {
            Ok(prediction.r2(truth).unwrap_or(0.0))
        })?;

        let results = ratios
            .iter()
            .zip(r2_values.iter())
            .map(|(&l1_ratio, &r2_score)| CvResult { l1_ratio, r2_score })
            .collect();

        Ok(results)
    }

    /// Saves the trained ElasticNet model (its coefficients) to a JSON file
    pub fn save_model(model: &ElasticNet<f64>, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let saved_data = SavedElasticNetModel {
            intercept: model.intercept(),
            params: model.hyperplane().to_vec(),
        };

        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, &saved_data)?;
        Ok(())
    }

    /// Loads the storage model from a JSON file
    pub fn load_model(path: &Path) -> Result<SavedElasticNetModel> {
        let file = std::fs::File::open(path)?;
        let model: SavedElasticNetModel = serde_json::from_reader(file)?;
        Ok(model)
    }

    /// Generates and saves a graph of the dependence of R² score on L1 ratio in the img/ folder
    pub fn render_cv_plot(results: &[CvResult], output_path: &Path) -> Result<()> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let root = BitMapBackend::new(output_path, (600, 350)).into_drawing_area();
        root.fill(&WHITE)?;

        let min_r2 = results
            .iter()
            .map(|r| r.r2_score)
            .fold(f64::INFINITY, f64::min)
            .min(0.0);
        let max_r2 = results
            .iter()
            .map(|r| r.r2_score)
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.6);

        let mut chart = ChartBuilder::on(&root)
            .caption("ElasticNet CV: L1 Ratio vs R² Score", ("sans-serif", 18))
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0f64..1.0f64, min_r2..max_r2)?;

        chart
            .configure_mesh()
            .x_desc("L1 Ratio (0 = Ridge, 1 = LASSO)")
            .y_desc("R² Score")
            .draw()?;

        let points: Vec<(f64, f64)> = results.iter().map(|r| (r.l1_ratio, r.r2_score)).collect();

        chart.draw_series(LineSeries::new(points.clone(), &RED))?;
        chart.draw_series(points.iter().map(|&(x, y)| Circle::new((x, y), 4, RED.filled())))?;

        root.present()?;
        Ok(())
    }
}
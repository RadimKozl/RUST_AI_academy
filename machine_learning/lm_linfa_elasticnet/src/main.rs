use anyhow::Result;
use eframe::egui;
use linfa::prelude::*;
use linfa_elasticnet::ElasticNet;
use lm_linfa_elasticnet::{CvResult, ElasticNetPipeline};
use std::path::Path;

struct AppState {
    cv_results: Vec<CvResult>,
    best_l1_ratio: f64,
    best_r2: f64,
    img_bytes: Option<Vec<u8>>,
    model_saved_path: String,
}

fn main() -> Result<()> {
    println!("📖 Loading Diabetes dataset...");
    let mut dataset = ElasticNetPipeline::load_dataset();

    let ratios = vec![0.0, 0.1, 0.2, 0.5, 0.7, 1.0];
    let penalty = 0.3;

    println!("📊 Running 5-Fold Cross Validation for ElasticNet...");
    let cv_results = ElasticNetPipeline::run_cross_validation(&mut dataset, &ratios, penalty)?;

    // Finding the best L1 ratio from cross validation
    let best = cv_results
        .iter()
        .max_by(|a, b| a.r2_score.partial_cmp(&b.r2_score).unwrap())
        .cloned()
        .unwrap_or(CvResult {
            l1_ratio: 0.0,
            r2_score: 0.0,
        });

    println!("🏋️ Training the final best model (L1 ratio = {})...", best.l1_ratio);
    let best_model = ElasticNet::params()
        .penalty(penalty)
        .l1_ratio(best.l1_ratio)
        .fit(&dataset)?;

    // Save the model to disk in the models/ folder
    let model_path = Path::new("models/elasticnet_diabetes_model.json");
    println!("💾 Saving trained model to '{:?}'...", model_path);
    ElasticNetPipeline::save_model(&best_model, model_path)?;

    // Generate the graph into the img/ folder
    let img_path = Path::new("img/elasticnet_cv_plot.png");
    println!("🖼️ Generating cross-validation graph into '{:?}'...", img_path);
    ElasticNetPipeline::render_cv_plot(&cv_results, img_path)?;

    // Load the graph bytes directly into memory for display in the UI
    let img_bytes = std::fs::read(img_path).ok();

    let state = AppState {
        cv_results,
        best_l1_ratio: best.l1_ratio,
        best_r2: best.r2_score,
        img_bytes,
        model_saved_path: model_path.to_string_lossy().into_owned(),
    };

    println!("🖥️ Opening Dashboard UI...");
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([650.0, 650.0])
            .with_title("ElasticNet CV Dashboard"),
        ..Default::default()
    };

    eframe::run_native(
        "ElasticNet Model Validation",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(ElasticNetApp::new(state)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("Application UI error: {:?}", e))
}

struct ElasticNetApp {
    state: AppState,
}

impl ElasticNetApp {
    fn new(state: AppState) -> Self {
        Self { state }
    }
}

impl eframe::App for ElasticNetApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("📊 ElasticNet Cross Validation Results");
        ui.separator();

        // Information summary
        ui.horizontal(|ui| {
            ui.label("Best L1 Ratio:");
            ui.label(format!("{:.2}", self.state.best_l1_ratio));
            ui.add_space(20.0);
            ui.label("Highest R^2 Score:");
            ui.label(format!("{:.4}", self.state.best_r2));
        });

        ui.label(format!("💾 Model saved: {}", self.state.model_saved_path));
        ui.add_space(10.0);

        // Cross validation results table
        egui::Grid::new("cv_grid")
            .num_columns(2)
            .spacing([40.0, 6.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("L1 Ratio (Penalization)");
                ui.label("Average R^2 Score");
                ui.end_row();

                for item in &self.state.cv_results {
                    ui.label(format!("{:.2}", item.l1_ratio));
                    ui.label(format!("{:.4}", item.r2_score));
                    ui.end_row();
                }
            });

        ui.add_space(15.0);
        ui.heading("🖼️ Graph of the Impact of L1 Ratio on Performance");
        ui.separator();

        // Display a graph from memory bytes
        if let Some(bytes) = &self.state.img_bytes {
            ui.add(
                egui::Image::from_bytes("bytes://elasticnet_cv_plot.png", bytes.clone())
                    .max_width(600.0)
                    .corner_radius(5.0),
            );
        } else {
            ui.label("⚠️ Chart in folder 'img/' not found.");
        }
    }
}
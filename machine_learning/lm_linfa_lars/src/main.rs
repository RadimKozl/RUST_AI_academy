use anyhow::Result;
use eframe::egui;
use linfa::dataset::Records;
use lm_linfa_lars::{LarsPipeline, RegressionMetrics};
use std::path::Path;

struct AppState {
    intercept: f64,
    test_samples: usize,
    r2: f64,
    mse: f64,
    img_bytes: Option<Vec<u8>>,
}

fn main() -> Result<()> {
    println!("📖 Loading and splitting the Diabetes dataset (90% train / 10% test)...");
    let (train_data, test_data) = LarsPipeline::load_and_split_data(0.90);

    println!("🏃 I'm training a LARS regression model...");
    let trained_model = LarsPipeline::train_model(&train_data)?;

    let model_path = Path::new("models/lars_diabetes_model.json");
    println!("💾 Saving model to '{:?}'...", model_path);
    LarsPipeline::save_model(&trained_model, model_path)?;

    println!("🔄 Loading model from disk...");
    let loaded_model = LarsPipeline::load_model(model_path)?;

    println!("📊 Running predictions and evaluations...");
    let test_samples = test_data.nsamples();
    let (predictions, RegressionMetrics { r2, mse }) =
        LarsPipeline::evaluate(&loaded_model, &test_data);

    let img_path = "img/lars_regression_plot.png";
    println!("🖼️ Generating prediction graph to '{:?}'...", img_path);
    LarsPipeline::render_regression_plot(test_data.targets(), &predictions, Path::new(img_path))?;

    // Load the generated image from disk into memory at runtime
    let img_bytes = std::fs::read(img_path).ok();

    let state = AppState {
        intercept: loaded_model.intercept,
        test_samples,
        r2,
        mse,
        img_bytes,
    };

    println!("🖥️ Opening UI...");
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([650.0, 600.0])
            .with_title("LARS Regression Dashboard"),
        ..Default::default()
    };

    eframe::run_native(
        "LARS Regression Results",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(LarsApp::new(state)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("Application UI error: {:?}", e))
}

struct LarsApp {
    state: AppState,
}

impl LarsApp {
    fn new(state: AppState) -> Self {
        Self { state }
    }
}

impl eframe::App for LarsApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("📊 LARS Regression Model Results");
        ui.separator();

        // Tabulka metrik
        egui::Grid::new("metrics_grid")
            .num_columns(2)
            .spacing([40.0, 8.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Intercept:");
                ui.label(format!("{:.4}", self.state.intercept));
                ui.end_row();

                ui.label("Number of test samples:");
                ui.label(format!("{}", self.state.test_samples));
                ui.end_row();

                ui.label("Coefficient of determination (R²):");
                ui.label(format!("{:.4}", self.state.r2));
                ui.end_row();

                ui.label("Mean Squared Error (MSE):");
                ui.label(format!("{:.2}", self.state.mse));
                ui.end_row();
            });

        ui.add_space(15.0);
        ui.heading("🖼️ Predictions vs. Reality Chart");
        ui.separator();

        // Render the image from the bytes in memory
        if let Some(bytes) = &self.state.img_bytes {
            ui.add(
                egui::Image::from_bytes("bytes://lars_regression_plot.png", bytes.clone())
                    .max_width(600.0)
                    .corner_radius(5.0),
            );
        } else {
            ui.label("⚠️ Failed to load chart image.");
        }
    }
}
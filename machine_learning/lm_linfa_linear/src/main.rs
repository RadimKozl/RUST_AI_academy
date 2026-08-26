use anyhow::Result;
use eframe::egui;
use linfa::prelude::*;
use lm_linfa_linear::RegressionPipeline;
use std::path::Path;

struct LinearRegressionApp {
    texture: Option<egui::TextureHandle>,
    info_text: String,
}

impl LinearRegressionApp {
    pub fn new(cc: &eframe::CreationContext<'_>, info_text: String, img_path: &Path) -> Self {
        let texture = match std::fs::read(img_path) {
            Ok(bytes) => match image::load_from_memory(&bytes) {
                Ok(img) => {
                    let size = [img.width() as usize, img.height() as usize];
                    let color_image =
                        egui::ColorImage::from_rgba_unmultiplied(size, &img.to_rgba8());
                    Some(cc.egui_ctx.load_texture(
                        "regression_plot",
                        color_image,
                        Default::default(),
                    ))
                }
                Err(_) => None,
            },
            Err(_) => None,
        };

        Self { texture, info_text }
    }
}

impl eframe::App for LinearRegressionApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("📈 Linear Regression — Diabetes Dataset");
        ui.separator();
        ui.label(&self.info_text);
        ui.separator();

        if let Some(texture) = &self.texture {
            ui.image((texture.id(), texture.size_vec2()));
        } else {
            ui.colored_label(egui::Color32::RED, "Error loading image in UI.");
        }
    }
}

fn main() -> Result<()> {
    println!("📖 Loading and splitting the Diabetes dataset (80% train / 20% test)...");
    let (train, test) = RegressionPipeline::load_and_split_data(0.8);

    println!("🏃 I'm training a Linear Regression model...");
    let model = RegressionPipeline::train_model(&train)?;

    let model_path = Path::new("models/linear_model.json");
    println!("💾 Saving trained model to '{}'...", model_path.display());
    RegressionPipeline::save_model(&model, model_path)?;

    println!("🔄 Loading model from disk...");
    let loaded_model = RegressionPipeline::load_model(model_path)?;

    println!("📊 Running predictions and evaluations...");
    let (predictions, metrics) = RegressionPipeline::evaluate(&loaded_model, &test);

    let img_path = Path::new("img/diabetes_regression.png");
    println!("🖼️ Plotting the graph into '{}'...", img_path.display());
    RegressionPipeline::render_scatter_plot(test.targets(), &predictions, img_path)?;

    let mut info = format!("Model intercept: {:.4}\n", loaded_model.intercept);
    info.push_str(&format!("Test samples: {}\n", test.nsamples()));
    info.push_str(&format!("Mean Absolute Error (MAE): {:.4}\n", metrics.mae));
    info.push_str(&format!("R² Score: {:.4}\n", metrics.r2_score));

    println!("\n{}", info);

    println!("💻 Opening UI Window...");
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([640.0, 580.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Linear Regression Visualizer",
        options,
        Box::new(move |cc| Ok(Box::new(LinearRegressionApp::new(cc, info, img_path)))),
    )
    .map_err(|e| anyhow::anyhow!("Eframe error: {:?}", e))?;

    Ok(())
}
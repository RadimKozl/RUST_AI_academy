use anyhow::Result;
use eframe::egui;
use linfa::dataset::Records;
use lm_linfa_logistic::LogisticPipeline;
use std::path::Path;

struct LogisticRegressionApp {
    texture: Option<egui::TextureHandle>,
    info_text: String,
}

impl LogisticRegressionApp {
    pub fn new(cc: &eframe::CreationContext<'_>, info_text: String, img_path: &Path) -> Self {
        let texture = match std::fs::read(img_path) {
            Ok(bytes) => match image::load_from_memory(&bytes) {
                Ok(img) => {
                    let size = [img.width() as usize, img.height() as usize];
                    let color_image =
                        egui::ColorImage::from_rgba_unmultiplied(size, &img.to_rgba8());
                    Some(cc.egui_ctx.load_texture(
                        "logistic_plot",
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

impl eframe::App for LogisticRegressionApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("🍷 Logistic Regression — Wine Quality Dataset");
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
    println!("📖 Loading and splitting the Wine Quality dataset (90% train / 10% test)...");
    let (train, test) = LogisticPipeline::load_and_split_data(0.9);

    println!("🏃 I'm training a Logistic Regression model...");
    let model = LogisticPipeline::train_model(&train)?;

    let model_path = Path::new("models/logistic_model.json");
    println!("💾 Saving trained model to '{}'...", model_path.display());
    LogisticPipeline::save_model(&model, model_path)?;

    println!("🔄 Loading model from disk...");
    let loaded_model = LogisticPipeline::load_model(model_path)?;

    println!("📊 Running predictions and evaluations...");
    let (_predictions, metrics) = LogisticPipeline::evaluate(&loaded_model, &test);

    let img_path = Path::new("img/wine_probabilities.png");
    println!("🖼️ Plotting the graph into '{}'...", img_path.display());
    let probabilities = loaded_model.predict_probabilities(test.records());
    LogisticPipeline::render_probability_plot(&probabilities, img_path)?;

    let mut info = format!("Intercept model: {:.4}\n", loaded_model.intercept);
    info.push_str(&format!("Test samples: {}\n", test.nsamples()));
    info.push_str(&format!("Accuracy: {:.2}%\n", metrics.accuracy * 100.0));
    info.push_str(&format!("Matthews Correlation Coefficient (MCC): {:.4}\n", metrics.mcc));

    println!("\n{}", info);

    println!("💻 Opening UI Window...");
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([640.0, 580.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Logistic Regression Visualizer",
        options,
        Box::new(move |cc| Ok(Box::new(LogisticRegressionApp::new(cc, info, img_path)))),
    )
    .map_err(|e| anyhow::anyhow!("Eframe error: {:?}", e))?;

    Ok(())
}

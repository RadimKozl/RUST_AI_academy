use eframe::egui;
use lm_smartcore_logistic_regression::{run_logistic_regression, LogisticRegressionResult};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([480.0, 360.0]),
        ..Default::default()
    };

    eframe::run_native(
        "SmartCore - Linear Models GUI",
        options,
        Box::new(|_cc| Ok(Box::new(GuiApp::default()))),
    )
}

struct GuiApp {
    result: Option<LogisticRegressionResult>,
    status: String,
}

impl Default for GuiApp {
    fn default() -> Self {
        Self {
            result: None,
            status: "Click to train Logistic Regression.".to_string(),
        }
    }
}

impl eframe::App for GuiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("📈 SmartCore: Logistic Regression");
        ui.add_space(10.0);

        if ui.button("🚀 Run Logistic Regression").clicked() {
            match run_logistic_regression() {
                Ok(res) => {
                    self.status = "Model successfully trained and saved to 'models/'!".to_string();
                    self.result = Some(res);
                }
                Err(e) => self.status = format!("Error: {}", e),
            }
        }

        ui.add_space(10.0);
        ui.label(format!("State: {}", self.status));
        ui.separator();

        if let Some(res) = &self.result {
            ui.add_space(10.0);
            ui.heading("📊 Results");
            ui.label(format!("Algorithm: {}", res.algorithm_name));
            ui.label(format!("Test Accuracy: {:.2}%", res.accuracy));

            ui.add_space(10.0);
            ui.label("Sample test predictions:");
            ui.code(format!("{:?}", res.test_predictions));
        }
    }
}
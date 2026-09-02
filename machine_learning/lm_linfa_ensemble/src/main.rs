use eframe::egui;
use lm_linfa_ensemble::{run_adaboost, run_random_forest, EnsembleResult};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([500.0, 400.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Linfa Ensemble ML Classifier",
        options,
        Box::new(|_cc| Ok(Box::new(GuiApp::default()))),
    )
}

struct GuiApp {
    result: Option<EnsembleResult>,
    status_msg: String,
}

impl Default for GuiApp {
    fn default() -> Self {
        Self {
            result: None,
            status_msg: "Select an algorithm to train.".to_string(),
        }
    }
}

impl eframe::App for GuiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("🤖 Linfa ML Classifier GUI");
        ui.add_space(10.0);

        ui.horizontal(|ui| {
            if ui.button("AdaBoost (Stumps)").clicked() {
                match run_adaboost(50, 1.0, 1) {
                    Ok(res) => {
                        self.status_msg = "AdaBoost completed!".to_string();
                        self.result = Some(res);
                    }
                    Err(e) => self.status_msg = format!("Error: {}", e),
                }
            }

            if ui.button("AdaBoost (Depth=2)").clicked() {
                match run_adaboost(50, 1.0, 2) {
                    Ok(res) => {
                        self.status_msg = "AdaBoost completed!".to_string();
                        self.result = Some(res);
                    }
                    Err(e) => self.status_msg = format!("Error: {}", e),
                }
            }

            if ui.button("Random Forest").clicked() {
                match run_random_forest(100, 0.7, 0.5) {
                    Ok(res) => {
                        self.status_msg = "Random Forest completed!".to_string();
                        self.result = Some(res);
                    }
                    Err(e) => self.status_msg = format!("Error: {}", e),
                }
            }
        });

        ui.add_space(15.0);
        ui.label(format!("State: {}", self.status_msg));
        ui.separator();

        if let Some(res) = &self.result {
            ui.add_space(10.0);
            ui.heading("📊 Model Results");
            ui.label(format!("Algorithm: {}", res.algorithm_name));
            ui.label(format!("Accuracy: {:.2}%", res.accuracy));

            ui.add_space(10.0);
            ui.label("Prediction (first elements):");
            let first_predictions: Vec<_> = res.predictions.iter().take(15).collect();
            ui.code(format!("{:?}", first_predictions));
        }
    }
}
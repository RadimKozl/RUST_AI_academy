use anyhow::Result;
use eframe::egui;
use lm_linfa_svm::{SvmEvaluation, SvmPipeline};

fn main() -> Result<()> {
    println!("=== 🖥️\t Running Linfa SVM Pipeline with GUI ===");

    // 1. Calculate the models
    let (train_bin, valid_bin) = SvmPipeline::load_binary_wine(42);
    let (bin_eval, bin_model) = SvmPipeline::train_binary_svm(&train_bin, &valid_bin, 0.6, 1.0, 1.0)?;

    let (train_multi, valid_multi) = SvmPipeline::load_multiclass_wine(42);
    let (multi_eval, _multi_model) = SvmPipeline::train_multiclass_svm(&train_multi, &valid_multi, 0.6)?;

    let (mse, summary, svr_model) = SvmPipeline::train_svr_regression(1000.0, 0.1, 0.1)?;

    // 2. Save the models
    SvmPipeline::save_model(&bin_model, "binary_svm_wine.json")?;
    SvmPipeline::save_model(&multi_eval, "multiclass_svm_wine.json")?;
    SvmPipeline::save_model(&svr_model, "svr_regression.json")?;

    // 3. Launch the GUI application
    let app = SvmGuiApp {
        bin_eval,
        multi_eval,
        svr_mse: mse,
        svr_summary: summary,
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([700.0, 500.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Linfa SVM Dashboard 📊",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI Error: {}", e))
}

struct SvmGuiApp {
    bin_eval: SvmEvaluation,
    multi_eval: SvmEvaluation,
    svr_mse: f64,
    svr_summary: String,
}

impl eframe::App for SvmGuiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("📊 Linfa SVM Results Dashboard");
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Binary classification
            ui.collapsing("🔍 Binary SVM Classification", |ui| {
                ui.label(format!("Type: {}", self.bin_eval.model_type));
                ui.label(format!("Accuracy: {:.2}%", self.bin_eval.accuracy_or_mse));
                if let Some(mcc) = self.bin_eval.mcc {
                    ui.label(format!("MCC: {:.4}", mcc));
                }
                ui.add_space(5.0);
                ui.label("Confusion Matrix:");
                ui.monospace(&self.bin_eval.confusion_matrix);
            });

            ui.add_space(10.0);

            // Multi-class classification
            ui.collapsing("🎯 Multi-Class SVM Classification", |ui| {
                ui.label(format!("Type: {}", self.multi_eval.model_type));
                ui.label(format!("Accuracy: {:.2}%", self.multi_eval.accuracy_or_mse));
                if let Some(mcc) = self.multi_eval.mcc {
                    ui.label(format!("MCC: {:.4}", mcc));
                }
                ui.add_space(5.0);
                ui.label("Confusion Matrix:");
                ui.monospace(&self.multi_eval.confusion_matrix);
            });

            ui.add_space(10.0);

            // Regression
            ui.collapsing("📈 SVR Regression", |ui| {
                ui.label(format!("Mean Squared Error (MSE): {:.6}", self.svr_mse));
                ui.add_space(5.0);
                ui.label("Model Details:");
                ui.monospace(&self.svr_summary);
            });

            ui.add_space(15.0);
            ui.separator();
            ui.label("💾 Models and evaluations saved in /models directory.");
        });
    }
}
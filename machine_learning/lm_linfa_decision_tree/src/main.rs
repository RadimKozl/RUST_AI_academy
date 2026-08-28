use anyhow::Result;
use eframe::egui;
use egui::{Color32, RichText};
use linfa_trees::SplitQuality;
use lm_linfa_decision_tree::{
    DecisionTreePipeline, ModelEvaluation, SavedTreeFeatures,
};
use std::path::Path;

struct AppState {
    gini_eval: ModelEvaluation,
    entropy_eval: ModelEvaluation,
    tikz_path: String,
    json_path: String,
}

fn main() -> Result<()> {
    println!("📖 Loading and splitting Iris dataset...");
    let (train, test) = DecisionTreePipeline::load_and_split_dataset(42);

    println!("🌳 I'm training a model with the Gini criterion...");
    let (gini_eval, gini_model) = DecisionTreePipeline::train_and_eval(
        &train,
        &test,
        SplitQuality::Gini,
        "Gini Impurity",
        1.0,
        1.0,
    )?;

    println!("🌳 I'm training a model with the Entropy criterion...");
    let (entropy_eval, _) = DecisionTreePipeline::train_and_eval(
        &train,
        &test,
        SplitQuality::Entropy,
        "Entropy",
        10.0,
        10.0,
    )?;

    // Export the Gini tree to TikZ / LaTeX
    let tikz_path = Path::new("export/decision_tree_example.tex");
    println!("📄 Exporting Gini tree to TikZ: '{:?}'...", tikz_path);
    DecisionTreePipeline::export_tikz(&gini_model, tikz_path)?;

    // Save information about trained features to JSON
    let json_path = Path::new("models/tree_features.json");
    let saved_info = SavedTreeFeatures {
        gini_features: gini_eval.features_used.clone(),
        entropy_features: entropy_eval.features_used.clone(),
    };
    println!("💾 Storing symptom information in '{:?}'...", json_path);
    DecisionTreePipeline::save_model_info(&saved_info, json_path)?;

    let state = AppState {
        gini_eval,
        entropy_eval,
        tikz_path: tikz_path.to_string_lossy().into_owned(),
        json_path: json_path.to_string_lossy().into_owned(),
    };

    println!("🖥️ Opening Dashboard UI...");
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([620.0, 580.0])
            .with_title("Decision Tree Iris Dashboard"),
        ..Default::default()
    };

    eframe::run_native(
        "Decision Tree Iris Classifier",
        options,
        Box::new(|_cc| Ok(Box::new(DecisionTreeApp::new(state)))),
    )
    .map_err(|e| anyhow::anyhow!("Application UI error: {:?}", e))
}

struct DecisionTreeApp {
    state: AppState,
}

impl DecisionTreeApp {
    fn new(state: AppState) -> Self {
        Self { state }
    }

    fn render_eval_card(ui: &mut egui::Ui, eval: &ModelEvaluation) {
        ui.group(|ui| {
            ui.heading(format!("Criteria: {}", eval.criterion));
            ui.horizontal(|ui| {
                ui.label("Accuracy:");
                ui.label(
                    RichText::new(format!("{:.2}%", eval.accuracy))
                        .strong()
                        .color(Color32::WHITE),
                );
            });
            ui.label(format!("Features used: {:?}", eval.features_used));
            ui.add_space(5.0);
            ui.label("Confusion Matrix:");
            ui.code(&eval.confusion_matrix_str);
        });
    }
}

impl eframe::App for DecisionTreeApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("🌿 Decision Trees on the Iris Dataset");
        ui.separator();

        ui.label(format!("📄 LaTeX TikZ saved: {}", self.state.tikz_path));
        ui.label(format!("💾 Symptoms info saved: {}", self.state.json_path));
        ui.add_space(10.0);

        Self::render_eval_card(ui, &self.state.gini_eval);
        ui.add_space(10.0);
        Self::render_eval_card(ui, &self.state.entropy_eval);
    }
}
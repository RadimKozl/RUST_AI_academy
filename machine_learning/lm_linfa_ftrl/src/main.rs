use anyhow::Result;
use eframe::egui::{self, Color32, Pos2, Rect, RichText, Stroke, StrokeKind, Vec2};
use lm_linfa_ftrl::{FtrlEvaluation, FtrlPipeline};

fn main() -> Result<()> {
    let eval = FtrlPipeline::train_ftrl(0.005, 1.0, 0.005, 1.0, 42)?;
    FtrlPipeline::save_to_models(&eval, "ftrl_model_results.json")?;

    let app = FtrlGuiApp { eval };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([920.0, 680.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Linfa FTRL Online Learning Dashboard",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI Error: {}", e))
}

struct FtrlGuiApp {
    eval: FtrlEvaluation,
}

impl eframe::App for FtrlGuiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading(
            RichText::new("⚡ FTRL Online Classification (Wine Quality > 6)")
                .color(Color32::WHITE)
                .size(20.0),
        );
        ui.separator();

        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Train set: {} samples", self.eval.train_samples)).color(Color32::LIGHT_GRAY));
            ui.label("|");
            ui.label(RichText::new(format!("Valid set: {} samples", self.eval.valid_samples)).color(Color32::LIGHT_GRAY));
            ui.label("|");
            ui.label(
                RichText::new(format!("Valid Log Loss: {:.4}", self.eval.valid_log_loss))
                    .color(Color32::GREEN)
                    .strong(),
            );
        });

        ui.add_space(8.0);

        ui.collapsing(RichText::new("⚙ Hyperparameters").color(Color32::LIGHT_BLUE), |ui| {
            ui.monospace(format!("Alpha: {}", self.eval.alpha));
            ui.monospace(format!("Beta: {}", self.eval.beta));
            ui.monospace(format!("L1 Ratio: {}", self.eval.l1_ratio));
            ui.monospace(format!("L2 Ratio: {}", self.eval.l2_ratio));
        });

        ui.add_space(10.0);
        ui.label(RichText::new("📈 First 40 predicted probabilities (Valid set)").color(Color32::KHAKI));
        ui.add_space(4.0);

        let (response, painter) = ui.allocate_painter(Vec2::new(ui.available_width(), 260.0), egui::Sense::hover());
        let rect = response.rect;

        painter.rect_filled(rect, 4.0, Color32::from_rgb(18, 22, 28));
        painter.rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::GRAY), StrokeKind::Outside);

        let n_items = 40.min(self.eval.target_probabilities.len());
        let margin_x = 35.0_f32;
        let margin_y = 25.0_f32;
        let plot_width = rect.width() - 2.0 * margin_x;
        let plot_height = rect.height() - 2.0 * margin_y;

        let threshold_y = rect.max.y - margin_y - (0.5 * plot_height);
        painter.line_segment(
            [Pos2::new(rect.min.x + margin_x, threshold_y), Pos2::new(rect.max.x - margin_x, threshold_y)],
            Stroke::new(1.0, Color32::RED),
        );

        let step_x = plot_width / n_items as f32;
        let bar_w = (step_x * 0.6).max(3.0);

        for i in 0..n_items {
            let prob = self.eval.target_probabilities[i] as f32;
            let is_true = self.eval.true_targets[i];

            let x = rect.min.x + margin_x + (i as f32 + 0.2) * step_x;
            let bar_h = prob * plot_height;
            let y_top = rect.max.y - margin_y - bar_h;
            let y_bot = rect.max.y - margin_y;

            let color = if is_true { Color32::LIGHT_GREEN } else { Color32::LIGHT_RED };

            painter.rect_filled(
                Rect::from_min_max(Pos2::new(x, y_top), Pos2::new(x + bar_w, y_bot)),
                1.0,
                color,
            );
        }

        ui.add_space(10.0);

        ui.horizontal(|ui| {
            ui.colored_label(Color32::LIGHT_GREEN, "■ True Target = True (>6)");
            ui.add_space(15.0);
            ui.colored_label(Color32::LIGHT_RED, "■ True Target = False (<=6)");
            ui.add_space(15.0);
            ui.colored_label(Color32::RED, "— Threshold value 0.5");
        });

        ui.add_space(15.0);

        ui.collapsing(RichText::new("🔍 Detailed forecast output").color(Color32::WHITE), |ui| {
            let mut table = String::from("  ID | Prob (%) | Prediction | Reality | State\n");
            table.push_str("---------------------------------------------------\n");

            for i in 0..n_items {
                let prob = self.eval.target_probabilities[i];
                let true_val = self.eval.true_targets[i];
                let pred_val = prob >= 0.5;
                let status = if pred_val == true_val { "OK" } else { "ERR" };

                table.push_str(&format!(
                    "{:4} | {:7.2}% | {:8} | {:10} | {}\n",
                    i, prob * 100.0, pred_val, true_val, status
                ));
            }

            ui.monospace(RichText::new(table).color(Color32::GREEN));
        });
    }
}
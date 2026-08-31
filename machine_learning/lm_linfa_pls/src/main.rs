use anyhow::Result;
use eframe::egui::{self, Color32, Pos2, Rect, RichText, Stroke, StrokeKind, Vec2};
use lm_linfa_pls::{PlsEvaluation, PlsPipeline};

fn main() -> Result<()> {
    let (_model, eval) = PlsPipeline::train_pls(1000, 10, 3, 42, 3)?;
    PlsPipeline::save_to_models(&eval, "pls_evaluation_results.json")?;

    let app = PlsGuiApp { eval };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Linfa PLS Regression Dashboard",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI Error: {}", e))
}

struct PlsGuiApp {
    eval: PlsEvaluation,
}

impl eframe::App for PlsGuiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading(
            RichText::new("📊 PLS Regression Coefficient Inspection")
                .color(Color32::WHITE)
                .size(20.0),
        );
        ui.separator();

        ui.label(
            RichText::new(format!(
                "Dataset Config: N = {} samples | P = {} features | Q = {} targets",
                self.eval.n_samples, self.eval.n_features, self.eval.n_targets
            ))
            .color(Color32::LIGHT_GRAY),
        );
        ui.add_space(10.0);

        ui.label(
            RichText::new("📈 Graphical dependence of coefficients (Target q0): True vs Estimated")
                .color(Color32::GREEN),
        );
        ui.add_space(5.0);

        // Draw your own chart using Painter
        let (response, painter) = ui.allocate_painter(Vec2::new(ui.available_width(), 240.0), egui::Sense::hover());
        let rect = response.rect;

        // Background and frame of the chart
        painter.rect_filled(rect, 4.0, Color32::from_rgb(20, 24, 30));
        painter.rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::GRAY), StrokeKind::Outside);

        // Legend
        painter.rect_filled(
            Rect::from_min_size(Pos2::new(rect.min.x + 15.0, rect.min.y + 10.0), Vec2::new(12.0, 12.0)),
            2.0,
            Color32::LIGHT_BLUE,
        );
        painter.text(
            Pos2::new(rect.min.x + 32.0, rect.min.y + 9.0),
            egui::Align2::LEFT_TOP,
            "True B",
            egui::FontId::proportional(13.0),
            Color32::WHITE,
        );

        painter.rect_filled(
            Rect::from_min_size(Pos2::new(rect.min.x + 95.0, rect.min.y + 10.0), Vec2::new(12.0, 12.0)),
            2.0,
            Color32::KHAKI,
        );
        painter.text(
            Pos2::new(rect.min.x + 112.0, rect.min.y + 9.0),
            egui::Align2::LEFT_TOP,
            "Estimated B",
            egui::FontId::proportional(13.0),
            Color32::WHITE,
        );

        // Parameters for drawing columns in f32
        let n_features = self.eval.n_features as f32;
        let margin_x = 40.0_f32;
        let margin_y = 30.0_f32;
        let plot_width = rect.width() - 2.0 * margin_x;
        let plot_height = rect.height() - 2.0 * margin_y;
        let zero_y = rect.max.y - margin_y - (plot_height * 0.1);
        let scale_y = plot_height * 0.35;

        // Zero axis
        painter.line_segment(
            [Pos2::new(rect.min.x + margin_x, zero_y), Pos2::new(rect.max.x - margin_x, zero_y)],
            Stroke::new(1.0, Color32::DARK_GRAY),
        );

        let group_width = plot_width / n_features;
        let bar_width = group_width * 0.35;

        for i in 0..self.eval.n_features {
            let center_x = rect.min.x + margin_x + (i as f32 + 0.5) * group_width;

            let true_val = self.eval.true_b[i][0] as f32;
            let est_val = self.eval.estimated_b[i][0] as f32;

            // True B Column (Blue)
            let true_h = true_val * scale_y;
            let true_rect = Rect::from_min_max(
                Pos2::new(center_x - bar_width, zero_y - true_h),
                Pos2::new(center_x, zero_y),
            );
            painter.rect_filled(true_rect, 1.0, Color32::LIGHT_BLUE);

            // Estimated B Column (Yellow)
            let est_h = est_val * scale_y;
            let est_rect = Rect::from_min_max(
                Pos2::new(center_x + 2.0, zero_y - est_h),
                Pos2::new(center_x + bar_width + 2.0, zero_y),
            );
            painter.rect_filled(est_rect, 1.0, Color32::KHAKI);

            // Flag index label (R0..R9)
            painter.text(
                Pos2::new(center_x, zero_y + 8.0),
                egui::Align2::CENTER_TOP,
                format!("R{}", i),
                egui::FontId::monospace(11.0),
                Color32::LIGHT_GRAY,
            );
        }

        ui.add_space(15.0);

        // Text table with clear green numbers
        ui.collapsing(
            RichText::new("🔍 Text Matrix B (All Targets q0..q2)").color(Color32::WHITE),
            |ui| {
                let mut table = String::from("Row |       True B (q0, q1, q2)       |    Estimated B (q0, q1, q2)\n");
                table.push_str("----------------------------------------------------------------------\n");

                for i in 0..self.eval.n_features {
                    let t = &self.eval.true_b[i];
                    let e = &self.eval.estimated_b[i];
                    table.push_str(&format!(
                        " {:2} | [{:4.1}, {:4.1}, {:4.1}]               | [{:5.2}, {:5.2}, {:5.2}]\n",
                        i, t[0], t[1], t[2], e[0], e[1], e[2]
                    ));
                }

                ui.monospace(RichText::new(table).color(Color32::LIGHT_GREEN));
            },
        );
    }
}
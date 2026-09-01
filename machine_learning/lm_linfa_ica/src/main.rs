use anyhow::Result;
use eframe::egui::{self, Color32, Pos2, RichText, Stroke, StrokeKind, Vec2};
use lm_linfa_ica::{IcaEvaluation, IcaPipeline};

fn main() -> Result<()> {
    let eval = IcaPipeline::run_ica()?;
    IcaPipeline::save_to_models(&eval, "ica_results.json")?;

    let app = IcaGuiApp { eval };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([960.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Linfa FastICA Signal Separation Dashboard",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI Error: {}", e))
}

struct IcaGuiApp {
    eval: IcaEvaluation,
}

impl IcaGuiApp {
    fn draw_signal_plot(
        ui: &mut egui::Ui,
        title: &str,
        data: &[(f64, f64)],
        color1: Color32,
        color2: Color32,
    ) {
        // Changed from .bold() to .strong()
        ui.label(RichText::new(title).color(Color32::KHAKI).strong());
        ui.add_space(2.0);

        let (response, painter) =
            ui.allocate_painter(Vec2::new(ui.available_width(), 160.0), egui::Sense::hover());
        let rect = response.rect;

        painter.rect_filled(rect, 4.0, Color32::from_rgb(18, 22, 28));
        painter.rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::GRAY), StrokeKind::Outside);

        if data.is_empty() {
            return;
        }

        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;
        for &(s1, s2) in data {
            if s1 < min_val { min_val = s1; }
            if s1 > max_val { max_val = s1; }
            if s2 < min_val { min_val = s2; }
            if s2 > max_val { max_val = s2; }
        }

        let margin = 10.0_f32;
        let w = rect.width() - 2.0 * margin;
        let h = rect.height() - 2.0 * margin;
        let len = data.len() as f32;

        for i in 0..(data.len() - 1) {
            let x1 = rect.min.x + margin + (i as f32 / len) * w;
            let x2 = rect.min.x + margin + ((i + 1) as f32 / len) * w;

            let norm_y1_s1 = ((data[i].0 - min_val) / (max_val - min_val + 1e-5)) as f32;
            let norm_y2_s1 = ((data[i + 1].0 - min_val) / (max_val - min_val + 1e-5)) as f32;

            let norm_y1_s2 = ((data[i].1 - min_val) / (max_val - min_val + 1e-5)) as f32;
            let norm_y2_s2 = ((data[i + 1].1 - min_val) / (max_val - min_val + 1e-5)) as f32;

            let py1_s1 = rect.max.y - margin - norm_y1_s1 * h;
            let py2_s1 = rect.max.y - margin - norm_y2_s1 * h;

            let py1_s2 = rect.max.y - margin - norm_y1_s2 * h;
            let py2_s2 = rect.max.y - margin - norm_y2_s2 * h;

            painter.line_segment([Pos2::new(x1, py1_s1), Pos2::new(x2, py2_s1)], Stroke::new(1.2, color1));
            painter.line_segment([Pos2::new(x1, py1_s2), Pos2::new(x2, py2_s2)], Stroke::new(1.2, color2));
        }

        ui.add_space(6.0);
    }
}

impl eframe::App for IcaGuiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading(
                RichText::new("📻 FastICA Signal Separation Dashboard")
                    .color(Color32::WHITE)
                    .size(20.0),
            );
            ui.separator();

            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("Samples: {}", self.eval.n_samples))
                        .color(Color32::LIGHT_GRAY),
                );
                ui.label("|");
                ui.label(RichText::new("Function: Logcosh(1.0)").color(Color32::GOLD));
            });

            ui.add_space(10.0);

            Self::draw_signal_plot(
                ui,
                "1. Original Signals (Sine + Sawtooth + Noise)",
                &self.eval.original_signals,
                Color32::from_rgb(0, 210, 255),
                Color32::from_rgb(255, 100, 100),
            );

            Self::draw_signal_plot(
                ui,
                "2. Mixed Signals (Observed Data)",
                &self.eval.mixed_signals,
                Color32::from_rgb(255, 200, 50),
                Color32::from_rgb(200, 100, 255),
            );

            Self::draw_signal_plot(
                ui,
                "3. Reconstructed Signals (FastICA Unmixed)",
                &self.eval.unmixed_signals,
                Color32::from_rgb(100, 255, 100),
                Color32::from_rgb(255, 128, 0),
            );

            ui.add_space(10.0);

            ui.collapsing(RichText::new("🔍 Raw Signal Values (Scrollable Table)").color(Color32::WHITE), |ui| {
                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .show(ui, |ui| {
                        let mut table = String::from("   Idx | Orig S1 | Orig S2 | Mix S1  | Mix S2  | ICA S1  | ICA S2\n");
                        table.push_str("----------------------------------------------------------------------\n");

                        for i in 0..self.eval.n_samples {
                            let (o1, o2) = self.eval.original_signals[i];
                            let (m1, m2) = self.eval.mixed_signals[i];
                            let (u1, u2) = self.eval.unmixed_signals[i];
                            table.push_str(&format!(
                                "{:6} | {:7.3} | {:7.3} | {:7.3} | {:7.3} | {:7.3} | {:7.3}\n",
                                i, o1, o2, m1, m2, u1, u2
                            ));
                        }

                        ui.monospace(RichText::new(table).color(Color32::GREEN));
                    });
            });
        });
    }
}
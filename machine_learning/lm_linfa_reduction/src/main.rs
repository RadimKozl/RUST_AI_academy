use anyhow::Result;
use eframe::egui::{self, Color32, Pos2, RichText, Stroke, StrokeKind, Vec2};
use lm_linfa_reduction::{
    DiffusionMapEvaluation, PcaEvaluation, ProjectionEvaluation, ReductionPipeline,
};

fn main() -> Result<()> {
    let pca_eval = ReductionPipeline::run_pca()?;
    ReductionPipeline::save_to_models(&pca_eval, "pca_results.json")?;

    let diff_eval = ReductionPipeline::run_diffusion_map()?;
    ReductionPipeline::save_to_models(&diff_eval, "diffusion_map_results.json")?;

    let gaussian_eval = ReductionPipeline::run_gaussian_projection()?;
    ReductionPipeline::save_to_models(&gaussian_eval, "gaussian_projection_results.json")?;

    let sparse_eval = ReductionPipeline::run_sparse_projection()?;
    ReductionPipeline::save_to_models(&sparse_eval, "sparse_projection_results.json")?;

    let app = ReductionGuiApp {
        pca_eval,
        diff_eval,
        gaussian_eval,
        sparse_eval,
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1000.0, 850.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Linfa Dimensionality Reduction Dashboard",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI Error: {}", e))
}

struct ReductionGuiApp {
    pca_eval: PcaEvaluation,
    diff_eval: DiffusionMapEvaluation,
    gaussian_eval: ProjectionEvaluation,
    sparse_eval: ProjectionEvaluation,
}

impl ReductionGuiApp {
    fn draw_2d_scatter(ui: &mut egui::Ui, title: &str, points: &[(f64, f64)], color: Color32) {
        ui.label(RichText::new(title).color(Color32::KHAKI).strong());
        ui.add_space(2.0);

        let (response, painter) =
            ui.allocate_painter(Vec2::new(ui.available_width(), 200.0), egui::Sense::hover());
        let rect = response.rect;

        painter.rect_filled(rect, 4.0, Color32::from_rgb(18, 22, 28));
        painter.rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::GRAY), StrokeKind::Outside);

        if points.is_empty() {
            return;
        }

        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;

        for &(x, y) in points {
            if x < min_x { min_x = x; }
            if x > max_x { max_x = x; }
            if y < min_y { min_y = y; }
            if y > max_y { max_y = y; }
        }

        let margin = 20.0_f32;
        let w = rect.width() - 2.0 * margin;
        let h = rect.height() - 2.0 * margin;

        for &(x, y) in points {
            let norm_x = ((x - min_x) / (max_x - min_x + 1e-5)) as f32;
            let norm_y = ((y - min_y) / (max_y - min_y + 1e-5)) as f32;

            let px = rect.min.x + margin + norm_x * w;
            let py = rect.max.y - margin - norm_y * h;

            painter.circle_filled(Pos2::new(px, py), 3.0, color);
        }

        ui.add_space(6.0);
    }
}

impl eframe::App for ReductionGuiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading(
                RichText::new("📉 Linfa Dimensionality Reduction Dashboard")
                    .color(Color32::WHITE)
                    .size(20.0),
            );
            ui.separator();

            // PCA
            ui.collapsing(RichText::new("1. Principal Component Analysis (PCA)").color(Color32::LIGHT_BLUE).strong(), |ui| {
                Self::draw_2d_scatter(ui, "PCA Projections", &self.pca_eval.points_2d, Color32::LIGHT_BLUE);
            });

            ui.add_space(5.0);

            // Diffusion Maps
            ui.collapsing(RichText::new("2. Diffusion Maps").color(Color32::GOLD).strong(), |ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        Self::draw_2d_scatter(ui, "Original Convoluted Rings", &self.diff_eval.points_2d, Color32::LIGHT_RED);
                    });
                    ui.vertical(|ui| {
                        Self::draw_2d_scatter(ui, "Diffusion Map Space", &self.diff_eval.embedded_2d, Color32::LIGHT_GREEN);
                    });
                });
            });

            ui.add_space(5.0);

            // Gaussian Random Projection
            ui.collapsing(RichText::new("3. Gaussian Random Projection").color(Color32::LIGHT_GREEN).strong(), |ui| {
                Self::draw_2d_scatter(ui, "Gaussian Reduced Space", &self.gaussian_eval.points_2d, Color32::LIGHT_GREEN);
            });

            ui.add_space(5.0);

            // Sparse Random Projection
            ui.collapsing(RichText::new("4. Sparse Random Projection").color(Color32::ORANGE).strong(), |ui| {
                Self::draw_2d_scatter(ui, "Sparse Reduced Space", &self.sparse_eval.points_2d, Color32::KHAKI);
            });
        });
    }
}
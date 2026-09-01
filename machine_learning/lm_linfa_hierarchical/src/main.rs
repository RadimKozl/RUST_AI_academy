use anyhow::Result;
use eframe::egui::{self, Color32, Pos2, RichText, Stroke, StrokeKind, Vec2};
use lm_linfa_hierarchical::{HierarchicalEvaluation, HierarchicalPipeline};

fn main() -> Result<()> {
    let eval = HierarchicalPipeline::train_hierarchical(3)?;
    HierarchicalPipeline::save_to_models(&eval, "hierarchical_model_results.json")?;

    let app = HierarchicalGuiApp { eval };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([920.0, 720.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Linfa Hierarchical Clustering Dashboard",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI Error: {}", e))
}

struct HierarchicalGuiApp {
    eval: HierarchicalEvaluation,
}

impl eframe::App for HierarchicalGuiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading(
            RichText::new("🌳 Hierarchical Agglomerative Clustering (Wine Quality)")
                .color(Color32::WHITE)
                .size(20.0),
        );
        ui.separator();

        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Samples: {}", self.eval.total_samples)).color(Color32::LIGHT_GRAY));
            ui.label("|");
            ui.label(RichText::new(format!("Features: {}", self.eval.num_features)).color(Color32::LIGHT_GRAY));
            ui.label("|");
            ui.label(
                RichText::new(format!("Target number of clusters (K): {}", self.eval.num_clusters))
                    .color(Color32::GOLD)
                    .strong(),
            );
        });

        ui.add_space(10.0);
        ui.label(RichText::new("📊 2D Scatter Plot of Clusters (Property 1 vs Property 2)").color(Color32::KHAKI));
        ui.add_space(4.0);

        let (response, painter) = ui.allocate_painter(Vec2::new(ui.available_width(), 380.0), egui::Sense::hover());
        let rect = response.rect;

        painter.rect_filled(rect, 4.0, Color32::from_rgb(18, 22, 28));
        painter.rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::GRAY), StrokeKind::Outside);

        // Finding min and max values ​​for normalizing coordinates to canvas
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;

        for &(x, y) in &self.eval.features_2d {
            if x < min_x { min_x = x; }
            if x > max_x { max_x = x; }
            if y < min_y { min_y = y; }
            if y > max_y { max_y = y; }
        }

        let margin = 30.0_f32;
        let plot_w = rect.width() - 2.0 * margin;
        let plot_h = rect.height() - 2.0 * margin;

        // Color palette for clusters
        let colors = [
            Color32::from_rgb(0, 210, 255),   // Cyan
            Color32::from_rgb(255, 100, 100), // Red
            Color32::from_rgb(100, 255, 100), // Green
            Color32::from_rgb(255, 200, 50),  // Yellow
        ];

        // Plotting points
        for (i, &(x, y)) in self.eval.features_2d.iter().enumerate() {
            let norm_x = ((x - min_x) / (max_x - min_x + 1e-5)) as f32;
            let norm_y = ((y - min_y) / (max_y - min_y + 1e-5)) as f32;

            let px = rect.min.x + margin + norm_x * plot_w;
            let py = rect.max.y - margin - norm_y * plot_h;

            let cluster_id = self.eval.cluster_assignments[i];
            let color = colors[cluster_id % colors.len()];

            painter.circle_filled(Pos2::new(px, py), 2.5, color);
        }

        ui.add_space(10.0);

        ui.horizontal(|ui| {
            ui.colored_label(colors[0], "■ Cluster 0");
            ui.add_space(15.0);
            ui.colored_label(colors[1], "■ Cluster 1");
            ui.add_space(15.0);
            ui.colored_label(colors[2], "■ Cluster 2");
        });

        ui.add_space(15.0);

        ui.collapsing(RichText::new("🔍 Sample assignment of the first 30 samples").color(Color32::WHITE), |ui| {
            let mut table = String::from("  ID | Property X | Property Y | Allocated Cluster\n");
            table.push_str("-----------------------------------------------------\n");

            let limit = 30.min(self.eval.total_samples);
            for i in 0..limit {
                let (x, y) = self.eval.features_2d[i];
                let cluster = self.eval.cluster_assignments[i];
                table.push_str(&format!("{:4} | {:11.3} | {:11.3} | Cluster {}\n", i, x, y, cluster));
            }

            ui.monospace(RichText::new(table).color(Color32::GREEN));
        });
    }
}
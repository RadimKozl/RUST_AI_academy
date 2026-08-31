use anyhow::Result;
use eframe::egui::{self, Color32, Pos2, RichText, Stroke, StrokeKind};
use lm_linfa_tsne::{TsnePipeline, TsneResult};

fn main() -> Result<()> {
    println!("=== 🖥️\t Running Linfa t-SNE Dashboard ===");

    let perplexity = 10.0;
    let approx_thresh = 0.1;

    // 1. Calculate the t-SNE embedding
    let result = TsnePipeline::run_iris_tsne(perplexity, approx_thresh)?;

    // 2. Save the resulting model/data
    TsnePipeline::save_result(&result, "models/iris_tsne_result.json")?;

    // 3. Launch the GUI
    let app = TsneGuiApp { result };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([850.0, 650.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Linfa t-SNE Iris Dashboard 📊",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI Error: {}", e))
}

struct TsneGuiApp {
    result: TsneResult,
}

impl eframe::App for TsneGuiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading(RichText::new("📊 t-SNE Iris Dataset Manifold Embedding").color(Color32::WHITE));
        ui.separator();

        ui.label(format!(
            "Perplexity: {} | Approx Threshold: {} | Počet vzorků: {}",
            self.result.perplexity,
            self.result.approx_threshold,
            self.result.points.len()
        ));
        ui.add_space(5.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Graphical Scatter Plot
            ui.collapsing(RichText::new("🎨 2D Scatter Plot Canvas").color(Color32::LIGHT_GRAY), |ui| {
                let (response, painter) = ui.allocate_painter(
                    egui::vec2(ui.available_width(), 350.0),
                    egui::Sense::hover(),
                );
                let rect = response.rect;

                // Render the canvas background and frame
                painter.rect_filled(rect, 4.0, Color32::from_rgb(20, 20, 20));
                painter.rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::GRAY), StrokeKind::Outside);

                // Finding min and max values ​​for scaling coordinates
                let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
                let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);

                for p in &self.result.points {
                    if p.x < min_x { min_x = p.x; }
                    if p.x > max_x { max_x = p.x; }
                    if p.y < min_y { min_y = p.y; }
                    if p.y > max_y { max_y = p.y; }
                }

                // Render individual points to the canvas
                for p in &self.result.points {
                    let norm_x = ((p.x - min_x) / (max_x - min_x + 1e-5)) as f32;
                    let norm_y = ((p.y - min_y) / (max_y - min_y + 1e-5)) as f32;

                    let screen_x = rect.min.x + 20.0 + norm_x * (rect.width() - 40.0);
                    let screen_y = rect.max.y - 20.0 - norm_y * (rect.height() - 40.0);

                    // Color by class (0: Red, 1: Green, 2: Blue)
                    let color = match p.label {
                        0 => Color32::from_rgb(255, 99, 71),   // Iris-Setosa
                        1 => Color32::from_rgb(60, 179, 113),  // Iris-Versicolor
                        _ => Color32::from_rgb(30, 144, 255),  // Iris-Virginica
                    };

                    painter.circle_filled(Pos2::new(screen_x, screen_y), 4.0, color);
                }
            });

            ui.add_space(10.0);

            // Print the resulting coordinates
            ui.collapsing(RichText::new("📋 Embedded Points Data").color(Color32::LIGHT_GRAY), |ui| {
                let mut data_str = String::from("  X          Y        Label\n---------------------------\n");
                for p in self.result.points.iter().take(30) {
                    data_str.push_str(&format!("{:10.4} {:10.4}    {}\n", p.x, p.y, p.label));
                }
                data_str.push_str("... (additional points stored in JSON)\n");

                ui.monospace(RichText::new(data_str).color(Color32::WHITE));
            });

            ui.add_space(15.0);
            ui.separator();
            ui.label("💾 Model execution and embedding results saved to 'iris_tsne_result.json'.");
        });
    }
}
use anyhow::Result;
use eframe::egui;
use lm_linfa_clustering::ClusteringPipeline;
use std::path::Path;

struct OpticsPlotApp {
    _image_bytes: Vec<u8>,
    texture: Option<egui::TextureHandle>,
    info_text: String,
}

impl OpticsPlotApp {
    pub fn new(cc: &eframe::CreationContext<'_>, image_bytes: Vec<u8>, info_text: String) -> Self {
        let texture = match image::load_from_memory(&image_bytes) {
            Ok(img) => {
                let size = [img.width() as usize, img.height() as usize];
                let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &img.to_rgba8());
                Some(cc.egui_ctx.load_texture("optics_plot", color_image, Default::default()))
            }
            Err(_) => None,
        };

        Self {
            _image_bytes: image_bytes,
            texture,
            info_text,
        }
    }
}

// Opraveno pro eframe v0.36+ (používá fn ui namísto fn update)
impl eframe::App for OpticsPlotApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("📊 OPTICS Reachability Plot & Clustering");
        ui.separator();
        ui.label(&self.info_text);
        ui.separator();

        if let Some(texture) = &self.texture {
            ui.image((texture.id(), texture.size_vec2()));
        } else {
            ui.colored_label(egui::Color32::RED, "Chyba při načítání obrázku v UI.");
        }
    }
}

fn main() -> Result<()> {
    let csv_path = Path::new("data/Mall_Customers.csv");
    if !csv_path.exists() {
        anyhow::bail!("File '{}' not found.", csv_path.display());
    }

    println!("📖 Loading Mall_Customers.csv...");
    let records = ClusteringPipeline::load_dataset(csv_path)?;

    println!("🏃 I'm doing K-Means (K=5)...");
    let kmeans_count = ClusteringPipeline::run_kmeans(&records, 5)?;

    println!("🏃 I'm doing a DBSCAN...");
    let dbscan_res = ClusteringPipeline::run_dbscan(&records, 5, 5.0)?;

    println!("🏃 I do OPTICS...");
    let reachability = ClusteringPipeline::run_optics(&records, 5, 10.0)?;

    let png_path = Path::new("img/optics_reachability.png");
    ClusteringPipeline::render_reachability_plot(&reachability, png_path)?;
    println!("🖼️ Chart saved to '{}'", png_path.display());

    let image_bytes = std::fs::read(png_path)?;

    let mut info = format!("Sample dataset: {}\n", records.shape()[0]);
    info.push_str(&format!("K-Means processed samples: {}\n", kmeans_count));
    info.push_str(&format!("DBSCAN noise points: {}, clusters: {}\n", dbscan_res.noise_count, dbscan_res.cluster_counts.len()));
    if let Some(score) = dbscan_res.silhouette_score {
        info.push_str(&format!("DBSCAN Silhouette score: {:.4}\n", score));
    }

    println!("\n💻 Opening UI Window...");
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([840.0, 520.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Clustering & Optics Plot Visualizer",
        options,
        Box::new(move |cc| Ok(Box::new(OpticsPlotApp::new(cc, image_bytes, info)))),
    ).map_err(|e| anyhow::anyhow!("Eframe error: {:?}", e))?;

    Ok(())
}
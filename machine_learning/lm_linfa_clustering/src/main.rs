use anyhow::Result;
use eframe::egui;
use lm_linfa_clustering::ClusteringPipeline;
use std::path::Path;

struct OpticsPlotApp {
    tex_kmeans: Option<egui::TextureHandle>,
    tex_dbscan: Option<egui::TextureHandle>,
    tex_optics: Option<egui::TextureHandle>,
    info_text: String,
}

impl OpticsPlotApp {
    fn load_tex(cc: &eframe::CreationContext<'_>, name: &str, path: &Path) -> Option<egui::TextureHandle> {
        let bytes = std::fs::read(path).ok()?;
        let img = image::load_from_memory(&bytes).ok()?;
        let size = [img.width() as usize, img.height() as usize];
        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &img.to_rgba8());
        Some(cc.egui_ctx.load_texture(name, color_image, Default::default()))
    }

    pub fn new(cc: &eframe::CreationContext<'_>, info_text: String) -> Self {
        Self {
            tex_kmeans: Self::load_tex(cc, "kmeans", Path::new("img/kmeans_clusters.png")),
            tex_dbscan: Self::load_tex(cc, "dbscan", Path::new("img/dbscan_clusters.png")),
            tex_optics: Self::load_tex(cc, "optics", Path::new("img/optics_reachability.png")),
            info_text,
        }
    }
}

impl eframe::App for OpticsPlotApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("📊 Srovnání Výsledků Shlukování (Mall Customers Dataset)");
            ui.separator();
            ui.label(&self.info_text);
            ui.separator();

            ui.columns(3, |cols| {
                cols[0].vertical(|ui| {
                    ui.label("🔴 K-Means (K=5)");
                    if let Some(tex) = &self.tex_kmeans {
                        ui.image((tex.id(), tex.size_vec2()));
                    }
                });

                cols[1].vertical(|ui| {
                    ui.label("🔵 DBSCAN (tol=5.0, min=5)");
                    if let Some(tex) = &self.tex_dbscan {
                        ui.image((tex.id(), tex.size_vec2()));
                    }
                });

                cols[2].vertical(|ui| {
                    ui.label("📈 OPTICS (reachability)");
                    if let Some(tex) = &self.tex_optics {
                        ui.image((tex.id(), tex.size_vec2()));
                    }
                });
            });
        });
    }
}

fn main() -> Result<()> {
    let csv_path = Path::new("data/Mall_Customers.csv");
    if !csv_path.exists() {
        anyhow::bail!("File '{}' not found.", csv_path.display());
    }

    let records = ClusteringPipeline::load_dataset(csv_path)?;

    // 1. K-Means calculation and saving of cluster graph
    let kmeans_res = ClusteringPipeline::run_kmeans(&records, 5)?;
    let kmeans_clusters: Vec<isize> = kmeans_res.assignments.iter().map(|&x| x as isize).collect();
    ClusteringPipeline::render_scatter_plot(&records, &kmeans_clusters, "K-Means Clusters", Path::new("img/kmeans_clusters.png"))?;

    // 2. DBSCAN compute and store cluster graph
    let dbscan_res = ClusteringPipeline::run_dbscan(&records, 5, 5.0)?;
    let dbscan_clusters: Vec<isize> = dbscan_res.assignments.iter().map(|&x| x.map(|c| c as isize).unwrap_or(-1)).collect();
    ClusteringPipeline::render_scatter_plot(&records, &dbscan_clusters, "DBSCAN Clusters", Path::new("img/dbscan_clusters.png"))?;

    // 3. OPTICS calculate and save reachability graph
    let reachability = ClusteringPipeline::run_optics(&records, 5, 10.0)?;
    ClusteringPipeline::render_reachability_plot(&reachability, Path::new("img/optics_reachability.png"))?;

    let mut info = format!("Number of samples in dataset: {}\n", records.shape()[0]);
    info.push_str(&format!("K-Means: Assigned {} points to 5 clusters\n", kmeans_res.sample_count));
    info.push_str(&format!("DBSCAN: Detected {} noise points (black), {} clusters\n", dbscan_res.noise_count, dbscan_res.cluster_counts.len()));
    if let Some(score) = dbscan_res.silhouette_score {
        info.push_str(&format!("DBSCAN Silhouette score: {:.4}", score));
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 500.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Clustering Visualizer",
        options,
        Box::new(move |cc| Ok(Box::new(OpticsPlotApp::new(cc, info)))),
    ).map_err(|e| anyhow::anyhow!("Eframe error: {:?}", e))?;

    Ok(())
}
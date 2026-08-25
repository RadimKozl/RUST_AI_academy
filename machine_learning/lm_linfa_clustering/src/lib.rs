use anyhow::{Context, Result};
use linfa::dataset::{DatasetBase, Labels};
use linfa::metrics::SilhouetteScore;
use linfa::traits::{Fit, Predict, Transformer};
use linfa_clustering::{Dbscan, KMeans, Optics};
use ndarray::Array2;
use plotters::prelude::*;
use polars::prelude::*;
use std::fs::File;
use std::path::Path;

pub struct ClusteringPipeline;

pub struct DbscanResult {
    pub noise_count: usize,
    pub cluster_counts: Vec<(usize, usize)>,
    pub silhouette_score: Option<f64>, // Changed from f32 to f64
}

impl ClusteringPipeline {
    pub fn load_dataset(path: &Path) -> Result<Array2<f64>> {
        let file = File::open(path)
            .with_context(|| format!("Unable to open CSV file {:?}", path))?;

        let df = CsvReader::new(file).finish()?;

        let rows = df.height();
        let income_col = df.column("Annual Income (k$)")?;
        let score_col = df.column("Spending Score (1-100)")?;

        let income_series = income_col.as_materialized_series();
        let score_series = score_col.as_materialized_series();

        let income_ca = income_series.i64()?;
        let score_ca = score_series.i64()?;

        let mut data = Vec::with_capacity(rows * 2);
        for (inc, scr) in income_ca.iter().zip(score_ca.iter()) {
            data.push(inc.unwrap_or(0) as f64);
            data.push(scr.unwrap_or(0) as f64);
        }

        Ok(Array2::from_shape_vec((rows, 2), data)?)
    }

    pub fn run_kmeans(records: &Array2<f64>, n_clusters: usize) -> Result<usize> {
        let dataset = DatasetBase::from(records.clone());
        let model = KMeans::params(n_clusters)
            .max_n_iterations(300)
            .tolerance(1e-4)
            .fit(&dataset)?;

        let kmeans_dataset = model.predict(dataset);
        Ok(kmeans_dataset.targets().len())
    }

    pub fn run_dbscan(records: &Array2<f64>, min_points: usize, tolerance: f64) -> Result<DbscanResult> {
        let dataset = DatasetBase::from(records.clone());
        let dbscan_memberships = Dbscan::params(min_points)
            .tolerance(tolerance)
            .transform(dataset)?;

        let mut label_count_map = dbscan_memberships.label_count().remove(0);
        let noise_count = label_count_map.remove(&None).unwrap_or(0);

        let mut cluster_counts: Vec<(usize, usize)> = label_count_map
            .into_iter()
            .filter_map(|(k, v)| k.map(|id| (id, v)))
            .collect();
        cluster_counts.sort_by_key(|&(id, _)| id);

        let silhouette_score = dbscan_memberships.silhouette_score().ok();

        Ok(DbscanResult {
            noise_count,
            cluster_counts,
            silhouette_score,
        })
    }

    pub fn run_optics(records: &Array2<f64>, min_points: usize, tolerance: f64) -> Result<Vec<f64>> {
        let optics_analysis = Optics::params(min_points)
            .tolerance(tolerance)
            .transform(records.view())?;

        let reachability_vec: Vec<f64> = optics_analysis
            .iter()
            .map(|x| x.reachability_distance().unwrap_or(f64::INFINITY))
            .collect();

        Ok(reachability_vec)
    }

    pub fn render_reachability_plot(reachability: &[f64], output_file: &Path) -> Result<()> {
        if let Some(parent) = output_file.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let root = BitMapBackend::new(output_file, (800, 400)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_y = reachability
            .iter()
            .filter(|&&val| val.is_finite())
            .cloned()
            .fold(0.0, f64::max);

        let safe_max_y = if max_y > 0.0 { max_y } else { 1.0 };

        let mut chart = ChartBuilder::on(&root)
            .caption("OPTICS Reachability Plot", ("sans-serif", 24).into_font())
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0..reachability.len(), 0.0..safe_max_y * 1.05)?;

        chart
            .configure_mesh()
            .x_desc("Index vzorku")
            .y_desc("Reachability Distance")
            .draw()?;

        chart.draw_series(LineSeries::new(
            reachability
                .iter()
                .enumerate()
                .map(|(idx, &val)| (idx, if val.is_finite() { val } else { safe_max_y })),
            &BLUE,
        ))?;

        root.present()?;
        Ok(())
    }
}
use anyhow::Result;
use linfa::dataset::DatasetBase;
use linfa::prelude::*;
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::{ArrayBase, Dim, OwnedRepr};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub type IrisDataset = DatasetBase<
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>>,
>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEvaluation {
    pub criterion: String,
    pub accuracy: f32,
    pub features_used: Vec<usize>,
    pub confusion_matrix_str: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedTreeFeatures {
    pub gini_features: Vec<usize>,
    pub entropy_features: Vec<usize>,
}

pub struct DecisionTreePipeline;

impl DecisionTreePipeline {
    /// Loads and splits the Iris dataset into a training and test set (80/20)
    pub fn load_and_split_dataset(seed: u64) -> (IrisDataset, IrisDataset) {
        let mut rng = SmallRng::seed_from_u64(seed);
        linfa_datasets::iris()
            .shuffle(&mut rng)
            .split_with_ratio(0.8)
    }

    /// Trains and evaluates a model with a given splitting criterion
    pub fn train_and_eval(
        train: &IrisDataset,
        test: &IrisDataset,
        split_quality: SplitQuality,
        criterion_name: &str,
        min_weight_split: f32,
        min_weight_leaf: f32,
    ) -> Result<(ModelEvaluation, DecisionTree<f64, usize>)> {
        let model = DecisionTree::params()
            .split_quality(split_quality)
            .max_depth(Some(100))
            .min_weight_split(min_weight_split)
            .min_weight_leaf(min_weight_leaf)
            .fit(train)?;

        let pred_y = model.predict(test);
        let cm = pred_y.confusion_matrix(test)?;

        let eval = ModelEvaluation {
            criterion: criterion_name.to_string(),
            accuracy: cm.accuracy() * 100.0,
            features_used: model.features(),
            confusion_matrix_str: format!("{:?}", cm),
        };

        Ok((eval, model))
    }

    /// Exports the tree to LaTeX TikZ format
    pub fn export_tikz(model: &DecisionTree<f64, usize>, output_path: &Path) -> Result<()> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut tikz_file = File::create(output_path)?;
        let tikz_code = model.export_to_tikz().with_legend().to_string();
        tikz_file.write_all(tikz_code.as_bytes())?;
        Ok(())
    }

    /// Saves selected model properties to a JSON file
    pub fn save_model_info(info: &SavedTreeFeatures, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, info)?;
        Ok(())
    }
}
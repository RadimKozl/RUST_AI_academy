use anyhow::Result;
use linfa::composing::MultiClassModel;
use linfa::dataset::DatasetBase;
use linfa::prelude::*;
use linfa_svm::Svm;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;

pub type WineDataset = DatasetBase<Array2<f64>, Array1<usize>>;
pub type BinaryWineDataset = DatasetBase<Array2<f64>, Array1<bool>>;

pub type BinarySvmModel = Svm<f64, bool>;
pub type MultiSvmModel = MultiClassModel<Array2<f64>, usize>;
pub type SvrModel = Svm<f64, f64>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SvmEvaluation {
    pub model_type: String,
    pub accuracy_or_mse: f64,
    pub mcc: Option<f64>,
    pub confusion_matrix: String,
}

pub struct SvmPipeline;

impl SvmPipeline {
    pub fn save_model<T: Serialize>(model: &T, filename: &str) -> Result<()> {
        let dir = Path::new("models");
        if !dir.exists() {
            create_dir_all(dir)?;
        }

        let filepath = dir.join(filename);
        let serialized = serde_json::to_string_pretty(model)?;
        let mut file = File::create(filepath)?;
        file.write_all(serialized.as_bytes())?;

        Ok(())
    }

    pub fn load_multiclass_wine(seed: u64) -> (WineDataset, WineDataset) {
        let mut rng = StdRng::seed_from_u64(seed);
        linfa_datasets::winequality()
            .shuffle(&mut rng)
            .split_with_ratio(0.9)
    }

    pub fn load_binary_wine(seed: u64) -> (BinaryWineDataset, BinaryWineDataset) {
        let mut rng = StdRng::seed_from_u64(seed);
        linfa_datasets::winequality()
            .map_targets(|&x| x > 6)
            .shuffle(&mut rng)
            .split_with_ratio(0.9)
    }

    pub fn train_binary_svm(
        train: &BinaryWineDataset,
        valid: &BinaryWineDataset,
        gamma: f64,
        pos_weight: f64,
        neg_weight: f64,
    ) -> Result<(SvmEvaluation, BinarySvmModel)> {
        let model = Svm::<_, bool>::params()
            .pos_neg_weights(pos_weight, neg_weight)
            .gaussian_kernel(gamma)
            .fit(train)?;

        let pred = model.predict(valid);
        let cm = pred.confusion_matrix(valid)?;

        let eval = SvmEvaluation {
            model_type: "Binary Classification (C-SVC)".to_string(),
            accuracy_or_mse: (cm.accuracy() * 100.0) as f64,
            mcc: Some(cm.mcc().into()),
            confusion_matrix: format!("{:?}", cm),
        };

        Ok((eval, model))
    }

    pub fn train_multiclass_svm(
        train: &WineDataset,
        valid: &WineDataset,
        gamma: f64,
    ) -> Result<(SvmEvaluation, MultiSvmModel)> {
        let params = Svm::<_, Pr>::params().gaussian_kernel(gamma);

        let model = train
            .one_vs_all()?
            .into_iter()
            .map(|(label, dataset)| (label, params.fit(&dataset).unwrap()))
            .collect::<MultiSvmModel>();

        let pred = model.predict(valid);
        let cm = pred.confusion_matrix(valid)?;

        let eval = SvmEvaluation {
            model_type: "Multi-Class One-Vs-All SVM".to_string(),
            accuracy_or_mse: (cm.accuracy() * 100.0) as f64,
            mcc: Some(cm.mcc().into()),
            confusion_matrix: format!("{:?}", cm),
        };

        Ok((eval, model))
    }

    pub fn train_svr_regression(c: f64, eps: f64, gamma: f64) -> Result<(f64, String, SvrModel)> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut x_vec: Vec<f64> = (0..40).map(|_| rng.gen_range(0.0..5.0)).collect();
        x_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let x = Array1::from_vec(x_vec);
        let mut y = x.mapv(|v| v.sin());

        y.iter_mut()
            .enumerate()
            .filter(|(i, _)| i % 5 == 0)
            .for_each(|(_, y_val)| *y_val = 3. * (0.5 - rng.gen_range(0.0..1.0)));

        let x = x.into_shape_with_order((40, 1))?;
        let dataset = DatasetBase::new(x, y);

        let model = Svm::params()
            .c_svr(c, Some(eps))
            .gaussian_kernel(gamma)
            .fit(&dataset)?;

        let predicted = model.predict(&dataset);
        let mse = predicted.mean_squared_error(&dataset)?;

        Ok((mse, format!("{}", model), model))
    }
}
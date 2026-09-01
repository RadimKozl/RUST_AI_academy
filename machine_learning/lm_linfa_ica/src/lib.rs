use anyhow::Result;
use linfa::dataset::DatasetBase;
use linfa::traits::{Fit, Predict};
use linfa_ica::fast_ica::{FastIca, GFunc};
use ndarray::{array, concatenate, Array, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcaEvaluation {
    pub n_samples: usize,
    pub original_signals: Vec<(f64, f64)>,
    pub mixed_signals: Vec<(f64, f64)>,
    pub unmixed_signals: Vec<(f64, f64)>,
}

pub struct IcaPipeline;

impl IcaPipeline {
    pub fn create_data() -> (Array2<f64>, Array2<f64>) {
        let nsamples = 2000;

        // Signal 1: Sine wave
        let source1 = Array::linspace(0., 8., nsamples).mapv(|x| (2f64 * x).sin());

        // Signal 2: Sawtooth / square wave
        let source2 = Array::linspace(0., 8., nsamples).mapv(|x| {
            let tmp = (4f64 * x).sin();
            if tmp > 0. {
                1.
            } else {
                -1.
            }
        });

        // Connecting both signals
        let mut sources_original = concatenate![
            Axis(1),
            source1.insert_axis(Axis(1)),
            source2.insert_axis(Axis(1))
        ];

        // Deterministic noise without external dependencies on the crate `rand`
        let mut seed: u64 = 42;
        sources_original.mapv_inplace(|x| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = ((seed >> 33) as f64) / (u32::MAX as f64) * 0.2;
            x + noise
        });

        // Signal mixing
        let mixing = array![[1., 1.], [0.5, 2.]];
        let sources_mixed = sources_original.dot(&mixing.t());

        (sources_original, sources_mixed)
    }

    pub fn run_ica() -> Result<IcaEvaluation> {
        let (sources_original, sources_mixed) = Self::create_data();

        let ica = FastIca::params().gfunc(GFunc::Logcosh(1.0));
        let model = ica
            .fit(&DatasetBase::from(sources_mixed.view()))
            .map_err(|e| anyhow::anyhow!("ICA Fit failed: {:?}", e))?;

        let sources_ica = model.predict(&sources_mixed);

        let n_samples = sources_original.nrows();
        let mut orig_vec = Vec::with_capacity(n_samples);
        let mut mix_vec = Vec::with_capacity(n_samples);
        let mut ica_vec = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            orig_vec.push((sources_original[[i, 0]], sources_original[[i, 1]]));
            mix_vec.push((sources_mixed[[i, 0]], sources_mixed[[i, 1]]));
            ica_vec.push((sources_ica[[i, 0]], sources_ica[[i, 1]]));
        }

        Ok(IcaEvaluation {
            n_samples,
            original_signals: orig_vec,
            mixed_signals: mix_vec,
            unmixed_signals: ica_vec,
        })
    }

    pub fn save_to_models(eval: &IcaEvaluation, filename: &str) -> Result<()> {
        let dir = Path::new("models");
        if !dir.exists() {
            fs::create_dir_all(dir)?;
        }
        let file_path = dir.join(filename);
        let json_data = serde_json::to_string_pretty(eval)?;
        fs::write(file_path, json_data)?;
        Ok(())
    }
}
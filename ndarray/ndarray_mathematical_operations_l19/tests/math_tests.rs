#[cfg(test)]
mod tests {
    use ndarray::{array, Array1};

    #[test]
    fn test_trig_identity() {
        // Explicit f64 typing for arrays
        let x: Array1<f64> = array![0.1, 0.5, 1.0, 1.2];
        
        // Alternatively, the type inside mapv can be annotated as (v: f64)
        let sin2 = x.mapv(|v: f64| v.sin().powi(2));
        let cos2 = x.mapv(|v: f64| v.cos().powi(2));
        let identity = sin2 + cos2;

        for val in identity.iter() {
            assert!((val - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_log_exp_roundtrip() {
        let original: Array1<f64> = array![0.5, 1.5, 10.0, 50.0];
        let roundtrip = original.mapv(f64::ln).mapv(f64::exp);

        for (a, b) in original.iter().zip(roundtrip.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
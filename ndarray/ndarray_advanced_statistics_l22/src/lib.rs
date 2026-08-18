use ndarray::{Array1, Array2, Axis};

/// Calculation of basic statistical indicators (Mean, Std Dev, Variance)
pub struct ArrayStats {
    pub mean: f64,
    pub var: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

pub fn compute_basic_stats(arr: &Array1<f64>) -> Option<ArrayStats> {
    if arr.is_empty() {
        return None;
    }

    let mean = arr.mean()?;

    // Clean and idiomatic calculation of variance (ddof = 0 for population)
    let var = arr.fold(0.0, |acc, &x| acc + (x - mean).powi(2)) / (arr.len() as f64);
    let std = var.sqrt();

    // Min and Max via standard fold (avoid NaN problems with f64)
    let min = arr.iter().copied().fold(f64::INFINITY, f64::min);
    let max = arr.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    Some(ArrayStats {
        mean,
        var,
        std,
        min,
        max,
    })
}

/// Calculating Median using in-place Quickselect (O(N) complexity)
pub fn median(arr: &Array1<f64>) -> Option<f64> {
    if arr.is_empty() {
        return None;
    }
    let mut v = arr.to_vec();
    let len = v.len();
    let mid = len / 2;

    if len % 2 == 1 {
        let (_, &mut elem, _) = v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
        Some(elem)
    } else {
        let (_, &mut elem1, _) = v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
        let (_, &mut elem2, _) = v.select_nth_unstable_by(mid - 1, |a, b| a.partial_cmp(b).unwrap());
        Some((elem1 + elem2) / 2.0)
    }
}

/// Cumulative sum (np.cumsum)
pub fn cumsum(arr: &Array1<f64>) -> Array1<f64> {
    let mut acc = 0.0;
    arr.mapv(|x| {
        acc += x;
        acc
    })
}

/// Average along the axes of a 2D matrix (np.mean(matrix, axis=0 or 1))
pub fn mean_by_axis(matrix: &Array2<f64>, axis: usize) -> Option<Array1<f64>> {
    let ax = match axis {
        0 => Axis(0),
        1 => Axis(1),
        _ => return None,
    };
    matrix.mean_axis(ax)
}
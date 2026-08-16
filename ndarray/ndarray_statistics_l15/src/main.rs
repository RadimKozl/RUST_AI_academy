use ndarray::{array, Array1, Array2, Axis};
use ndarray_stats::{Quantile1dExt, QuantileExt};
use noisy_float::types::n64;

fn main() {
    println!("=== 1. Basic Aggregation (Sum, Mean, Min, Max) ===");
    let data: Array2<f64> = array![
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0]
    ];

    let total_sum = data.sum();
    let mean_val = data.mean().unwrap();
    let min_val = data.min().unwrap();
    let max_val = data.max().unwrap();

    println!("Total Sum: {}", total_sum);
    println!("Mean: {}", mean_val);
    println!("Min: {}, Max: {}\n", min_val, max_val);

    println!("=== 2. Aggregation along Axes (Axis Reductions) ===");
    let sum_axis0 = data.sum_axis(Axis(0));
    let mean_axis1 = data.mean_axis(Axis(1)).unwrap();

    println!("Sum along Axis(0): {:?}", sum_axis0);
    println!("Mean along Axis(1): {:?}\n", mean_axis1);

    println!("=== 3. Calculation of Median (via ndarray-stats & noisy_float) ===");
    let vec_data = vec![15.0, 3.0, 90.0, 45.0, 12.0];
    let noisy_vec: Vec<_> = vec_data.into_iter().map(n64).collect();
    let mut arr_1d: Array1<_> = Array1::from(noisy_vec);

    let median_val = arr_1d.quantile_mut(n64(0.5), &ndarray_stats::interpolate::Nearest)
        .unwrap();

    println!("1D Array: {:?}", arr_1d.mapv(|x| x.raw()));
    println!("Median Value: {}\n", median_val.raw());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_aggregations() {
        let arr: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(arr.sum(), 15.0);
        assert_eq!(arr.mean().unwrap(), 3.0);
        assert_eq!(*arr.min().unwrap(), 1.0);
        assert_eq!(*arr.max().unwrap(), 5.0);
    }

    #[test]
    fn test_axis_reductions() {
        let mat: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(mat.sum_axis(Axis(0)), array![4.0, 6.0]);
        assert_eq!(mat.mean_axis(Axis(1)).unwrap(), array![1.5, 3.5]);
    }

    #[test]
    fn test_median_calculation() {
        let raw = vec![7.0, 1.0, 3.0];
        let noisy: Vec<_> = raw.into_iter().map(n64).collect();
        let mut arr = Array1::from(noisy);

        let med = arr.quantile_mut(n64(0.5), &ndarray_stats::interpolate::Nearest).unwrap();
        assert_eq!(med.raw(), 3.0);
    }
}
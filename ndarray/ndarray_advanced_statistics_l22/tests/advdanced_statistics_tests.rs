#[cfg(test)]
mod tests {
    use ndarray::array;
    
    use ndarray_advanced_statistics_l22::{compute_basic_stats, cumsum, mean_by_axis, median};

    #[test]
    fn test_basic_stats() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_basic_stats(&data).unwrap();

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.var, 2.0);
    }

    #[test]
    fn test_median_odd_and_even() {
        let odd_data = array![7.0, 1.0, 3.0];
        assert_eq!(median(&odd_data), Some(3.0));

        let even_data = array![1.0, 2.0, 10.0, 20.0];
        assert_eq!(median(&even_data), Some(6.0));
    }

    #[test]
    fn test_cumsum() {
        let data = array![1.0, 2.0, 3.0];
        let expected = array![1.0, 3.0, 6.0];
        assert_eq!(cumsum(&data), expected);
    }

    #[test]
    fn test_mean_axis() {
        let matrix = array![
            [1.0, 10.0],
            [3.0, 30.0]
        ];
        let col_means = mean_by_axis(&matrix, 0).unwrap();
        let row_means = mean_by_axis(&matrix, 1).unwrap();

        assert_eq!(col_means, array![2.0, 20.0]);
        assert_eq!(row_means, array![5.5, 16.5]);
    }
}
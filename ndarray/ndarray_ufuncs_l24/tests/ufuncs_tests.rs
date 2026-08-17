#[cfg(test)]
mod tests {
    use ndarray::array;
    use ndarray_ufuncs_l24::{
        custom_gelu_ufunc, diff_ufunc, gcd_lcm_ufuncs, intersect1d_ufunc,
        product_and_sum_ufuncs, unique_ufunc,
    };

    #[test]
    fn test_diff_and_accumulators() {
        let arr = array![10.0, 15.0, 25.0];
        let diffs = diff_ufunc(&arr);
        assert_eq!(diffs, array![5.0, 10.0]);

        let (sum, prod) = product_and_sum_ufuncs(&arr);
        assert_eq!(sum, 50.0);
        assert_eq!(prod, 3750.0);
    }

    #[test]
    fn test_gcd_lcm() {
        let a = array![12, 15];
        let b = array![18, 20];
        let (gcd_res, lcm_res) = gcd_lcm_ufuncs(&a, &b);

        assert_eq!(gcd_res, array![6, 5]);
        assert_eq!(lcm_res, array![36, 60]);
    }

    #[test]
    fn test_set_operations() {
        let a = array![4, 2, 2, 1, 4];
        let b = array![2, 3, 4];

        assert_eq!(unique_ufunc(&a), array![1, 2, 4]);
        assert_eq!(intersect1d_ufunc(&a, &b), array![2, 4]);
    }

    #[test]
    fn test_custom_gelu() {
        let x = array![0.0];
        let out = custom_gelu_ufunc(&x);
        assert!((out[0] - 0.0).abs() < 1e-6);
    }
}
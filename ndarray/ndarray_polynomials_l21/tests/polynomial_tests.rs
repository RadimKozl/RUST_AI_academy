#[cfg(test)]
mod tests {
    use ndarray::array;
    use ndarray_polynomials_l21::{polyder, polyint, polyval, quadratic_roots};

    #[test]
    fn test_horner_evaluation() {
        // P(x) = 5 - 2x + 3x^2
        let coeffs = array![5.0, -2.0, 3.0];
        // P(3) = 5 - 6 + 27 = 26
        assert_eq!(polyval(&coeffs, 3.0), 26.0);
    }

    #[test]
    fn test_poly_derivative_integral_roundtrip() {
        let p = array![1.0, 4.0, 9.0]; // 1 + 4x + 9x^2
        let der = polyder(&p);          // 4 + 18x
        let back = polyint(&der, 1.0);  // 1 + 4x + 9x^2

        for (a, b) in p.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_polynomial_roots() {
        // P(x) = x^2 - 5x + 6 = (x - 2)(x - 3) -> c0=6, c1=-5, c2=1
        let roots = quadratic_roots(6.0, -5.0, 1.0);
        let r1 = roots[0].0;
        let r2 = roots[1].0;

        assert!((r1 - 3.0).abs() < 1e-10 || (r1 - 2.0).abs() < 1e-10);
        assert!((r2 - 3.0).abs() < 1e-10 || (r2 - 2.0).abs() < 1e-10);
    }
}
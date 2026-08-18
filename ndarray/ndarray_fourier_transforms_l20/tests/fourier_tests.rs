#[cfg(test)]
mod tests {
    use ndarray::{array, Array1};
    use num_complex::Complex;
    use rustfft::FftPlanner;

    fn fft_1d_test(signal: &Array1<Complex<f64>>, inverse: bool) -> Array1<Complex<f64>> {
        let len = signal.len();
        let mut planner = FftPlanner::new();
        let fft = if inverse {
            planner.plan_fft_inverse(len)
        } else {
            planner.plan_fft_forward(len)
        };

        let mut buffer = signal.to_vec();
        fft.process(&mut buffer);

        let mut result = Array1::from_vec(buffer);
        if inverse {
            result.map_inplace(|v| *v /= len as f64);
        }
        result
    }

    fn convolve_1d_test(signal: &Array1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
        let s_len = signal.len();
        let k_len = kernel.len();
        let out_len = s_len + k_len - 1;
        let mut output = Array1::zeros(out_len);

        for i in 0..out_len {
            let mut sum = 0.0;
            for j in 0..k_len {
                if i >= j && (i - j) < s_len {
                    sum += signal[i - j] * kernel[j];
                }
            }
            output[i] = sum;
        }

        output
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let original: Array1<Complex<f64>> = array![
            Complex::new(1.0, 0.0),
            Complex::new(2.5, 0.0),
            Complex::new(-3.0, 0.0),
            Complex::new(4.2, 0.0)
        ];

        let spectrum = fft_1d_test(&original, false);
        let reconstructed = fft_1d_test(&spectrum, true);

        for (a, b) in original.iter().zip(reconstructed.iter()) {
            assert!((a.re - b.re).abs() < 1e-10);
            assert!((a.im - b.im).abs() < 1e-10);
        }
    }

    // ==========================================
    // Tests for Convolution
    // ==========================================

    #[test]
    fn test_convolution_basic() {
        let signal = array![1.0, 2.0, 3.0, 4.0];
        let kernel = array![0.5, 0.5];

        let convolved = convolve_1d_test(&signal, &kernel);
        // Expected manual calculations for mode="full":
        // y[0] = 1.0 * 0.5 = 0.5
        // y[1] = 2.0 * 0.5 + 1.0 * 0.5 = 1.5
        // y[2] = 3.0 * 0.5 + 2.0 * 0.5 = 2.5
        // y[3] = 4.0 * 0.5 + 3.0 * 0.5 = 3.5
        // y[4] = 4.0 * 0.5 = 2.0
        let expected = array![0.5, 1.5, 2.5, 3.5, 2.0];

        for (a, b) in convolved.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_convolution_identity_delta() {
        let signal = array![10.0, -5.0, 3.0, 8.0];
        // Dirac delta as kernel: [1.0] unmoved signal will not change
        let kernel = array![1.0];

        let convolved = convolve_1d_test(&signal, &kernel);
        assert_eq!(convolved, signal);
    }
}
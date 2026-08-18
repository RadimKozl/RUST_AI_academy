use ndarray::{array, Array1};
use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Applies Forward or Inverse FFT to a 1D ndarray of complex numbers
pub fn fft_1d(signal: &Array1<Complex<f64>>, inverse: bool) -> Array1<Complex<f64>> {
    let len = signal.len();
    let mut planner = FftPlanner::new();
    let fft = if inverse {
        planner.plan_fft_inverse(len)
    } else {
        planner.plan_fft_forward(len)
    };

    // Convert ndarray to Vec for rustfft
    let mut buffer: Vec<Complex<f64>> = signal.to_vec();
    fft.process(&mut buffer);

    let mut result = Array1::from_vec(buffer);

    // With IFFT, it is necessary to divide the result by the signal length (normalization)
    if inverse {
        result.map_inplace(|v| *v /= len as f64);
    }

    result
}

/// Discrete Convolution of two 1D arrays ("full" mode)
pub fn convolve_1d(signal: &Array1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
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

fn main() {
    println!("=== 1. Generating the Input Signal ===");
    let sample_count = 8;
    // Generating a sine signal: sin(2 * PI * t)
    let time_steps: Array1<f64> = Array1::linspace(0.0, 1.0, sample_count);
    let real_signal: Array1<f64> = time_steps.mapv(|t| (2.0 * PI * t).sin());

    // Convert real signal to complex (imaginary part = 0.0)
    let complex_signal: Array1<Complex<f64>> =
        real_signal.mapv(|v| Complex::new(v, 0.0));

    println!("Time signal (real values):\n{:.4}", real_signal);

    println!("\n=== 2. Forward Fast Fourier Transform (FFT) ===");
    let fft_spectrum = fft_1d(&complex_signal, false);
    println!("Frequency spectrum (Complex numbers):\n{:.4}", fft_spectrum);

    // Amplitude spectrum |X[k]|
    let magnitudes = fft_spectrum.mapv(|c| c.norm());
    println!("Amplitudes of spectral components:\n{:.4}", magnitudes);

    println!("\n=== 3. Inverse FFT (IFFT) ===");
    let reconstructed_complex = fft_1d(&fft_spectrum, true);
    let reconstructed_real = reconstructed_complex.mapv(|c| c.re);

    println!("Reconstructed signal from IFFT:\n{:.4}", reconstructed_real);

    println!("\n=== 4. Signal Convolution (Convolution) ===");
    let input_signal = array![1.0, 2.0, 3.0, 4.0];
    let filter_kernel = array![0.5, 0.5]; // Simple moving average (Smoothing filter)

    let convolved = convolve_1d(&input_signal, &filter_kernel);
    println!("Input signal: {}", input_signal);
    println!("Filter kernel: {}", filter_kernel);
    println!("Convolution result (Full):\n{:.4}", convolved);
}
use ndarray::{Array1};

fn main() {
    println!("=== 1. Numerical Ranges with Step (NumPy: np.arange) ===");

    // Integers via Rust std iterator
    let int_range: Array1<i32> = Array1::from_iter(0..10);
    println!("Integer range (0..10):\n{:?}\n", int_range);

    // Floating line numbers with step (start, stop, step)
    let float_range: Array1<f64> = Array1::range(0.0, 2.0, 0.5);
    println!("Float range (0.0 to 2.0, step 0.5):\n{:?}\n", float_range);

    println!("=== 2. Linear Space Generation (NumPy: np.linspace) ===");

    // 5 evenly spaced points between 0.0 and 10.0 (including both boundaries)
    let linear_space: Array1<f64> = Array1::linspace(0.0, 10.0, 5);
    println!("Linspace (0.0 to 10.0, 5 elements):\n{:?}\n", linear_space);

    println!("=== 3. Logarithmic Space Generation (NumPy: np.logspace) ===");

    // 4 points in logarithmic scale: base^x for x from 1.0 to 3.0 (base 10.0)
    // Generates: [10^1, 10^1.666..., 10^2.333..., 10^3]
    let log_space: Array1<f64> = Array1::logspace(10.0, 1.0, 3.0, 4);
    println!("Logspace (base 10, 10^1 to 10^3, 4 elements):\n{:?}\n", log_space);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linspace_boundaries() {
        let arr = Array1::linspace(0.0, 1.0, 11);
        assert_eq!(arr.len(), 11);
        assert_eq!(arr[0], 0.0);
        assert_eq!(arr[10], 1.0);
        assert_eq!(arr[5], 0.5);
    }

    #[test]
    fn test_logspace_values() {
        let arr = Array1::logspace(10.0, 0.0, 2.0, 3);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], 1.0);   // 10^0
        assert_eq!(arr[1], 10.0);  // 10^1
        assert_eq!(arr[2], 100.0); // 10^2
    }

    #[test]
    fn test_range_step() {
        let arr = Array1::range(0.0, 5.0, 1.5);
        assert_eq!(arr.as_slice().unwrap(), &[0.0, 1.5, 3.0, 4.5]);
    }
}

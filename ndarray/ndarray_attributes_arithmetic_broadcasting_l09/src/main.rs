use ndarray::{array, Array1, Array2};
use std::mem::size_of;

fn main() {
    println!("=== 1. Array Attributes & Memory Layout ===");

    let matrix: Array2<f64> = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ];

    println!("Shape: {:?}", matrix.shape());
    println!("NDim: {}", matrix.ndim());
    println!("Total Size (elements): {}", matrix.len());
    println!("Strides: {:?}", matrix.strides());
    println!("Itemsize (bytes): {}", size_of::<f64>());
    println!("Total memory size (bytes): {}\n", matrix.len() * size_of::<f64>());

    println!("=== 2. Element-Wise Arithmetic Operations ===");

    let a: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
    let b: Array2<f32> = array![[10.0, 20.0], [30.0, 40.0]];

    println!("Addition (a + b):\n{:?}", &a + &b);
    println!("Subtraction (a - b):\n{:?}", &a - &b);
    println!("Multiplication (a * b):\n{:?}", &a * &b);
    println!("Division (a / b):\n{:?}\n", &a / &b);

    println!("=== 3. Explicit and Implicit Broadcasting ===");

    let mat: Array2<f32> = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ];
    let row_vec: Array1<f32> = array![10.0, 20.0, 30.0];

    // Implicit broadcasting: Adding a 1D vector to a 2D matrix along the corresponding axis
    let broadcasted_add = &mat + &row_vec;
    println!("Matrix + 1D Row Vector (Broadcasted):\n{:?}\n", broadcasted_add);

    // Scalar broadcasting
    let scalar_mul = &mat * 2.0;
    println!("Matrix * Scalar (2.0):\n{:?}\n", scalar_mul);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_attributes() {
        let arr: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.len(), 6);
        assert_eq!(size_of::<i32>(), 4);
    }

    #[test]
    fn test_broadcasting_rules() {
        let mat = array![[1.0, 2.0], [3.0, 4.0]];
        let vec = array![10.0, 20.0];

        let result = &mat + &vec;

        assert_eq!(result, array![[11.0, 22.0], [13.0, 24.0]]);
    }

    #[test]
    fn test_elementwise_arithmetic() {
        let a = array![2.0, 4.0];
        let b = array![2.0, 2.0];

        assert_eq!(&a * &b, array![4.0, 8.0]);
        assert_eq!(&a / &b, array![1.0, 2.0]);
    }
}

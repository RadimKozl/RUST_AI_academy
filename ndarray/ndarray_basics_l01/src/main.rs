use ndarray::{array, Array1, Array2, ArrayD, IxDyn};

fn main() {
    println!("=== 1. Creating fields ===");

    // NumPy: a = np.array([1, 2, 3])
    let a: Array1<i32> = array![1, 2, 3];
    println!("1D Array (array! macro):\n{:?}\n", a);

    // NumPy: b = np.zeros((2, 3), dtype=float)
    let b: Array2<f64> = Array2::zeros((2, 3));
    println!("2D Zeros Matrix (2x3):\n{:?}\n", b);

    // NumPy: c = np.ones((3, 2), dtype=int)
    let c: Array2<i32> = Array2::ones((3, 2));
    println!("2D Ones Matrix (3x2):\n{:?}\n", c);

    // NumPy: d = np.arange(0, 10, 2) -> linspace nebo Array1::from_iter
    let d: Array1<f64> = Array1::linspace(0.0, 8.0, 5);
    println!("Linspace Array (0.0 to 8.0, 5 elements):\n{:?}\n", d);

    println!("=== 2. Checking attributes (Shape, Strides, Ndim) ===");

    let matrix: Array2<f32> = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ];

    // NumPy: matrix.ndim
    println!("Dimensions (ndim): {}", matrix.ndim());

    // NumPy: matrix.shape
    println!("Shape: {:?}", matrix.shape());

    // NumPy: matrix.strides (in Rust in number of elements, not bytes)
    println!("Strides: {:?}", matrix.strides());

    // NumPy: matrix.size
    println!("Total elements (len): {}", matrix.len());

    println!("\n=== 3. Dynamické dimenze (ArrayD) ===");
    
    // NumPy: np.zeros((2, 2, 2)) without static dimension type
    let dyn_array: ArrayD<f32> = ArrayD::zeros(IxDyn(&[2, 2, 2]));
    println!("Dynamic 3D Array shape: {:?}", dyn_array.shape());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_creation_and_shape() {
        let arr: Array2<f64> = Array2::from_elem((3, 4), 42.0);
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr[[0, 0]], 42.0);
        assert_eq!(arr.len(), 12);
    }
}
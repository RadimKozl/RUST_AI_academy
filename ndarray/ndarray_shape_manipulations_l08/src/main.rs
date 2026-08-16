use ndarray::{array, concatenate, stack, Array1, Array2, Axis};

fn main() {
    println!("=== 1. Reshaping & Memory Layout ===");

    let a: Array1<i32> = Array1::from_iter(0..6);
    // Reshape (6,) -> (2, 3) zero-copy
    let reshaped: Array2<i32> = a.into_shape_with_order((2, 3)).unwrap();
    println!("Reshaped (2x3):\n{:?}\n", reshaped);

    println!("=== 2. Flattening (Ravel vs Flatten) ===");

    let matrix: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];

    // Zero-copy view as slice (NumPy: ravel)
    if let Some(slice) = matrix.as_slice() {
        println!("Flat view as slice: {:?}", slice);
    }

    // Owned copy (NumPy: flatten)
    let flat_owned: Array1<i32> = matrix.iter().cloned().collect();
    println!("Flat owned Array1: {:?}\n", flat_owned);

    println!("=== 3. Transposition & Axis Swapping ===");

    let t_matrix = matrix.t(); // Zero-copy, only changes strides
    println!("Transposed Matrix (3x2):\n{:?}", t_matrix);
    println!("Original Strides: {:?}, Transposed Strides: {:?}\n", matrix.strides(), t_matrix.strides());

    println!("=== 4. Concatenation & Stacking ===");

    let b1: Array2<i32> = array![[1, 2], [3, 4]];
    let b2: Array2<i32> = array![[5, 6], [7, 8]];

    // Concatenate along existing axis 0: (2,2) + (2,2) -> (4,2)
    let concat_ax0 = concatenate(Axis(0), &[b1.view(), b2.view()]).unwrap();
    println!("Concatenated Axis 0:\n{:?}\n", concat_ax0);

    // Stack along NEW axis 0: (2,2) + (2,2) -> (2, 2, 2)
    let stacked = stack(Axis(0), &[b1.view(), b2.view()]).unwrap();
    println!("Stacked Shape: {:?}", stacked.shape());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transposition_strides() {
        let arr = array![[1, 2, 3], [4, 5, 6]];
        let transposed = arr.t();

        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(arr.strides(), &[3, 1]);
        assert_eq!(transposed.strides(), &[1, 3]);
        assert_eq!(arr[[0, 1]], transposed[[1, 0]]);
    }

    #[test]
    fn test_stack_creates_new_axis() {
        let a = array![1, 2, 3];
        let b = array![4, 5, 6];
        let stacked = stack(Axis(0), &[a.view(), b.view()]).unwrap();

        assert_eq!(stacked.ndim(), 2);
        assert_eq!(stacked.shape(), &[2, 3]);
    }
}
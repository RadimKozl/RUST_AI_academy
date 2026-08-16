use ndarray::{array, concatenate, stack, Array2, Axis};

fn main() {
    println!("=== 1. Transposition & Axis Swapping (NumPy: transpose, swapaxes) ===");

    let a: Array2<i32> = array![
        [1, 2, 3],
        [4, 5, 6]
    ];
    println!("Original Matrix (2x3):\n{:?}\nStrides: {:?}", a, a.strides());

    // Transpose (2x3 -> 3x2) - zero allocations, change strides interpretation
    let a_transposed = a.t();
    println!("Transposed Matrix (3x2):\n{:?}\nStrides: {:?}\n", a_transposed, a_transposed.strides());

    // Swap specific axes
    let a_swapped = a.clone().reversed_axes();
    println!("Reversed Axes:\n{:?}\n", a_swapped);


    println!("=== 2. Joining Arrays (NumPy: concatenate, stack) ===");

    let b1: Array2<i32> = array![[1, 2], [3, 4]];
    let b2: Array2<i32> = array![[5, 6], [7, 8]];

    // Concatenate along axis 0 (Rows): (2x2) + (2x2) -> (4x2)
    let concat_axis0 = concatenate(Axis(0), &[b1.view(), b2.view()]).unwrap();
    println!("Concatenate Axis(0):\n{:?}\n", concat_axis0);

    // Concatenate along axis 1 (Columns): (2x2) + (2x2) -> (2x4)
    let concat_axis1 = concatenate(Axis(1), &[b1.view(), b2.view()]).unwrap();
    println!("Concatenate Axis(1):\n{:?}\n", concat_axis1);

    // Stacking: Create NEW dimension -> (2x2) + (2x2) -> (2x2x2)
    let stacked = stack(Axis(0), &[b1.view(), b2.view()]).unwrap();
    println!("Stacked Array Shape (Axis 0): {:?}\n", stacked.shape());


    println!("=== 3. Dimension Alterations (NumPy: expand_dims, squeeze) ===");

    let c: Array2<i32> = array![[1, 2, 3]]; // Shape (1, 3)

    // 3a. Expand dimension (expand_dims) -> Shape (1, 1, 3)
    let c_expanded = c.clone().insert_axis(Axis(0));
    println!("Expanded shape: {:?}", c_expanded.shape());

    // 3b. Squeeze for static types: Removing a specific 1-element axis
    let c_squeezed_static = c.clone().remove_axis(Axis(0)); // Shape (3,)
    println!("Static Squeezed (remove_axis) shape: {:?}", c_squeezed_static.shape());

    // 3c. Squeeze for dynamic types: Convert to ArrayD and call squeeze()
    let c_dyn_squeezed = c.into_dyn().squeeze();
    println!("Dynamic Squeezed shape: {:?}\n", c_dyn_squeezed.shape());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transposition_strides() {
        let arr: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
        let transposed = arr.t();

        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(transposed.shape(), &[3, 2]);

        assert_eq!(arr.strides(), &[3, 1]);
        assert_eq!(transposed.strides(), &[1, 3]);

        assert_eq!(arr[[0, 1]], transposed[[1, 0]]);
    }

    #[test]
    fn test_concatenation_mismatch_fails() {
        let a1: Array2<i32> = array![[1, 2], [3, 4]];
        let a2: Array2<i32> = array![[5, 6, 7]];

        assert!(concatenate(Axis(0), &[a1.view(), a2.view()]).is_err());
    }

    #[test]
    fn test_remove_axis() {
        let arr: Array2<i32> = array![[1, 2, 3]]; // (1, 3)
        let removed = arr.remove_axis(Axis(0));   // (3,)
        
        assert_eq!(removed.ndim(), 1);
        assert_eq!(removed.shape(), &[3]);
    }
}
use ndarray::{array, Array1, Array2, ShapeBuilder};

fn main() {
    println!("=== 1. Reshaping (NumPy: reshape) ===");

    let a: Array1<i32> = Array1::from_iter(0..6);
    println!("Original 1D array:\n{:?}\n", a);

    // Transform 1D (6,) -> 2D (2, 3) without reallocation
    let b: Array2<i32> = a.into_shape_with_order((2, 3)).unwrap();
    println!("Reshaped 2D (2x3):\n{:?}\n", b);

    println!("=== 2. Flattening & Views (NumPy: ravel vs flatten) ===");

    let matrix: Array2<i32> = array![
        [1, 2, 3],
        [4, 5, 6]
    ];

    // Zero-copy view as 1D (NumPy: ravel)
    if let Some(slice) = matrix.as_slice() {
        println!("Flat view as slice (zero-copy): {:?}", slice);
    }

    // Convert to 1D owned array (NumPy: flatten - make a copy/reallocate if needed)
    let flat_owned: Array1<i32> = matrix.iter().cloned().collect();
    println!("Flat owned Array1:\n{:?}\n", flat_owned);

    // Modern idiomatic way to extract the raw storage buffer
    let (vec_data, offset) = flat_owned.into_raw_vec_and_offset();
    println!("Raw Vec<i32>: {:?} (offset: {:?})\n", vec_data, offset);

    println!("=== 3. Memory Layout (C-Order vs Fortran-Order) ===");

    // Create a 2D array explicitly in Fortran order (column-major)
    let f_matrix = Array2::from_shape_vec((2, 3).f(), vec![1, 4, 2, 5, 3, 6]).unwrap();
    println!("Fortran-order Matrix (2x3):\n{:?}", f_matrix);
    println!("Strides pro F-order: {:?}", f_matrix.strides());

    // C-order flattening vs F-order flattening
    let c_flat: Vec<_> = f_matrix.iter().cloned().collect(); // std iterator respects C layout
    println!("Iterated elements (row-major order): {:?}", c_flat);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reshape_and_flatten() {
        let arr = Array2::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();
        
        // Reshape to 3x2
        let reshaped = arr.clone().into_shape_with_order((3, 2)).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped[[0, 0]], 1);
        assert_eq!(reshaped[[2, 1]], 6);

        // Modern raw vector extraction
        let (flat_vec, _offset) = arr.into_raw_vec_and_offset();
        assert_eq!(flat_vec, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_non_contiguous_reshape_fails() {
        let arr = array![[1, 2, 3], [4, 5, 6]];
        let sliced = arr.slice(ndarray::s![.., ..;2]); // Discontinuous array (step 2)
        
        // Reshaping a disjoint view without allocation will fail
        assert!(sliced.to_shape((3, 1)).is_err());
    }
}
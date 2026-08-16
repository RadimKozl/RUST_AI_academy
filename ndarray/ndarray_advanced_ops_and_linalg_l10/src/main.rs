use ndarray::{array, Array1, Array2};
use std::collections::HashSet;

fn main() {
    println!("=== 1. Linear Algebra (Matrix Multiplication & Dot Product) ===");

    let a: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let b: Array2<f64> = array![[5.0, 6.0], [7.0, 8.0]];

    // Matmul via .dot()
    let dot_res = a.dot(&b);
    println!("Matrix Product (a.dot(b)):\n{:?}\n", dot_res);

    println!("=== 2. Element-wise Comparisons & Searching ===");

    let arr: Array1<i32> = array![10, 25, 30, 45, 50];
    
    // Condition mapping (NumPy: arr > 25)
    let gt_25: Vec<bool> = arr.iter().map(|&x| x > 25).collect();
    println!("Elements > 25 (mask): {:?}", gt_25);

    // Find indices where condition holds (NumPy: np.where(arr > 25))
    let indices: Vec<usize> = arr
        .iter()
        .enumerate()
        .filter_map(|(idx, &val)| if val > 25 { Some(idx) } else { None })
        .collect();
    println!("Indices of elements > 25: {:?}\n", indices);

    println!("=== 3. Unique Rows & Sorting ===");

    let data: Array1<i32> = array![4, 2, 2, 8, 4, 1, 8];
    
    // Unique values via std::collections::HashSet
    let mut unique_set = HashSet::new();
    let unique_vec: Vec<i32> = data.iter().cloned().filter(|x| unique_set.insert(*x)).collect();
    println!("Unique elements: {:?}", unique_vec);

    // Sorting 1D array
    let mut sorted_data = data.to_vec();
    sorted_data.sort_unstable();
    println!("Sorted elements: {:?}\n", sorted_data);

    println!("=== 4. Binary / Bitwise Operations ===");

    let bin1: Array1<u8> = array![0b1010, 0b1100];
    let bin2: Array1<u8> = array![0b0110, 0b0100];

    println!("Bitwise AND: {:?}", &bin1 & &bin2);
    println!("Bitwise OR:  {:?}", &bin1 | &bin2);
    println!("Bitwise XOR: {:?}", &bin1 ^ &bin2);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiplication() {
        let a = array![[1, 2], [3, 4]];
        let b = array![[2, 0], [1, 2]];
        let expected = array![[4, 4], [10, 8]];

        assert_eq!(a.dot(&b), expected);
    }

    #[test]
    fn test_bitwise_operations() {
        let a = array![0b1010u8];
        let b = array![0b1100u8];

        assert_eq!(&a & &b, array![0b1000u8]);
    }
}
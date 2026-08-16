use ndarray::{array, s, Array2, ArrayView2};

fn main() {
    println!("=== 1. Basic 2D Slicing (NumPy: arr[0:2, 1:3]) ===");

    let matrix: Array2<i32> = array![
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33]
    ];
    println!("Original Matrix (3x4):\n{:?}\n", matrix);

    // Cut: first 2 rows and columns at index 1 to 2 (exclusively 3)
    let subview: ArrayView2<i32> = matrix.slice(s![0..2, 1..3]);
    println!("Subview (0..2, 1..3):\n{:?}", subview);
    println!("Subview shape: {:?}\n", subview.shape());


    println!("=== 2. Strided Slicing (NumPy: arr[::2, ::2]) ===");

    // Cut with step 2 across both rows and columns
    let strided_view = matrix.slice(s![..;2, ..;2]);
    println!("Strided View (step 2):\n{:?}\n", strided_view);


    println!("=== 3. In-Place Mutable Slicing ===");

    let mut mutable_matrix: Array2<i32> = array![
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ];

    // Cutout of the lower right corner (2x2) and modify in place
    {
        let mut sub_slice = mutable_matrix.slice_mut(s![1.., 1..]);
        sub_slice.mapv_inplace(|x| x * 10);
    } // Mutable borrow ends here

    println!("Modified Original Matrix after in-place slicing:\n{:?}\n", mutable_matrix);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_view_pointers() {
        let arr: Array2<i32> = array![[1, 2], [3, 4]];
        let sub = arr.slice(s![1.., 0..1]);

        assert_eq!(sub[[0, 0]], 3);
        // The cut array points directly to the address in the original array
        assert_eq!(sub.as_ptr(), &arr[[1, 0]] as *const i32);
    }

    #[test]
    fn test_mutable_slicing() {
        let mut arr: Array2<i32> = Array2::zeros((3, 3));
        
        {
            let mut center = arr.slice_mut(s![1..2, 1..2]);
            center[[0, 0]] = 99;
        }

        assert_eq!(arr[[1, 1]], 99);
        assert_eq!(arr[[0, 0]], 0);
    }
}
use ndarray::{array, Zip, Array1, Array2, Axis};

fn main() {
    println!("=== 1. Boolean Indexing / Masking (NumPy: arr[arr > 5]) ===");

    let data: Array2<i32> = array![
        [1, 8, 3],
        [4, 5, 9]
    ];

    // Create a boolean mask
    let mask = data.mapv(|x| x > 5);
    println!("Mask (data > 5):\n{:?}\n", mask);

    // Extract elements meeting the condition into a 1D array
    let filtered: Vec<i32> = data
        .iter()
        .zip(mask.iter())
        .filter_map(|(&val, &keep)| if keep { Some(val) } else { None })
        .collect();
    println!("Filtered values (Vec): {:?}\n", filtered);


    println!("=== 2. Fancy Indexing / Selection (NumPy: arr[[0, 2], :]) ===");

    let matrix: Array2<i32> = array![
        [10, 11, 12],
        [20, 21, 22],
        [30, 31, 32],
        [40, 41, 42]
    ];

    // Select specific rows by indices [0, 3] along Axis(0)
    let selected_rows = matrix.select(Axis(0), &[0, 3]);
    println!("Selected rows (0 and 3):\n{:?}\n", selected_rows);


    println!("=== 3. Axis Iteration (NumPy: for x in np.nditer(a)) ===");

    let tensor: Array2<i32> = array![
        [1, 2],
        [3, 4]
    ];

    // Iteration after the first dimension (Outer Iteration - rows)
    println!("Outer iteration (rows):");
    for row in tensor.outer_iter() {
        println!("  Row view: {:?}", row);
    }

    // Iterate along a specific axis (Axis 1 = columns)
    println!("\nAxis iteration (columns):");
    for col in tensor.axis_iter(Axis(1)) {
        println!("  Col view: {:?}", col);
    }


    println!("\n=== 4. Lock-step Parallel Iteration (Zip Macro) ===");

    let mut a: Array1<i32> = array![1, 2, 3];
    let b: Array1<i32> = array![10, 20, 30];

    // In-place operations over multiple fields at once without allocation
    Zip::from(&mut a).and(&b).for_each(|x, &y| {
        *x += y;
    });
    println!("Zipped result (a + b in-place): {:?}", a);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_rows() {
        let arr: Array2<i32> = array![[1, 2], [3, 4], [5, 6]];
        let selected = arr.select(Axis(0), &[0, 2]);

        assert_eq!(selected.shape(), &[2, 2]);
        assert_eq!(selected[[0, 0]], 1);
        assert_eq!(selected[[1, 0]], 5);
    }

    #[test]
    fn test_zip_lockstep() {
        let mut a = array![1, 2];
        let b = array![3, 4];

        Zip::from(&mut a).and(&b).for_each(|x, &y| *x *= y);

        assert_eq!(a, array![3, 8]);
    }
}
use ndarray::{aview1, aview2, Array1, ArrayView1};

fn main() {
    println!("=== 1. Array Views from Standard Slices (Zero-Copy) ===");

    let raw_data: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // NumPy: np.asarray(raw_data) -> Create a read-only view of the slice
    let view_1d: ArrayView1<f64> = ArrayView1::from(&raw_data[..]);
    println!("1D View from slice:\n{:?}", view_1d);

    // Using the aview1 macro
    let view_macro = aview1(&raw_data);
    println!("1D View (aview1):\n{:?}", view_macro);

    // 2D View of 1D slice (2x3) without allocation
    let view_2d = aview2(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]);
    println!("2D View (aview2):\n{:?}\n", view_2d);


    println!("=== 2. Creating Owned Array from Vec (Move semantics) ===");

    let vec_data: Vec<i32> = vec![10, 20, 30, 40, 50];
    
    // NumPy: np.fromiter / np.array(vec) -> Take ownership of buffer without copying
    let owned_array: Array1<i32> = Array1::from_vec(vec_data);
    println!("Owned Array1 from Vec:\n{:?}\n", owned_array);


    println!("=== 3. Creating Array from Text/Iterators (NumPy: frombuffer/fromiter) ===");

    let text_data = "1.5 2.5 3.5 4.5";
    
    // Convert text stream/string to float array via iterator
    let parsed_array: Array1<f32> = text_data
        .split_whitespace()
        .map(|s| s.parse::<f32>().unwrap())
        .collect(); // Array1 implements FromIterator

    println!("Parsed Array1 from string iterator:\n{:?}\n", parsed_array);

    // Parsing a byte buffer (example of reading binary data from a database/network)
    let byte_buffer: &[u8] = b"HELLO";
    let byte_array: Array1<u8> = byte_buffer.iter().copied().collect();
    println!("Byte Array1 from byte slice:\n{:?}\n", byte_array);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_view() {
        let slice_data = [10.0, 20.0, 30.0];
        let view = ArrayView1::from(&slice_data[..]);

        assert_eq!(view.shape(), &[3]);
        assert_eq!(view[1], 20.0);
        // Verify that the data pointer is identical to the original slice
        assert_eq!(view.as_ptr(), slice_data.as_ptr());
    }

    #[test]
    fn test_from_vec_no_reallocation() {
        let mut vec = Vec::with_capacity(100);
        vec.extend_from_slice(&[1, 2, 3, 4]);
        
        let ptr_before = vec.as_ptr();
        let arr = Array1::from_vec(vec);

        assert_eq!(arr.as_ptr(), ptr_before);
        assert_eq!(arr.len(), 4);
    }

    #[test]
    fn test_from_iterator() {
        let arr: Array1<i32> = (0..5).map(|x| x * 2).collect();
        assert_eq!(arr, Array1::from_vec(vec![0, 2, 4, 6, 8]));
    }
}
use ndarray::{array, Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct DataRecord {
    id: u32,
    score: f64,
    label: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 1. Sorting Along Axes & Fancy Indexing ===");

    let mut matrix: Array2<i32> = array![
        [5, 2, 9],
        [1, 7, 3]
    ];

    // Sort each row in-place
    for mut row in matrix.rows_mut() {
        if let Some(slice) = row.as_slice_mut() {
            slice.sort();
        }
    }
    println!("Matrix sorted along rows:\n{:?}\n", matrix);

    // Argsort on 1D array
    let values: Array1<i32> = array![40, 10, 30, 20];
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by_key(|&i| values[i]);
    println!("Original values: {:?}", values);
    println!("Argsort indices: {:?}\n", indices);

    println!("=== 2. Axis Insertion (np.newaxis) ===");

    let a: Array1<i32> = array![1, 2, 3];
    
    // Adding an axis via the insert_axis method: (3,) -> (1, 3)
    let row_matrix = a.view().insert_axis(Axis(0));
    println!("Row Matrix shape (1, 3): {:?}", row_matrix.shape());

    // Adding an axis via the insert_axis method: (3,) -> (3, 1)
    let col_matrix = a.view().insert_axis(Axis(1));
    println!("Col Matrix shape (3, 1): {:?}\n", col_matrix.shape());

    println!("=== 3. Structured Records & Binary I/O ===");

    let dataset = vec![
        DataRecord { id: 1, score: 95.5, label: "A".into() },
        DataRecord { id: 2, score: 82.1, label: "B".into() },
    ];

    let file_path = "dataset.bin";
    let encoded: Vec<u8> = bincode::serialize(&dataset)?;
    let mut file = File::create(file_path)?;
    file.write_all(&encoded)?;
    println!("Saved {} records to binary file.", dataset.len());

    let mut read_file = File::open(file_path)?;
    let mut buffer = Vec::new();
    read_file.read_to_end(&mut buffer)?;
    let decoded: Vec<DataRecord> = bincode::deserialize(&buffer)?;
    println!("Loaded records: {:?}\n", decoded);

    std::fs::remove_file(file_path)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_sorting() {
        let mut arr = array![[3, 1, 2], [9, 7, 8]];
        for mut row in arr.rows_mut() {
            if let Some(slice) = row.as_slice_mut() {
                slice.sort();
            }
        }
        assert_eq!(arr, array![[1, 2, 3], [7, 8, 9]]);
    }

    #[test]
    fn test_insert_axis() {
        let vec = array![10, 20, 30];
        let expanded = vec.view().insert_axis(Axis(0));
        assert_eq!(expanded.shape(), &[1, 3]);
    }

    #[test]
    fn test_structured_record_serialization() {
        let record = DataRecord { id: 10, score: 3.14, label: "test".into() };
        let encoded = bincode::serialize(&record).unwrap();
        let decoded: DataRecord = bincode::deserialize(&encoded).unwrap();
        assert_eq!(record, decoded);
    }
}
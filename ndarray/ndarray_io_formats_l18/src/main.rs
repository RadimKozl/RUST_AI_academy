use ndarray::Array2;
use ndarray_io_formats_l18::{
    load_csv, load_json, load_parquet, load_txt_dat, save_csv, save_json, save_parquet,
    save_txt_dat,
};
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let original = Array2::from_shape_vec((2, 3), vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5])?;
    println!("--- Original ndarray Matrix ---\n{:?}\n", original);

    // 1. CSV Test
    save_csv(&original, "matrix.csv")?;
    let csv_loaded = load_csv("matrix.csv", 2, 3)?;
    assert_eq!(original, csv_loaded);
    println!("[OK] CSV roundtrip successful.");

    // 2. JSON Test
    save_json(&original, "matrix.json")?;
    let json_loaded = load_json("matrix.json")?;
    assert_eq!(original, json_loaded);
    println!("[OK] JSON roundtrip successful.");

    // 3. TXT / DAT Test
    save_txt_dat(&original, "matrix.dat")?;
    let dat_loaded = load_txt_dat("matrix.dat")?;
    assert_eq!(original, dat_loaded);
    println!("[OK] DAT/TXT roundtrip successful.");

    // 4. Parquet Test
    save_parquet(&original, "matrix.parquet")?;
    let parquet_loaded = load_parquet("matrix.parquet")?;
    assert_eq!(original, parquet_loaded);
    println!("[OK] Parquet roundtrip successful.");

    // Cleaning up temporary files
    for file in &["matrix.csv", "matrix.json", "matrix.dat", "matrix.parquet"] {
        if Path::new(file).exists() {
            std::fs::remove_file(file)?;
        }
    }

    println!("\nAll I/O formats successfully verified!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_all_formats_consistency() {
        let data = array![[10.0, 20.0], [30.0, 40.0]];

        // CSV
        save_csv(&data, "test.csv").unwrap();
        assert_eq!(data, load_csv("test.csv", 2, 2).unwrap());
        std::fs::remove_file("test.csv").unwrap();

        // JSON
        save_json(&data, "test.json").unwrap();
        assert_eq!(data, load_json("test.json").unwrap());
        std::fs::remove_file("test.json").unwrap();

        // DAT
        save_txt_dat(&data, "test.dat").unwrap();
        assert_eq!(data, load_txt_dat("test.dat").unwrap());
        std::fs::remove_file("test.dat").unwrap();

        // Parquet
        save_parquet(&data, "test.parquet").unwrap();
        assert_eq!(data, load_parquet("test.parquet").unwrap());
        std::fs::remove_file("test.parquet").unwrap();
    }
}
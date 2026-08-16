use ndarray::{array, Array1, Array2, Axis};

fn main() {
    println!("=== 1. Identifying missing values ​​(f64 NaN) ===");
    let data_nan: Array2<f64> = array![
        [1.0, f64::NAN, 3.0],
        [4.0, 5.0, f64::NAN],
        [7.0, 8.0, 9.0]
    ];

    // Detect NaN values ​​in the array (create a boolean mask)
    let is_missing = data_nan.mapv(|x| x.is_nan());
    println!("Data Matrix:\n{:?}", data_nan);
    println!("Is Missing Mask:\n{:?}\n", is_missing);

    println!("=== 2. Removing rows with missing data (Filtering) ===");
    // Filter out rows that contain at least one NaN
    let clean_rows: Vec<_> = data_nan
        .outer_iter()
        .filter(|row| !row.iter().any(|x| x.is_nan()))
        .collect();

    // Create a new matrix from the filtered rows
    if !clean_rows.is_empty() {
        let view_refs: Vec<_> = clean_rows.iter().map(|r| r.view()).collect();
        let cleaned_matrix = ndarray::stack(Axis(0), &view_refs).unwrap();
        println!("Cleaned Matrix (Rows without NaN):\n{:?}\n", cleaned_matrix);
    }

    println!("=== 3. Imputation of missing data (By column average) ===");
    let mut imputed_matrix = data_nan.clone();

    // Loop through the columns and replace NaN with the average of that column
    for mut col in imputed_matrix.columns_mut() {
        let valid_values: Vec<f64> = col.iter().copied().filter(|x| !x.is_nan()).collect();
        
        if !valid_values.is_empty() {
            let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
            col.mapv_inplace(|x| if x.is_nan() { mean } else { x });
        }
    }
    println!("Imputed Matrix (Mean Imputation):\n{:?}\n", imputed_matrix);

    println!("=== 4. Type-safe missing data using Option<T> ===");
    let data_option: Array1<Option<i32>> = array![Some(10), None, Some(30), None, Some(50)];

    // Identifying missing elements
    let missing_count = data_option.iter().filter(|x| x.is_none()).count();
    println!("Option Array: {:?}", data_option);
    println!("Missing elements count: {}", missing_count);

    // Imputation by constant (Default value replacement)
    let default_imputed: Array1<i32> = data_option.mapv(|x| x.unwrap_or(0));
    println!("Imputed Option Array (with default 0): {:?}\n", default_imputed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nan_detection() {
        let arr = array![1.0, f64::NAN, 2.0];
        let mask = arr.mapv(|x| x.is_nan());
        assert_eq!(mask, array![false, true, false]);
    }

    #[test]
    fn test_option_imputation() {
        let arr: Array1<Option<i32>> = array![Some(5), None, Some(15)];
        let imputed = arr.mapv(|x| x.unwrap_or(-1));
        assert_eq!(imputed, array![5, -1, 15]);
    }

    #[test]
    fn test_mean_imputation() {
        let mut arr: Array1<f64> = array![2.0, f64::NAN, 4.0];
        let valid_sum: f64 = arr.iter().filter(|x| !x.is_nan()).sum();
        let valid_count = arr.iter().filter(|x| !x.is_nan()).count();
        let mean = valid_sum / valid_count as f64;

        arr.mapv_inplace(|x| if x.is_nan() { mean } else { x });
        assert_eq!(arr, array![2.0, 3.0, 4.0]);
    }
}

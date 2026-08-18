use ndarray::array;

use ndarray_advanced_statistics_l22::{compute_basic_stats, cumsum, mean_by_axis, median};

fn main() {
    println!("=== 1. Descriptive Statistics (1D Array) ===");
    let data = array![10.0, 20.0, 30.0, 40.0, 50.0, 100.0];

    if let Some(stats) = compute_basic_stats(&data) {
        println!("Data:                      {}", data);
        println!("Mean:                      {:.2}", stats.mean);
        println!("Variance:                  {:.2}", stats.var);
        println!("Standard deviation (Std):  {:.2}", stats.std);
        println!("Min / Max:                 {} / {}", stats.min, stats.max);
    }

    if let Some(med) = median(&data) {
        println!("Median:               {:.2}", med);
    }

    println!("\n=== 2. Cumulative sum (np.cumsum) ===");
    let cum = cumsum(&data);
    println!("Cumsum:               {}", cum);

    println!("\n=== 3. Statistics along the axes of a 2D Matrix ===");
    let matrix = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];

    let mean_cols = mean_by_axis(&matrix, 0).unwrap();
    let mean_rows = mean_by_axis(&matrix, 1).unwrap();

    println!("Original matrix:\n{}", matrix);
    println!("Average by columns (Axis 0): {}", mean_cols);
    println!("Average by rows (Axis 1): {}", mean_rows);
}
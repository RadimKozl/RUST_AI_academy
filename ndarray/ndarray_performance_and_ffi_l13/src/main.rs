use ndarray::{array, par_azip, Array2};

fn process_matrix_in_place(mut matrix: Array2<f64>) -> Array2<f64> {
    // Paralelní in-place operace přes Rayon iterátor
    matrix.par_map_inplace(|x| *x *= 2.0);
    matrix
}

fn main() {
    println!("=== 1. Memory Layout & Contiguity Optimization ===");
    let row_major: Array2<f64> = Array2::zeros((1000, 1000));
    let col_major = row_major.t();

    println!("Row-Major standard layout: {}", row_major.is_standard_layout());
    println!("Transposed standard layout: {}", col_major.is_standard_layout());

    let contiguous_copy = col_major.to_shape((1000, 1000)).unwrap().to_owned();
    println!("Forced contiguous copy: {}\n", contiguous_copy.is_standard_layout());

    println!("=== 2. Parallel Processing with Rayon & Lock-Free Azip ===");
    let a = Array2::<f64>::ones((500, 500));
    let b = Array2::<f64>::from_elem((500, 500), 3.0);
    let mut c = Array2::<f64>::zeros((500, 500));

    // Lock-step paralelní výpočet: C = A * 2.0 + B
    par_azip!((c in &mut c, &a in &a, &b in &b) {
        *c = a * 2.0 + b;
    });

    println!("Parallel computation sample C[0,0]: {}", c[[0, 0]]);
    println!("Parallel computation sample C[499,499]: {}\n", c[[499, 499]]);

    println!("=== 3. Zero-Copy Processing ===");
    let input_matrix = array![[1.5, 2.5], [3.5, 4.5]];
    let processed = process_matrix_in_place(input_matrix);
    println!("Processed Array:\n{:?}\n", processed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_azip() {
        let mut dest = Array2::<i32>::zeros((10, 10));
        let src = Array2::<i32>::from_elem((10, 10), 5);

        par_azip!((d in &mut dest, &s in &src) {
            *d = s + 10;
        });

        assert_eq!(dest[[0, 0]], 15);
        assert_eq!(dest[[9, 9]], 15);
    }

    #[test]
    fn test_contiguity_check() {
        let arr = Array2::<f64>::zeros((5, 5));
        assert!(arr.is_standard_layout());

        let transposed = arr.t();
        assert!(!transposed.is_standard_layout());
    }
}
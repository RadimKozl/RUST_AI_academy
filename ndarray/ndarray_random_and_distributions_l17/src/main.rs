use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::{
    Binomial, ChiSquared, Exp, Normal, Pareto, Poisson, Uniform,
};
use ndarray_rand::RandomExt;
use rand::seq::SliceRandom;
use rand::thread_rng;

fn main() {
    let mut rng = thread_rng();

    println!("=== 1. Uniform & Normal Distributions ===");
    let uniform_dist = Uniform::new(0.0, 1.0).unwrap();
    let uniform_arr: Array2<f64> = Array2::random((2, 3), uniform_dist);
    println!("Uniform (2x3):\n{:?}", uniform_arr);

    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    let normal_arr: Array2<f64> = Array2::random((2, 3), normal_dist);
    println!("Normal (2x3):\n{:?}\n", normal_arr);

    println!("=== 2. Discrete Distributions (Binomial, Poisson) ===");
    let binomial_dist = Binomial::new(10, 0.5).unwrap();
    let poisson_dist = Poisson::new(3.0).unwrap();

    let binomial_arr: Array1<u64> = Array1::random(5, binomial_dist);
    let poisson_arr: Array1<f64> = Array1::random(5, poisson_dist);

    println!("Binomial (n=10, p=0.5): {:?}", binomial_arr);
    println!("Poisson (lambda=3.0): {:?}\n", poisson_arr);

    println!("=== 3. Continuous Specialized Distributions ===");
    let exp_dist = Exp::new(1.5).unwrap();
    let pareto_dist = Pareto::new(1.0, 1.0).unwrap();
    let chi_square_dist = ChiSquared::new(1.0).unwrap();

    let exp_arr: Array1<f64> = Array1::random(4, exp_dist);
    let pareto_arr: Array1<f64> = Array1::random(4, pareto_dist);
    let chi_square_arr: Array1<f64> = Array1::random(4, chi_square_dist);

    println!("Exponential: {:?}", exp_arr);
    println!("Pareto: {:?}", pareto_arr);
    println!("Chi-Square: {:?}\n", chi_square_arr);

    println!("=== 4. Permutations and In-Place Shuffling ===");
    let mut data_to_shuffle: Array1<i32> = Array1::from_vec(vec![10, 20, 30, 40, 50]);

    if let Some(slice) = data_to_shuffle.as_slice_mut() {
        slice.shuffle(&mut rng);
    }
    println!("Shuffled Array1: {:?}\n", data_to_shuffle);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_shapes() {
        let dist = Uniform::new(-1.0, 1.0).unwrap();
        let arr: Array2<f64> = Array2::random((4, 4), dist);
        assert_eq!(arr.shape(), &[4, 4]);
    }

    #[test]
    fn test_normal_bounds() {
        let dist = Normal::new(0.0, 1.0).unwrap();
        let arr: Array1<f64> = Array1::random(100, dist);
        assert_eq!(arr.len(), 100);
    }

    #[test]
    fn test_shuffling() {
        let mut arr = Array1::from_vec(vec![1, 2, 3, 4, 5]);
        let original_sum: i32 = arr.sum();

        if let Some(slice) = arr.as_slice_mut() {
            slice.shuffle(&mut thread_rng());
        }

        assert_eq!(arr.sum(), original_sum);
    }
}
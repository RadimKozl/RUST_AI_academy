use ndarray::{array, Array2};
use std::f64::consts::PI;

fn main() {
    println!("=== 1. Trigonometric functions ===");
    let angles_deg = array![0.0, 30.0, 45.0, 60.0, 90.0];
    let angles_rad = &angles_deg * (PI / 180.0);

    let sin_values = angles_rad.mapv(f64::sin);
    let cos_values = angles_rad.mapv(f64::cos);

    println!("Angles (rad): {:.4}", angles_rad);
    println!("Sine:         {:.4}", sin_values);
    println!("Cosine:       {:.4}", cos_values);

    let arcsin_vals = sin_values.mapv(f64::asin);
    println!("Arcsin (back to rad): {:.4}\n", arcsin_vals);

    println!("=== 2. Exponential and Logarithmic Functions ===");
    let x = array![1.0, 2.0, 3.0, 10.0];

    let exp_x = x.mapv(f64::exp);
    let ln_x = x.mapv(f64::ln);
    let log10_x = x.mapv(f64::log10);
    let log2_x = x.mapv(f64::log2);

    println!("Original:  {}", x);
    println!("exp(x):    {:.4}", exp_x);
    println!("ln(x):     {:.4}", ln_x);
    println!("log10(x):  {:.4}", log10_x);
    println!("log2(x):   {:.4}\n", log2_x);

    println!("=== 3. Hyperbolic functions ===");
    let h_input = array![-1.0, 0.0, 1.0];
    let sinh_vals = h_input.mapv(f64::sinh);
    let cosh_vals = h_input.mapv(f64::cosh);
    let tanh_vals = h_input.mapv(f64::tanh);

    println!("sinh: {:.4}", sinh_vals);
    println!("cosh: {:.4}", cosh_vals);
    println!("tanh: {:.4}\n", tanh_vals);

    println!("=== 4. Rounding function ===");
    let floats = array![-1.7, -1.2, 0.2, 1.5, 1.8];

    let floor_vals = floats.mapv(f64::floor);
    let ceil_vals = floats.mapv(f64::ceil);
    let round_vals = floats.mapv(f64::round);
    let trunc_vals = floats.mapv(f64::trunc);

    println!("Original: {}", floats);
    println!("floor:    {}", floor_vals);
    println!("ceil:     {}", ceil_vals);
    println!("round:    {}", round_vals);
    println!("trunc:    {}\n", trunc_vals);

    println!("=== 5. In-place Mutation & Optimization ===");
    let mut matrix: Array2<f64> = array![[1.0, 4.0], [9.0, 16.0]];
    matrix.map_inplace(|v| *v = v.sqrt());
    println!("Root-removed in-place:\n{}", matrix);
}

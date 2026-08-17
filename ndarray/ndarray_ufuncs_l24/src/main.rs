use ndarray::array;
use ndarray_ufuncs_l24::{
    custom_gelu_ufunc, diff_ufunc, gcd_lcm_ufuncs, intersect1d_ufunc, log_and_round_ufuncs,
    product_and_sum_ufuncs, trig_and_hyperbolic_ufuncs, unique_ufunc,
};

fn main() {
    println!("=== 1. Trigonometric & Hyperbolic Ufuncs ===");
    let angles = array![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI];
    let (sin_x, cos_x, sinh_x) = trig_and_hyperbolic_ufuncs(&angles);
    println!("Angles: {}\nSin:    {:.4}\nCos:    {:.4}\nSinh:   {:.4}", angles, sin_x, cos_x, sinh_x);

    println!("\n=== 2. Logarithmic, Rounding & Difference ===");
    let values = array![1.2, 2.7, 5.1, 10.8];
    let (logs, rounded) = log_and_round_ufuncs(&values);
    let diffs = diff_ufunc(&values);
    println!("Original:   {}", values);
    println!("Ln:         {:.4}", logs);
    println!("Rounded:    {}", rounded);
    println!("Diffs:      {}", diffs);

    println!("\n=== 3. Accumulative Ufuncs (Sum and Product) ===");
    let nums = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let (sum_val, prod_val) = product_and_sum_ufuncs(&nums);
    println!("Sum: {}, Product (Factorial of 5!): {}", sum_val, prod_val);

    println!("\n=== 4. Number Theory (GCD and LCM) ===");
    let a = array![12, 24, 30];
    let b = array![18, 16, 45];
    let (gcd_vals, lcm_vals) = gcd_lcm_ufuncs(&a, &b);
    println!("A: {}\nB: {}\nGCD: {}\nLCM: {}", a, b, gcd_vals, lcm_vals);

    println!("\n=== 5. Set operations ===");
    let arr1 = array![1, 3, 5, 3, 1, 7];
    let arr2 = array![5, 7, 9, 11];
    println!("Unique in arr1:                {}", unique_ufunc(&arr1));
    println!("Intersection of arr1 and arr2: {}", intersect1d_ufunc(&arr1, &arr2));

    println!("\n=== 6. Custom Ufunc (GELU Activation Function) ===");
    let input = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    let gelu_out = custom_gelu_ufunc(&input);
    println!("Input:    {}", input);
    println!("GELU Out: {:.4}", gelu_out);
}

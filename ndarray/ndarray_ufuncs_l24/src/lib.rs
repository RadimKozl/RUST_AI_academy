use ndarray::Array1;
use num_integer::Integer;

/// 1. Trigonometric and Hyperbolic ufuncs
pub fn trig_and_hyperbolic_ufuncs(x: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let sin_x = x.mapv(|val| val.sin());
    let cos_x = x.mapv(|val| val.cos());
    let sinh_x = x.mapv(|val| val.sinh());
    (sin_x, cos_x, sinh_x)
}

/// 2. Logarithmic and Rounding functions
pub fn log_and_round_ufuncs(x: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let log_x = x.mapv(|val| val.ln());
    let round_x = x.mapv(|val| val.round());
    (log_x, round_x)
}

/// 3. Difference ufunc (np.diff - calculates adjacent differences x[i+1] - x[i])
pub fn diff_ufunc(x: &Array1<f64>) -> Array1<f64> {
    if x.len() <= 1 {
        return Array1::zeros(0);
    }
    let mut result = Array1::zeros(x.len() - 1);
    for i in 0..x.len() - 1 {
        result[i] = x[i + 1] - x[i];
    }
    result
}

/// 4. Accumulation ufuncs: Product and Sum
pub fn product_and_sum_ufuncs(x: &Array1<f64>) -> (f64, f64) {
    let sum_val = x.sum();
    let prod_val = x.fold(1.0, |acc, &v| acc * v);
    (sum_val, prod_val)
}

/// 5. GCD and LCM ufuncs (Greatest Common Divisor and Least Common Multiple for integers)
pub fn gcd_lcm_ufuncs(a: &Array1<i64>, b: &Array1<i64>) -> (Array1<i64>, Array1<i64>) {
    let gcd_res = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.gcd(&y))
        .collect();

    let lcm_res = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.lcm(&y))
        .collect();

    (gcd_res, lcm_res)
}

/// 6. Set operations (Set operations: Unique, Intersection)
pub fn unique_ufunc<T: Ord + Copy>(arr: &Array1<T>) -> Array1<T> {
    let mut vec = arr.to_vec();
    vec.sort();
    vec.dedup();
    Array1::from_vec(vec)
}

pub fn intersect1d_ufunc<T: Ord + Copy>(a: &Array1<T>, b: &Array1<T>) -> Array1<T> {
    let mut set_a = a.to_vec();
    set_a.sort();
    set_a.dedup();

    let mut result = Vec::new();
    for item in b.iter() {
        if set_a.binary_search(item).is_ok() && !result.contains(item) {
            result.push(*item);
        }
    }
    result.sort();
    Array1::from_vec(result)
}

/// 7. Custom Vectorized Ufunc (Custom GELU activation function)
pub fn custom_gelu_ufunc(x: &Array1<f64>) -> Array1<f64> {
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let const_factor = (2.0 / std::f64::consts::PI).sqrt();
    x.mapv(|v| {
        0.5 * v * (1.0 + (const_factor * (v + 0.044715 * v.powi(3))).tanh())
    })
}
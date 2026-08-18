use ndarray::Array1;

/// 1. Evaluation of the polynomial at point x using the Horner scheme
/// The coefficients [c0, c1, c2, ...] correspond to P(x) = c0 + c1*x + c2*x^2 + ...
pub fn polyval(coeffs: &Array1<f64>, x: f64) -> f64 {
    coeffs
        .iter()
        .rev()
        .fold(0.0, |acc, &coeff| acc * x + coeff)
}

/// 2. Derivation of a polynomial
pub fn polyder(coeffs: &Array1<f64>) -> Array1<f64> {
    if coeffs.len() <= 1 {
        return ndarray::array![0.0];
    }
    let mut der = Array1::zeros(coeffs.len() - 1);
    for i in 1..coeffs.len() {
        der[i - 1] = coeffs[i] * (i as f64);
    }
    der
}

/// 3. Integration of a polynomial (with integration constant k)
pub fn polyint(coeffs: &Array1<f64>, k: f64) -> Array1<f64> {
    let mut integrated = Array1::zeros(coeffs.len() + 1);
    integrated[0] = k;
    for i in 0..coeffs.len() {
        integrated[i + 1] = coeffs[i] / ((i + 1) as f64);
    }
    integrated
}

/// 4. Multiplication of two polynomials (Discrete convolution of coefficients)
pub fn polymul(p1: &Array1<f64>, p2: &Array1<f64>) -> Array1<f64> {
    let out_len = p1.len() + p2.len() - 1;
    let mut res = Array1::zeros(out_len);
    for i in 0..p1.len() {
        for j in 0..p2.len() {
            res[i + j] += p1[i] * p2[j];
        }
    }
    res
}

/// 5. Calculating the roots of the quadratic polynomial a*x^2 + b*x + c = 0
pub fn quadratic_roots(c0: f64, c1: f64, c2: f64) -> Vec<(f64, f64)> {
    let a = c2;
    let b = c1;
    let c = c0;
    let disc = b * b - 4.0 * a * c;

    if disc >= 0.0 {
        let r1 = (-b + disc.sqrt()) / (2.0 * a);
        let r2 = (-b - disc.sqrt()) / (2.0 * a);
        vec![(r1, 0.0), (r2, 0.0)]
    } else {
        let real = -b / (2.0 * a);
        let imag = (-disc).sqrt() / (2.0 * a);
        vec![(real, imag), (real, -imag)]
    }
}
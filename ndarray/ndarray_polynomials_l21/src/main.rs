use ndarray::array;
use ndarray_polynomials_l21::{polyder, polyint, polymul, polyval, quadratic_roots};

fn main() {
    println!("=== 1. Polynomial: P(x) = 2 + 3x + 1x^2 (ie (x+1)(x+2)) ===");
    let p = array![2.0, 3.0, 1.0]; // c0=2, c1=3, c2=1

    let x_val = 2.0;
    let eval_res = polyval(&p, x_val);
    println!("P({}) = {}", x_val, eval_res);

    println!("\n=== 2. Derivation and Integration ===");
    let der = polyder(&p);
    println!("P'(x) coefficients: {}", der);

    let integ = polyint(&p, 0.0);
    println!("∫P(x) dx coefficients (k=0): {}", integ);

    println!("\n=== 3. Multiplication of polynomials ===");
    let p1 = array![1.0, 1.0];
    let p2 = array![2.0, 1.0];
    let prod = polymul(&p1, &p2);
    println!("(x + 1)(x + 2) = {}", prod);

    println!("\n=== 4. Finding Roots ===");
    let roots = quadratic_roots(p[0], p[1], p[2]);
    for (i, (re, im)) in roots.iter().enumerate() {
        if *im == 0.0 {
            println!("Root {}: {}", i + 1, re);
        } else {
            println!("Root {}: {} + {}i", i + 1, re, im);
        }
    }
}
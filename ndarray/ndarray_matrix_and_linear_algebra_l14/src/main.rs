use ndarray::{array, Array2};
use nalgebra::{DMatrix, DVector};

fn main() {
    println!("=== 1. Basic Matrix Operations (Element-wise & Multiplication) ===");
    let a: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let b: Array2<f64> = array![[5.0, 6.0], [7.0, 8.0]];

    // Element-wise addition, subtraction, multiplication
    let add_res = &a + &b;
    let sub_res = &a - &b;
    let elem_mul = &a * &b;

    // Matrix Multiplication (Dot product)
    let dot_res = a.dot(&b);

    println!("A + B:\n{:?}", add_res);
    println!("A - B:\n{:?}", sub_res);
    println!("Element-wise A * B:\n{:?}", elem_mul);
    println!("Matrix Product A.dot(B):\n{:?}\n", dot_res);

    println!("=== 2. Advanced Linear Algebra via nalgebra Interop ===");
    // Conversion from ndarray Array2 to nalgebra DMatrix
    let na_a = DMatrix::from_row_slice(2, 2, a.as_slice_memory_order().unwrap());

    // Determinant & Matrix Inversion
    let det = na_a.determinant();
    let inv = na_a.clone().try_inverse().expect("Matrix is singular");

    println!("Determinant of A: {}", det);
    println!("Inverse of A:\n{:?}\n", inv);

    // Matrix Norm (Frobenius Norm)
    let norm = na_a.norm();
    println!("Frobenius Norm of A: {}\n", norm);

    println!("=== 3. Solving Linear Equations (Ax = b) ===");
    // System: 1x + 2y = 5, 3x + 4y = 11
    let b_vec = DVector::from_vec(vec![5.0, 11.0]);
    let x_sol = na_a.clone().solve_lower_triangular(&b_vec)
        .unwrap_or_else(|| na_a.clone().lu().solve(&b_vec).unwrap());

    println!("Solution x for Ax = b:\n{:?}\n", x_sol);

    println!("=== 4. Decompositions: Eigenvalues & SVD ===");
    // Eigenvalues and Eigenvectors (Symmetric Matrix required for real EVD)
    let sym_matrix = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
    let eigen = sym_matrix.symmetric_eigen();

    println!("Eigenvalues:\n{:?}", eigen.eigenvalues);
    println!("Eigenvectors:\n{:?}\n", eigen.eigenvectors);

    // Singular Value Decomposition (SVD)
    let svd = na_a.svd(true, true);
    println!("SVD Singular Values:\n{:?}", svd.singular_values);
    println!("SVD U Matrix:\n{:?}", svd.u);
    println!("SVD V^T Matrix:\n{:?}\n", svd.v_t);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![[4.0, 5.0], [6.0, 7.0]];
        assert_eq!(a.dot(&b), b);
    }

    #[test]
    fn test_matrix_inverse() {
        let m = DMatrix::from_row_slice(2, 2, &[4.0, 7.0, 2.0, 6.0]);
        let inv = m.clone().try_inverse().unwrap();
        let identity = m * inv;

        let diff_00: f64 = identity[(0, 0)] - 1.0;
        let diff_11: f64 = identity[(1, 1)] - 1.0;
        
        assert!(diff_00.abs() < 1e-9);
        assert!(diff_11.abs() < 1e-9);
    }
}

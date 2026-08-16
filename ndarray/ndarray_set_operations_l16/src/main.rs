use ndarray::{array, Array1, Array2};
use std::collections::BTreeSet;

/// Returns unique elements from an N-dimensional array in sorted order.
fn unique_elements<T: Ord + Copy>(arr: &Array2<T>) -> Array1<T> {
    let set: BTreeSet<T> = arr.iter().copied().collect();
    Array1::from_iter(set)
}

/// Computes the intersection of two 1D arrays (sorted, unique elements).
fn array_intersection<T: Ord + Copy>(a: &Array1<T>, b: &Array1<T>) -> Array1<T> {
    let set_a: BTreeSet<T> = a.iter().copied().collect();
    let set_b: BTreeSet<T> = b.iter().copied().collect();
    
    let intersection = set_a.intersection(&set_b).copied();
    Array1::from_iter(intersection)
}

/// Computes the union of two 1D arrays (sorted, unique elements).
fn array_union<T: Ord + Copy>(a: &Array1<T>, b: &Array1<T>) -> Array1<T> {
    let set_a: BTreeSet<T> = a.iter().copied().collect();
    let set_b: BTreeSet<T> = b.iter().copied().collect();

    let union = set_a.union(&set_b).copied();
    Array1::from_iter(union)
}

/// Computes the set difference (A - B) of two 1D arrays (sorted, unique elements).
fn array_difference<T: Ord + Copy>(a: &Array1<T>, b: &Array1<T>) -> Array1<T> {
    let set_a: BTreeSet<T> = a.iter().copied().collect();
    let set_b: BTreeSet<T> = b.iter().copied().collect();

    let difference = set_a.difference(&set_b).copied();
    Array1::from_iter(difference)
}

fn main() {
    let matrix_a: Array2<i32> = array![
        [5, 2, 8, 2],
        [1, 5, 9, 1]
    ];
    let vec_b: Array1<i32> = array![2, 5, 12, 15];

    println!("=== 1. Unique Elements ===");
    let uniques = unique_elements(&matrix_a);
    println!("Matrix A:\n{:?}", matrix_a);
    println!("Unique elements of A: {:?}\n", uniques);

    println!("=== 2. Set Intersection ===");
    let intersect = array_intersection(&uniques, &vec_b);
    println!("Intersection of A and B: {:?}\n", intersect);

    println!("=== 3. Set Union ===");
    let union_res = array_union(&uniques, &vec_b);
    println!("Union of A and B: {:?}\n", union_res);

    println!("=== 4. Set Difference (A - B) ===");
    let diff_res = array_difference(&uniques, &vec_b);
    println!("Difference (A - B): {:?}\n", diff_res);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_elements() {
        let arr = array![[3, 1, 3], [2, 1, 2]];
        let uniques = unique_elements(&arr);
        assert_eq!(uniques, array![1, 2, 3]);
    }

    #[test]
    fn test_set_operations() {
        let a = array![1, 2, 3, 4];
        let b = array![3, 4, 5, 6];

        assert_eq!(array_intersection(&a, &b), array![3, 4]);
        assert_eq!(array_union(&a, &b), array![1, 2, 3, 4, 5, 6]);
        assert_eq!(array_difference(&a, &b), array![1, 2]);
    }
}
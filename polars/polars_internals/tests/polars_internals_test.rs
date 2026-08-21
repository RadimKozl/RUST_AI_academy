use polars::prelude::*;
use polars_internals::{extract_f64_column, process_advanced_expressions};

#[test]
fn test_window_expression_and_downcast() {
    let df = df!(
        "category" => &["X", "X"],
        "amount" => &[10.0, 30.0]
    )
    .unwrap();

    let res = process_advanced_expressions(df).unwrap();
    
    // The average of category X is 20.0. Only the second row (30.0) is > 20.0
    assert_eq!(res.height(), 1);

    let scores = extract_f64_column(&res, "relative_score").unwrap();
    assert_eq!(scores.len(), 1);
    assert!((scores[0] - 0.5).abs() < 1e-6); // (30 - 20) / 20 = 0.5
}
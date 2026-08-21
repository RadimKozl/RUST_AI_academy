use anyhow::{bail, Result};
use polars::prelude::*;

pub fn process_advanced_expressions(df: DataFrame) -> Result<DataFrame> {
    // 1. Get category average as Expr (added ? to unpack Result from over())
    let mean_expr = col("amount")
        .mean()
        .over([col("category")])?
        .alias("category_avg_amount");

    // 2. Building a lazy plan
    let lazy_plan = df
        .lazy()
        .with_column(mean_expr)
        .with_column(
            ((col("amount") - col("category_avg_amount")) / col("category_avg_amount"))
                .alias("relative_score"),
        )
        .filter(col("relative_score").gt(lit(0.0)));

    // 3. Explicitly using anyhow::Context resolves the conflict with PolarsContext
    let evaluated_df = anyhow::Context::context(
        lazy_plan.collect(),
        "Failed to execute advanced expression plan",
    )?;

    Ok(evaluated_df)
}

pub fn extract_f64_column(df: &DataFrame, col_name: &str) -> Result<Vec<f64>> {
    let series = df.column(col_name)?;
    let ca = series.f64().map_err(|_| {
        anyhow::anyhow!("Column '{}' is not of type Float64", col_name)
    })?;

    if ca.null_count() > 0 {
        bail!("Column '{}' contains null values", col_name);
    }

    Ok(ca.into_no_null_iter().collect())
}
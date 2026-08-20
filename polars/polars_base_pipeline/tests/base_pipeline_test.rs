use anyhow::Result;
use polars_base_pipeline::process_ecommerce_data;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_ecommerce_pipeline_transformations() -> Result<()> {
    let csv_content = "\
event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session
2019-11-01 00:00:00 UTC,view,1003461,2053013555631882655,electronics.smartphone,apple,100.0,5123,sess-1
2019-11-01 00:00:01 UTC,view,1003462,2053013555631882655,electronics.smartphone,samsung,200.0,5124,sess-2
";

    let mut temp_file = NamedTempFile::new()?;
    write!(temp_file, "{}", csv_content)?;

    let df = process_ecommerce_data(temp_file.path())?;

    assert_eq!(df.height(), 2);
    assert!(df.column("brand2").is_ok());
    assert!(df.column("price2").is_ok());
    assert!(df.column("avg_price_by_category").is_ok());

    let brand2 = df.column("brand2")?.str()?;
    assert_eq!(brand2.get(0), Some("brand-apple"));

    let price2 = df.column("price2")?.f64()?;
    assert_eq!(price2.get(0), Some(10000.0));

    let avg_price = df.column("avg_price_by_category")?.f64()?;
    assert_eq!(avg_price.get(0), Some(150.0));

    Ok(())
}
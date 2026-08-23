use polars_data_conversion::FeatureExtractor;
use sqlx::sqlite::SqlitePoolOptions;
use std::fs;
use tempfile::NamedTempFile;

async fn create_mock_db() -> (NamedTempFile, String) {
    let file = NamedTempFile::new().unwrap();
    let db_path = file.path().to_str().unwrap().to_string();
    let db_url = format!("sqlite://{}", db_path);

    let pool = SqlitePoolOptions::new()
        .connect(&db_url)
        .await
        .unwrap();

    sqlx::query(
        r#"
        CREATE TABLE market_tickers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            bid REAL NOT NULL,
            ask REAL NOT NULL,
            last_price REAL NOT NULL,
            volume REAL NOT NULL,
            timestamp INTEGER NOT NULL
        );
        "#,
    )
    .execute(&pool)
    .await
    .unwrap();

    let prices = vec![95005.0, 95015.0, 95010.0, 95025.0, 95035.0];
    for p in prices {
        sqlx::query("INSERT INTO market_tickers (symbol, bid, ask, last_price, volume, timestamp) VALUES ('tBTCUSD', 0, 0, ?, 1.0, 0)")
            .bind(p)
            .execute(&pool)
            .await
            .unwrap();
    }

    (file, db_url)
}

// Added flavor = "multi_thread" parameter
#[tokio::test(flavor = "multi_thread")]
async fn test_process_from_db_calculates_indicators() {
    let (_file, db_url) = create_mock_db().await;

    let df_result = FeatureExtractor::process_from_db(&db_url).await;
    assert!(df_result.is_ok(), "Failed to process data from DB");

    let df = df_result.unwrap();
    assert!(df.column("sma_5").is_ok());
    assert!(df.column("bband_upper").is_ok());
    assert!(df.column("bband_lower").is_ok());
    assert_eq!(df.height(), 5);
}

// Added flavor = "multi_thread" parameter
#[tokio::test(flavor = "multi_thread")]
async fn test_export_to_parquet_creates_file() {
    let (_file, db_url) = create_mock_db().await;
    let parquet_path = "test_db_output.parquet";

    let df = FeatureExtractor::process_from_db(&db_url).await.unwrap();
    let export_result = FeatureExtractor::export_to_parquet(&df, parquet_path);

    assert!(export_result.is_ok(), "Parquet export failed");
    assert!(std::path::Path::new(parquet_path).exists());

    let _ = fs::remove_file(parquet_path);
}
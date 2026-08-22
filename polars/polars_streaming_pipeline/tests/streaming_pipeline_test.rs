use polars_streaming_pipeline::{StreamIngestor, TickerData};
use serde_json::json;
use sqlx::SqlitePool;
use tempfile::NamedTempFile;

/// 1. Unit Test: Mocking Bitfinex JSON parsing logic
#[test]
fn test_bitfinex_json_parsing() {
    let mock_ws_msg = json!([
        238471,
        [
            65000.1, // Bid
            1.5,     // Bid Size
            65001.0, // Ask
            2.0,     // Ask Size
            1500.0,  // Daily change
            0.02,    // Daily change relative
            65000.5, // Last Price
            120.5,   // Volume
            65100.0, // High
            64800.0  // Low
        ]
    ])
    .to_string();

    let parsed: serde_json::Value = serde_json::from_str(&mock_ws_msg).unwrap();
    let data_arr = parsed.get(1).unwrap().as_array().unwrap();

    let ticker = TickerData {
        symbol: "tBTCUSD".to_string(),
        bid: data_arr[0].as_f64().unwrap(),
        ask: data_arr[2].as_f64().unwrap(),
        last_price: data_arr[6].as_f64().unwrap(),
        volume: data_arr[7].as_f64().unwrap(),
        timestamp: 1700000000000,
    };

    assert_eq!(ticker.symbol, "tBTCUSD");
    assert_eq!(ticker.bid, 65000.1);
    assert_eq!(ticker.ask, 65001.0);
    assert_eq!(ticker.last_price, 65000.5);
    assert_eq!(ticker.volume, 120.5);
}

/// 2. Integration Test: Test batch writing via Polars to SQLite
#[tokio::test]
async fn test_full_batch_ingestion_to_sqlite() {
    // Create a temporary SQLite file
    let tmp_file = NamedTempFile::new().unwrap();
    let db_url = format!("sqlite://{}?mode=rwc", tmp_file.path().to_str().unwrap());

    // Initialize the schema
    let _ingestor = StreamIngestor::new(&db_url, 10).await.unwrap();

    // Create a synthetic batch
    let mock_batch = vec![
        TickerData {
            symbol: "tBTCUSD".to_string(),
            bid: 100.0,
            ask: 101.0,
            last_price: 100.5,
            volume: 10.0,
            timestamp: 1000,
        },
        TickerData {
            symbol: "tBTCUSD".to_string(),
            bid: 102.0,
            ask: 103.0,
            last_price: 102.5,
            volume: 15.0,
            timestamp: 2000,
        },
    ];

    // Direct validation of batch write method
    let db_pool = SqlitePool::connect(&db_url).await.unwrap();
    
    // Apply persistence
    StreamIngestor::persist_batch_via_polars(&mock_batch, &db_pool)
        .await
        .expect("Persistence via Polars failed");

    // Validation SQL query over SQLite
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM market_tickers")
        .fetch_one(&db_pool)
        .await
        .unwrap();

    assert_eq!(count.0, 2, "Do SQLite měly být zapsány přesně 2 řádky");

    // Verification of entered data
    let row: (String, f64, f64) = sqlx::query_as(
        "SELECT symbol, last_price, volume FROM market_tickers WHERE timestamp = 2000",
    )
    .fetch_one(&db_pool)
    .await
    .unwrap();

    assert_eq!(row.0, "tBTCUSD");
    assert_eq!(row.1, 102.5);
    assert_eq!(row.2, 15.0);
}

/// 3. Edge Case Test: Empty batch must not fail or write
#[tokio::test]
async fn test_empty_batch_handling() {
    let tmp_file = NamedTempFile::new().unwrap();
    let db_url = format!("sqlite://{}?mode=rwc", tmp_file.path().to_str().unwrap());

    let _ingestor = StreamIngestor::new(&db_url, 10).await.unwrap();
    let db_pool = SqlitePool::connect(&db_url).await.unwrap();

    let empty_batch: Vec<TickerData> = vec![];

    let result = StreamIngestor::persist_batch_via_polars(&empty_batch, &db_pool).await;
    assert!(result.is_ok());

    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM market_tickers")
        .fetch_one(&db_pool)
        .await
        .unwrap();

    assert_eq!(count.0, 0);
}
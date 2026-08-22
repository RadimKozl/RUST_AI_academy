use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use sqlx::{sqlite::SqlitePoolOptions, SqlitePool};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerData {
    pub symbol: String,
    pub bid: f64,
    pub ask: f64,
    pub last_price: f64,
    pub volume: f64,
    pub timestamp: i64,
}

pub struct StreamIngestor {
    db_pool: SqlitePool,
    buffer_size: usize,
}

impl StreamIngestor {
    pub async fn new(db_url: &str, buffer_size: usize) -> Result<Self> {
        let db_pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(db_url)
            .await
            .context("Failed to connect to SQLite database")?;

        // Create a table with an index for the time series
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS market_tickers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                bid REAL NOT NULL,
                ask REAL NOT NULL,
                last_price REAL NOT NULL,
                volume REAL NOT NULL,
                timestamp INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tickers_ts ON market_tickers(timestamp);
            "#,
        )
        .execute(&db_pool)
        .await
        .context("Failed to initialize SQLite schema")?;

        Ok(Self {
            db_pool,
            buffer_size,
        })
    }

    pub async fn run_bitfinex_stream(&self, symbol: &str) -> Result<()> {
        let url = "wss://api-pub.bitfinex.com/ws/2";
        let (ws_stream, _) = connect_async(url)
            .await
            .context("Failed to connect to Bitfinex WebSocket")?;

        let (mut write, mut read) = ws_stream.split();

        // Bitfinex v2 subscribe payload
        let subscribe_msg = serde_json::json!({
            "event": "subscribe",
            "channel": "ticker",
            "symbol": symbol
        });

        write
            .send(Message::Text(subscribe_msg.to_string().into()))
            .await
            .context("Failed to send subscription request")?;

        let (tx, rx) = mpsc::channel::<TickerData>(1000);

        // Worker for aggregation and writing via Polars
        let pool_clone = self.db_pool.clone();
        let batch_limit = self.buffer_size;
        tokio::spawn(async move {
            Self::batch_writer_worker(rx, pool_clone, batch_limit).await;
        });

        // Event loop for processing incoming messages
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                        // Bitfinex sends Ticker in array format [CHANNEL_ID, [BID, BID_SIZE, ASK, ASK_SIZE, ... LAST_PRICE, VOLUME]]
                        if parsed.is_array() {
                            if let Some(data) = parsed.get(1) {
                                if data.is_array() && data.as_array().map_or(0, |a| a.len()) >= 10 {
                                    let arr = data.as_array().unwrap();
                                    let ticker = TickerData {
                                        symbol: symbol.to_string(),
                                        bid: arr[0].as_f64().unwrap_or(0.0),
                                        ask: arr[2].as_f64().unwrap_or(0.0),
                                        last_price: arr[6].as_f64().unwrap_or(0.0),
                                        volume: arr[7].as_f64().unwrap_or(0.0),
                                        timestamp: chrono::Utc::now().timestamp_millis(),
                                    };

                                    if tx.send(ticker).await.is_err() {
                                        break; // Receiver has been terminated
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(Message::Ping(_)) => {
                    let _ = write.send(Message::Pong(vec![].into())).await;
                }
                Err(e) => {
                    tracing::error!("WebSocket error: {:?}", e);
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    async fn batch_writer_worker(
        mut rx: mpsc::Receiver<TickerData>,
        pool: SqlitePool,
        batch_limit: usize,
    ) {
        let mut buffer: Vec<TickerData> = Vec::with_capacity(batch_limit);

        while rx.recv_many(&mut buffer, batch_limit).await > 0 {
            if let Err(e) = Self::persist_batch_via_polars(&buffer, &pool).await {
                tracing::error!("Failed to persist batch to SQLite: {:?}", e);
            }
            buffer.clear();
        }
    }

    pub async fn persist_batch_via_polars(batch: &[TickerData], pool: &SqlitePool) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        // 1. Converting the vector of structures to a Polars DataFrame
        let symbols: Vec<&str> = batch.iter().map(|t| t.symbol.as_str()).collect();
        let bids: Vec<f64> = batch.iter().map(|t| t.bid).collect();
        let asks: Vec<f64> = batch.iter().map(|t| t.ask).collect();
        let last_prices: Vec<f64> = batch.iter().map(|t| t.last_price).collect();
        let volumes: Vec<f64> = batch.iter().map(|t| t.volume).collect();
        let timestamps: Vec<i64> = batch.iter().map(|t| t.timestamp).collect();

        let df = df!(
            "symbol" => symbols,
            "bid" => bids,
            "ask" => asks,
            "last_price" => last_prices,
            "volume" => volumes,
            "timestamp" => timestamps
        )?;

        // 2. Extracting data from Polars for batch INSERT transactions in SQLite
        let mut tx = pool.begin().await?;

        let symbol_s = df.column("symbol")?.str()?;
        let bid_s = df.column("bid")?.f64()?;
        let ask_s = df.column("ask")?.f64()?;
        let last_price_s = df.column("last_price")?.f64()?;
        let volume_s = df.column("volume")?.f64()?;
        let timestamp_s = df.column("timestamp")?.i64()?;

        for i in 0..df.height() {
            sqlx::query(
                "INSERT INTO market_tickers (symbol, bid, ask, last_price, volume, timestamp) VALUES (?, ?, ?, ?, ?, ?)"
            )
            .bind(symbol_s.get(i).unwrap_or_default())
            .bind(bid_s.get(i).unwrap_or_default())
            .bind(ask_s.get(i).unwrap_or_default())
            .bind(last_price_s.get(i).unwrap_or_default())
            .bind(volume_s.get(i).unwrap_or_default())
            .bind(timestamp_s.get(i).unwrap_or_default())
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        println!("💾 Successfully persisted batch of {} items to SQLite", df.height());

        Ok(())
    }
}
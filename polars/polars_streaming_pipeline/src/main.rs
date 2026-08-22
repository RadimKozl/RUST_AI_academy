use anyhow::Result;
use polars_streaming_pipeline::StreamIngestor;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let db_url = "sqlite://market_data.db?mode=rwc";
    let batch_size = 10; // Write every 10 ticks

    println!("⚡ Initializing Real-time Streaming Ingestor...");
    let ingestor = StreamIngestor::new(db_url, batch_size).await?;

    println!("📡 Connecting to Bitfinex WebSocket (tBTCUSD)...");
    ingestor.run_bitfinex_stream("tBTCUSD").await?;

    Ok(())
}

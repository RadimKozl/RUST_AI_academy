use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::time::timeout;
use tracing::{info, warn};

#[derive(Deserialize)]
struct JsonRpcWireResponse {
    id: Option<Value>,
    result: Option<Value>,
    error: Option<JsonRpcWireError>,
}

#[derive(Deserialize)]
struct JsonRpcWireError {
    code: i64,
    message: String,
}

pub fn parse_sse_event_block(block: &str) -> Option<(String, String)> {
    let mut event_name = "message".to_string();
    let mut data_lines = Vec::new();

    for line in block.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("event:") {
            event_name = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim_start().to_string());
        }
    }

    if data_lines.is_empty() {
        None
    } else {
        Some((event_name, data_lines.join("\n")))
    }
}

pub fn origin_from_url(url: &str) -> String {
    if let Some(scheme_sep) = url.find("://") {
        let after_scheme = &url[scheme_sep + 3..];
        let end_host = after_scheme.find('/').unwrap_or(after_scheme.len());
        return url[..scheme_sep + 3 + end_host].to_string();
    }
    "http://localhost:8000".to_string()
}

pub fn resolve_endpoint_url(base_origin: &str, endpoint_data: &str) -> String {
    if endpoint_data.starts_with("http://") || endpoint_data.starts_with("https://") {
        endpoint_data.to_string()
    } else {
        format!(
            "{}/{}",
            base_origin.trim_end_matches('/'),
            endpoint_data.trim_start_matches('/')
        )
    }
}

async fn wait_for_endpoint<S, E>(
    byte_stream: &mut S,
    buffer: &mut String,
    base_origin: &str,
) -> Result<String>
where
    S: futures_util::Stream<Item = std::result::Result<bytes::Bytes, E>> + Unpin,
    E: std::error::Error + Send + Sync + 'static,
{
    while let Some(chunk) = byte_stream.next().await {
        let bytes = chunk.context("Error reading SSE chunk")?;
        buffer.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(pos) = buffer.find("\n\n") {
            let raw_block = buffer[..pos].to_string();
            buffer.drain(..pos + 2);

            if let Some((event_name, data)) = parse_sse_event_block(&raw_block) {
                if event_name == "endpoint" {
                    return Ok(resolve_endpoint_url(base_origin, &data));
                }
            }
        }
    }

    Err(anyhow!("SSE stream closed before receiving endpoint event"))
}

async fn wait_for_json_rpc_response<S, E>(
    byte_stream: &mut S,
    buffer: &mut String,
    expected_id: &Value,
) -> Result<Value>
where
    S: futures_util::Stream<Item = std::result::Result<bytes::Bytes, E>> + Unpin,
    E: std::error::Error + Send + Sync + 'static,
{
    while let Some(chunk) = byte_stream.next().await {
        let bytes = chunk.context("Error reading SSE response chunk")?;
        buffer.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(pos) = buffer.find("\n\n") {
            let raw_block = buffer[..pos].to_string();
            buffer.drain(..pos + 2);

            if let Some((event_name, data)) = parse_sse_event_block(&raw_block) {
                if event_name == "message" {
                    let parsed: JsonRpcWireResponse = serde_json::from_str(&data)
                        .context("Invalid JSON-RPC payload in SSE message")?;
                    if parsed.id.as_ref() == Some(expected_id) {
                        if let Some(err) = parsed.error {
                            bail!("MCP error ({}): {}", err.code, err.message);
                        }
                        return Ok(parsed.result.unwrap_or(Value::Null));
                    }
                }
            }
        }
    }

    Err(anyhow!(
        "SSE stream closed before receiving response for id {expected_id}"
    ))
}

pub async fn fetch_graph_statistics_over_sse(sse_url: &str) -> Result<Value> {
    info!(%sse_url, "Connecting to MCP SSE server");

    let http_client = Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .context("Failed to build HTTP client")?;

    let response = http_client
        .get(sse_url)
        .header("Accept", "text/event-stream")
        .send()
        .await
        .context("Failed to connect to SSE endpoint")?;

    if !response.status().is_success() {
        bail!("Unexpected HTTP status from /sse: {}", response.status());
    }

    let mut byte_stream = response.bytes_stream();
    let mut buffer = String::new();
    let base_origin = origin_from_url(sse_url);

    let post_url = timeout(
        Duration::from_secs(10),
        wait_for_endpoint(&mut byte_stream, &mut buffer, &base_origin),
    )
    .await
    .context("Timed out waiting for SSE endpoint event")??;

    info!(%post_url, "Connected! Sending MCP initialize handshake");

    let init_id = json!(1);
    let init_req = json!({
        "jsonrpc": "2.0",
        "id": init_id,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {
                "name": "rust-neo4j-sse-client",
                "version": "0.1.0"
            }
        }
    });

    let init_status = http_client
        .post(&post_url)
        .json(&init_req)
        .send()
        .await
        .context("Failed to send initialize request")?
        .status();

    if !init_status.is_success() {
        bail!("initialize request failed with status {init_status}");
    }

    let init_resp = timeout(
        Duration::from_secs(10),
        wait_for_json_rpc_response(&mut byte_stream, &mut buffer, &init_id),
    )
    .await
    .context("Timed out waiting for initialize response")??;

    info!(server_info = %init_resp, "MCP session initialized");

    let initialized_notif = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    });
    let _ = http_client.post(&post_url).json(&initialized_notif).send().await;

    info!("Launching the graph_statistics tool...");
    let call_id = json!(2);
    let call_req = json!({
        "jsonrpc": "2.0",
        "id": call_id,
        "method": "tools/call",
        "params": {
            "name": "graph_statistics",
            "arguments": {}
        }
    });

    let call_status = http_client
        .post(&post_url)
        .json(&call_req)
        .send()
        .await
        .context("Failed to send tools/call request")?
        .status();

    if !call_status.is_success() {
        bail!("tools/call request failed with status {call_status}");
    }

    let tool_result = timeout(
        Duration::from_secs(10),
        wait_for_json_rpc_response(&mut byte_stream, &mut buffer, &call_id),
    )
    .await
    .context("Timed out waiting for graph_statistics response")??;

    Ok(tool_result)
}

pub fn format_tool_result(tool_result: &Value) -> Result<String> {
    let mut output = Vec::new();

    if let Some(items) = tool_result.get("content").and_then(Value::as_array) {
        for item in items {
            if let Some(text) = item.get("text").and_then(Value::as_str) {
                if let Ok(parsed_json) = serde_json::from_str::<Value>(text) {
                    output.push(serde_json::to_string_pretty(&parsed_json)?);
                } else {
                    output.push(text.to_string());
                }
            } else {
                warn!("Received non-text content item: {item}");
                output.push(item.to_string());
            }
        }
    } else {
        output.push(serde_json::to_string_pretty(tool_result)?);
    }

    Ok(output.join("\n"))
}
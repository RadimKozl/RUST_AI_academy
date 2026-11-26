use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use futures_util::{Stream, StreamExt};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::time::timeout;
use tracing::info;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphRagClientSummary {
    pub tools: Vec<String>,
    pub schema_output: Option<String>,
    pub prompts: Vec<String>,
    pub system_instructions_output: Option<String>,
    pub resource_output: Option<String>,
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
    S: Stream<Item = std::result::Result<bytes::Bytes, E>> + Unpin,
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
    S: Stream<Item = std::result::Result<bytes::Bytes, E>> + Unpin,
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

async fn rpc_call<S, E>(
    client: &Client,
    post_url: &str,
    byte_stream: &mut S,
    buffer: &mut String,
    id: i64,
    method: &str,
    params: Value,
) -> Result<Value>
where
    S: Stream<Item = std::result::Result<bytes::Bytes, E>> + Unpin,
    E: std::error::Error + Send + Sync + 'static,
{
    let req_id = json!(id);
    let body = json!({
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params
    });

    let status = client
        .post(post_url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("Failed to send {method}"))?
        .status();

    if !status.is_success() {
        bail!("{method} failed with status {status}");
    }

    timeout(
        Duration::from_secs(10),
        wait_for_json_rpc_response(byte_stream, buffer, &req_id),
    )
    .await
    .with_context(|| format!("Timed out waiting for {method} response"))?
}

fn first_content_text(result: &Value) -> Option<String> {
    result
        .get("content")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(|item| item.get("text"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn first_prompt_text(result: &Value) -> Option<String> {
    result
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(|msg| msg.get("content"))
        .and_then(|c| c.get("text"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn first_resource_text(result: &Value) -> Option<String> {
    result
        .get("contents")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(|item| item.get("text"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

pub async fn run_graphrag_client_checks(sse_url: &str) -> Result<GraphRagClientSummary> {
    info!(%sse_url, "Connecting to MCP Server via SSE");

    let client = Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .context("Failed to build HTTP client")?;

    let response = client
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
    .context("Timed out waiting for endpoint event")??;

    let _init = rpc_call(
        &client,
        &post_url,
        &mut byte_stream,
        &mut buffer,
        1,
        "initialize",
        json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {
                "name": "universal-neo4j-mcp-client",
                "version": "0.1.0"
            }
        }),
    )
    .await?;

    let _ = client
        .post(&post_url)
        .json(&json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }))
        .send()
        .await;

    let tools_val = rpc_call(
        &client,
        &post_url,
        &mut byte_stream,
        &mut buffer,
        2,
        "tools/list",
        json!({}),
    )
    .await?;

    let tools = tools_val
        .get("tools")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|t| t.get("name").and_then(Value::as_str).map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default();

    let schema_output = match rpc_call(
        &client,
        &post_url,
        &mut byte_stream,
        &mut buffer,
        3,
        "tools/call",
        json!({ "name": "get_graph_schema", "arguments": {} }),
    )
    .await
    {
        Ok(res) => first_content_text(&res),
        Err(e) => Some(format!("Error calling tool: {e}")),
    };

    let (prompts, system_instructions_output) = match rpc_call(
        &client,
        &post_url,
        &mut byte_stream,
        &mut buffer,
        4,
        "prompts/list",
        json!({}),
    )
    .await
    {
        Ok(plist) => {
            let names = plist
                .get("prompts")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|p| p.get("name").and_then(Value::as_str).map(ToString::to_string))
                        .collect()
                })
                .unwrap_or_default();

            let prompt_txt = rpc_call(
                &client,
                &post_url,
                &mut byte_stream,
                &mut buffer,
                5,
                "prompts/get",
                json!({ "name": "system_instructions", "arguments": {} }),
            )
            .await
            .ok()
            .and_then(|r| first_prompt_text(&r));

            (names, prompt_txt)
        }
        Err(e) => (Vec::new(), Some(format!("Error working with prompts: {e}"))),
    };

    let resource_uri =
        std::env::var("MCP_RESOURCE_URI").unwrap_or_else(|_| "node://Movie/11443".to_string());
    let resource_output = match rpc_call(
        &client,
        &post_url,
        &mut byte_stream,
        &mut buffer,
        6,
        "resources/read",
        json!({ "uri": resource_uri }),
    )
    .await
    {
        Ok(res) => first_resource_text(&res),
        Err(e) => Some(format!("Resource note ({resource_uri}): {e}")),
    };

    Ok(GraphRagClientSummary {
        tools,
        schema_output,
        prompts,
        system_instructions_output,
        resource_output,
    })
}
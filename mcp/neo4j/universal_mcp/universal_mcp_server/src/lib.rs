use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use neo4rs::{
    query, BoltBoolean, BoltFloat, BoltMap, BoltNode, BoltRelation, BoltType, ConfigBuilder, Graph,
    Row,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use thiserror::Error;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tracing::{error, info, warn};
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("Read-only mode is enabled; write operations are forbidden")]
    ReadOnlyViolation,
    #[error("Custom tool '{0}' not found in configuration")]
    UnknownCustomTool(String),
    #[error("Unknown resource URI '{0}'")]
    UnknownResource(String),
    #[error("Neo4j error: {0}")]
    Neo4j(#[from] neo4rs::Error),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Neo4jConfig {
    #[serde(default = "default_neo4j_uri")]
    pub uri: String,
    #[serde(default = "default_neo4j_user")]
    pub username: String,
    #[serde(default = "default_neo4j_password")]
    pub password: String,
    #[serde(default = "default_neo4j_database")]
    pub database: String,
}

fn default_neo4j_uri() -> String {
    "bolt://localhost:7687".to_string()
}

fn default_neo4j_user() -> String {
    "neo4j".to_string()
}

fn default_neo4j_password() -> String {
    "password".to_string()
}

fn default_neo4j_database() -> String {
    "neo4j".to_string()
}

impl Default for Neo4jConfig {
    fn default() -> Self {
        Self {
            uri: default_neo4j_uri(),
            username: default_neo4j_user(),
            password: default_neo4j_password(),
            database: default_neo4j_database(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResourceTemplateConfig {
    pub uri_pattern: String,
    pub description: String,
    pub cypher_query: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CustomToolConfig {
    pub name: String,
    pub description: String,
    pub cypher_query: String,
    #[serde(default)]
    pub parameters: Vec<String>,
    #[serde(default)]
    pub write: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptTemplateConfig {
    pub name: String,
    pub description: String,
    pub template: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    #[serde(default = "default_transport")]
    pub mcp_transport: String,
    #[serde(default)]
    pub read_only: bool,
    #[serde(default)]
    pub neo4j: Neo4jConfig,
    #[serde(default = "default_agent_instructions")]
    pub agent_instructions: String,
    #[serde(default)]
    pub resources: Vec<ResourceTemplateConfig>,
    #[serde(default)]
    pub custom_tools: Vec<CustomToolConfig>,
    #[serde(default)]
    pub prompts: Vec<PromptTemplateConfig>,
}

fn default_transport() -> String {
    "sse".to_string()
}

fn default_agent_instructions() -> String {
    "Jsi expertní databázový asistent pro Neo4j. Vždy analyzuj strukturu grafu před složitějšími dotazy a vracej výsledky v přehledném Markdownu.".to_string()
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            mcp_transport: default_transport(),
            read_only: false,
            neo4j: Neo4jConfig::default(),
            agent_instructions: default_agent_instructions(),
            resources: Vec::new(),
            custom_tools: Vec::new(),
            prompts: Vec::new(),
        }
    }
}

pub fn expand_env_placeholders<F>(input: &str, lookup: &F) -> String
where
    F: Fn(&str) -> Option<String>,
{
    let mut out = String::with_capacity(input.len());
    let mut rest = input;

    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after_open = &rest[start + 2..];
        let Some(end) = after_open.find('}') else {
            out.push_str(&rest[start..]);
            return out;
        };

        let inner = &after_open[..end];
        let (var_name, default_val) = if let Some(sep_idx) = inner.find(":-") {
            (&inner[..sep_idx], Some(&inner[sep_idx + 2..]))
        } else {
            (inner, None)
        };

        if let Some(val) = lookup(var_name).filter(|s| !s.is_empty()) {
            out.push_str(&val);
        } else if let Some(def) = default_val {
            out.push_str(def);
        }

        rest = &after_open[end + 1..];
    }

    out.push_str(rest);
    out
}

pub fn expand_json_env_vars<F>(value: Value, lookup: &F) -> Value
where
    F: Fn(&str) -> Option<String>,
{
    match value {
        Value::String(s) => Value::String(expand_env_placeholders(&s, lookup)),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|item| expand_json_env_vars(item, lookup))
                .collect(),
        ),
        Value::Object(map) => {
            let mut new_map = Map::with_capacity(map.len());
            for (k, v) in map {
                new_map.insert(k, expand_json_env_vars(v, lookup));
            }
            Value::Object(new_map)
        }
        other => other,
    }
}

pub fn resolve_config_path(base_dir: &Path, config_path_env: &str) -> PathBuf {
    let candidate = Path::new(config_path_env);
    if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        let file_name = candidate
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("config.json");
        base_dir.join(file_name)
    }
}

pub fn load_server_config_from_path(path: &Path) -> Result<ServerConfig> {
    if !path.exists() {
        warn!(config_path = %path.display(), "Config file not found, using defaults");
        return Ok(ServerConfig::default());
    }

    let raw_text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file {}", path.display()))?;
    let raw_json: Value = serde_json::from_str(&raw_text)
        .with_context(|| format!("Invalid JSON in config file {}", path.display()))?;

    let env_lookup = |key: &str| std::env::var(key).ok();
    let expanded = expand_json_env_vars(raw_json, &env_lookup);

    serde_json::from_value(expanded)
        .with_context(|| format!("Failed to deserialize ServerConfig from {}", path.display()))
}

pub fn determine_read_only_mode(cli_read_only: bool, config_read_only: bool) -> bool {
    let env_read_only = std::env::var("READ_ONLY")
        .map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes"))
        .unwrap_or(false);

    cli_read_only || env_read_only || config_read_only
}

fn json_value_to_bolt(val: &Value) -> BoltType {
    match val {
        Value::Null => BoltType::String("".into()),
        Value::Bool(b) => BoltType::Boolean(BoltBoolean::new(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                BoltType::Integer(i.into())
            } else if let Some(u) = n.as_u64() {
                BoltType::Integer((u as i64).into())
            } else if let Some(f) = n.as_f64() {
                BoltType::Float(BoltFloat::new(f))
            } else {
                BoltType::String(n.to_string().into())
            }
        }
        Value::String(s) => BoltType::String(s.as_str().into()),
        Value::Array(items) => {
            let mut list = neo4rs::BoltList::new();
            for item in items {
                list.push(json_value_to_bolt(item));
            }
            BoltType::List(list)
        }
        Value::Object(map) => {
            let mut bmap = BoltMap::new();
            for (k, v) in map {
                bmap.put(k.as_str().into(), json_value_to_bolt(v));
            }
            BoltType::Map(bmap)
        }
    }
}

fn bolt_to_json_value(bolt: BoltType) -> Value {
    match bolt {
        BoltType::Boolean(b) => Value::Bool(b.value),
        BoltType::Integer(i) => json!(i.value),
        BoltType::Float(f) => json!(f.value),
        BoltType::String(s) => Value::String(s.value),
        BoltType::List(list) => {
            Value::Array(list.value.into_iter().map(bolt_to_json_value).collect())
        }
        BoltType::Map(bmap) => {
            let mut obj = Map::new();
            for (k, v) in bmap.value {
                obj.insert(k.value, bolt_to_json_value(v));
            }
            Value::Object(obj)
        }
        BoltType::Node(node) => bolt_node_to_json_value(&node),
        BoltType::Relation(rel) => bolt_relation_to_json_value(&rel),
        BoltType::UnboundedRelation(urel) => json!({
            "id": urel.id.value,
            "type": urel.typ.value,
            "properties": bolt_to_json_value(BoltType::Map(urel.properties))
        }),
        BoltType::Point2D(pt) => json!({
            "srid": pt.sr_id.value,
            "x": pt.x.value,
            "y": pt.y.value
        }),
        BoltType::Point3D(pt) => json!({
            "srid": pt.sr_id.value,
            "x": pt.x.value,
            "y": pt.y.value,
            "z": pt.z.value
        }),
        other => Value::String(format!("{other:?}")),
    }
}

fn bolt_node_to_json_value(node: &BoltNode) -> Value {
    let labels: Vec<String> = node
        .labels
        .value
        .iter()
        .map(|item| match bolt_to_json_value(item.clone()) {
            Value::String(s) => s,
            other => other.to_string().trim_matches('"').to_string(),
        })
        .collect();
    let mut props = Map::new();
    for (key, val) in &node.properties.value {
        props.insert(key.value.clone(), bolt_to_json_value(val.clone()));
    }
    json!({
        "id": node.id.value,
        "labels": labels,
        "properties": props
    })
}

fn bolt_relation_to_json_value(rel: &BoltRelation) -> Value {
    let mut props = Map::new();
    for (key, val) in &rel.properties.value {
        props.insert(key.value.clone(), bolt_to_json_value(val.clone()));
    }
    json!({
        "id": rel.id.value,
        "startNodeId": rel.start_node_id.value,
        "endNodeId": rel.end_node_id.value,
        "type": rel.typ.value,
        "properties": props
    })
}

fn row_to_json_value(row: &Row) -> Value {
    let mut map = Map::new();
    if let Ok(node) = row.get::<BoltNode>("n") {
        map.insert("n".to_string(), bolt_node_to_json_value(&node));
    }
    if let Ok(props) = row.get::<BoltMap>("props") {
        map.insert("props".to_string(), bolt_to_json_value(BoltType::Map(props)));
    }
    if let Ok(labels) = row.get::<Vec<String>>("labels") {
        map.insert("labels".to_string(), json!(labels));
    }
    if let Ok(label) = row.get::<String>("label") {
        map.insert("label".to_string(), Value::String(label));
    }
    if let Ok(rel_type) = row.get::<String>("relationshipType") {
        map.insert("relationshipType".to_string(), Value::String(rel_type));
    }
    if let Ok(rel_type) = row.get::<String>("rel_type") {
        map.insert("rel_type".to_string(), Value::String(rel_type));
    }
    if let Ok(target_labels) = row.get::<Vec<String>>("target_labels") {
        map.insert("target_labels".to_string(), json!(target_labels));
    }
    if let Ok(target_name) = row.get::<String>("target_name") {
        map.insert("target_name".to_string(), Value::String(target_name));
    }
    if let Ok(rel_props) = row.get::<BoltMap>("rel_props") {
        map.insert(
            "rel_props".to_string(),
            bolt_to_json_value(BoltType::Map(rel_props)),
        );
    }
    if let Ok(nodes) = row.get::<i64>("nodes") {
        map.insert("nodes".to_string(), json!(nodes));
    }
    if let Ok(rels) = row.get::<i64>("relationships") {
        map.insert("relationships".to_string(), json!(rels));
    }
    if let Ok(data_map) = row.get::<BoltMap>("data") {
        map.insert("data".to_string(), bolt_to_json_value(BoltType::Map(data_map)));
    }
    Value::Object(map)
}

type SessionSenders = Arc<Mutex<HashMap<String, mpsc::Sender<String>>>>;

#[derive(Clone)]
pub struct GraphRagAppState {
    pub graph: Arc<Graph>,
    pub database: String,
    pub config: ServerConfig,
    pub read_only: bool,
    pub sessions: SessionSenders,
}

impl GraphRagAppState {
    pub fn new(graph: Arc<Graph>, config: ServerConfig, read_only: bool) -> Self {
        let database = config.neo4j.database.clone();
        Self {
            graph,
            database,
            config,
            read_only,
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn execute_read_cypher(
        &self,
        cypher_str: &str,
        params: Option<HashMap<String, Value>>,
    ) -> Result<Vec<Value>, ServerError> {
        let mut q = query(cypher_str);
        if let Some(p) = params {
            for (k, v) in p {
                q = q.param(&k, json_value_to_bolt(&v));
            }
        }

        let mut rows = self.graph.execute(q).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(row_to_json_value(&row));
        }
        Ok(out)
    }

    pub async fn execute_write_cypher(
        &self,
        cypher_str: &str,
        params: Option<HashMap<String, Value>>,
    ) -> Result<String, ServerError> {
        if self.read_only {
            return Err(ServerError::ReadOnlyViolation);
        }

        let mut q = query(cypher_str);
        if let Some(p) = params {
            for (k, v) in p {
                q = q.param(&k, json_value_to_bolt(&v));
            }
        }

        self.graph.run(q).await?;
        Ok("Zápis dokončen úspěšně.".to_string())
    }

    pub async fn get_graph_schema(&self) -> Result<Value, ServerError> {
        let labels = self
            .execute_read_cypher("CALL db.labels() YIELD label RETURN label", None)
            .await?;
        let rels = self
            .execute_read_cypher(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType",
                None,
            )
            .await?;
        let stats = self
            .execute_read_cypher(
                "RETURN COUNT {()} AS nodes, COUNT {()-[]-()} AS relationships",
                None,
            )
            .await?;

        Ok(json!({
            "labels": labels,
            "relationship_types": rels,
            "statistics": stats.into_iter().next().unwrap_or_else(|| json!({"nodes": 0, "relationships": 0}))
        }))
    }

    pub async fn run_configured_tool(
        &self,
        tool_name: &str,
        params: HashMap<String, Value>,
    ) -> Result<Vec<Value>, ServerError> {
        let cfg = self
            .config
            .custom_tools
            .iter()
            .find(|t| t.name == tool_name)
            .ok_or_else(|| ServerError::UnknownCustomTool(tool_name.to_string()))?;

        if cfg.write && self.read_only {
            return Err(ServerError::ReadOnlyViolation);
        }

        self.execute_read_cypher(&cfg.cypher_query, Some(params)).await
    }

    pub async fn read_node_resource(&self, label: &str, id: &str) -> Result<String, ServerError> {
        let cypher = "MATCH (n) WHERE $label IN labels(n) AND (toString(n.id) = $id OR toString(n.tmdbId) = $id OR toString(n.movie_tmdbId) = $id OR toString(n.movieId) = $id OR toString(n.uid) = $id OR toString(n.name) = $id OR toString(n.title) = $id) RETURN properties(n) AS props, labels(n) AS labels";
        let mut params = HashMap::new();
        params.insert("label".to_string(), Value::String(label.to_string()));
        params.insert("id".to_string(), Value::String(id.to_string()));

        let rows = self.execute_read_cypher(cypher, Some(params)).await?;
        if rows.is_empty() {
            return Ok(format!("Uzel s labelem '{label}' a ID '{id}' nebyl nalezen."));
        }
        Ok(serde_json::to_string_pretty(&rows)?)
    }

    pub async fn read_node_relationships_resource(
        &self,
        label: &str,
        id: &str,
    ) -> Result<String, ServerError> {
        let cypher = "MATCH (n)-[r]-(target) WHERE $label IN labels(n) AND (toString(n.id) = $id OR toString(n.tmdbId) = $id OR toString(n.movie_tmdbId) = $id OR toString(n.movieId) = $id OR toString(n.uid) = $id OR toString(n.name) = $id OR toString(n.title) = $id) RETURN type(r) AS rel_type, labels(target) AS target_labels, coalesce(target.name, target.title, target.id) AS target_name, properties(r) AS rel_props LIMIT 100";
        let mut params = HashMap::new();
        params.insert("label".to_string(), Value::String(label.to_string()));
        params.insert("id".to_string(), Value::String(id.to_string()));

        let rows = self.execute_read_cypher(cypher, Some(params)).await?;
        if rows.is_empty() {
            return Ok(format!(
                "Pro uzel s labelem '{label}' a ID '{id}' nebyly nalezeny žádné vazby."
            ));
        }
        Ok(serde_json::to_string_pretty(&rows)?)
    }

    pub fn resolve_prompt(&self, prompt_name: &str) -> String {
        if prompt_name == "system_instructions" {
            return self.config.agent_instructions.clone();
        }

        if let Some(found) = self.config.prompts.iter().find(|p| p.name == prompt_name) {
            return found.template.clone();
        }

        format!("Prompt '{prompt_name}' nebyl v config.json nalezen.")
    }
}

#[derive(Deserialize)]
struct MessageQuery {
    #[serde(rename = "sessionId")]
    session_id: Option<String>,
}

#[derive(Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: Option<String>,
    pub id: Option<Value>,
    pub method: String,
    pub params: Option<Value>,
}

#[derive(Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Serialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

fn rpc_ok(id: Option<Value>, result: Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: Some(result),
        error: None,
    }
}

fn rpc_err(id: Option<Value>, code: i64, message: impl Into<String>) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(JsonRpcError {
            code,
            message: message.into(),
        }),
    }
}

pub async fn dispatch_json_rpc(
    state: &GraphRagAppState,
    req: JsonRpcRequest,
) -> Option<JsonRpcResponse> {
    let id = req.id.clone();

    match req.method.as_str() {
        "initialize" => Some(rpc_ok(
            id,
            json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "serverInfo": {
                    "name": "Universal Neo4j GraphRAG Server",
                    "version": env!("CARGO_PKG_VERSION")
                },
                "instructions": state.config.agent_instructions
            }),
        )),
        "notifications/initialized" => None,
        "tools/list" => {
            let mut tools = vec![
                json!({
                    "name": "read_cypher",
                    "description": "Execute a read-only Cypher query with optional parameters.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" },
                            "parameters": { "type": "object" }
                        },
                        "required": ["query"]
                    }
                }),
                json!({
                    "name": "write_cypher",
                    "description": "Execute a write Cypher query (respects read_only mode).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" },
                            "parameters": { "type": "object" }
                        },
                        "required": ["query"]
                    }
                }),
                json!({
                    "name": "get_graph_schema",
                    "description": "Returns node labels, relationship types, and node/relationship counts.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }),
                json!({
                    "name": "run_configured_tool",
                    "description": "Executes a pre-defined Cypher tool from config.json.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "tool_name": { "type": "string" },
                            "params": { "type": "object" }
                        },
                        "required": ["tool_name"]
                    }
                }),
            ];

            for ct in &state.config.custom_tools {
                tools.push(json!({
                    "name": ct.name,
                    "description": ct.description,
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }));
            }

            Some(rpc_ok(id, json!({ "tools": tools })))
        }
        "tools/call" => {
            let params = req.params.as_ref();
            let tool_name = params
                .and_then(|p| p.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let args = params
                .and_then(|p| p.get("arguments"))
                .and_then(Value::as_object);

            match tool_name {
                "get_graph_schema" => match state.get_graph_schema().await {
                    Ok(schema_val) => Some(rpc_ok(
                        id,
                        json!({
                            "content": [{ "type": "text", "text": schema_val.to_string() }],
                            "isError": false
                        }),
                    )),
                    Err(e) => Some(rpc_err(id, -32603, e.to_string())),
                },
                "read_cypher" => {
                    let cypher = args
                        .and_then(|m| m.get("query"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let cypher_params = args
                        .and_then(|m| m.get("parameters"))
                        .and_then(Value::as_object)
                        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect());

                    match state.execute_read_cypher(cypher, cypher_params).await {
                        Ok(rows) => Some(rpc_ok(
                            id,
                            json!({
                                "content": [{ "type": "text", "text": json!(rows).to_string() }],
                                "isError": false
                            }),
                        )),
                        Err(e) => Some(rpc_err(id, -32603, e.to_string())),
                    }
                }
                "write_cypher" => {
                    let cypher = args
                        .and_then(|m| m.get("query"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let cypher_params = args
                        .and_then(|m| m.get("parameters"))
                        .and_then(Value::as_object)
                        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect());

                    match state.execute_write_cypher(cypher, cypher_params).await {
                        Ok(msg) => Some(rpc_ok(
                            id,
                            json!({
                                "content": [{ "type": "text", "text": msg }],
                                "isError": false
                            }),
                        )),
                        Err(e) => Some(rpc_err(id, -32603, e.to_string())),
                    }
                }
                "run_configured_tool" => {
                    let target = args
                        .and_then(|m| m.get("tool_name"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let call_params = args
                        .and_then(|m| m.get("params"))
                        .and_then(Value::as_object)
                        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                        .unwrap_or_default();

                    match state.run_configured_tool(target, call_params).await {
                        Ok(rows) => Some(rpc_ok(
                            id,
                            json!({
                                "content": [{ "type": "text", "text": json!(rows).to_string() }],
                                "isError": false
                            }),
                        )),
                        Err(e) => Some(rpc_err(id, -32603, e.to_string())),
                    }
                }
                other => Some(rpc_err(id, -32601, format!("Unknown tool: {other}"))),
            }
        }
        "prompts/list" => {
            let mut prompts = vec![
                json!({
                    "name": "system_instructions",
                    "description": "Main system instructions for Neo4j assistant"
                }),
                json!({
                    "name": "get_prompt_by_name",
                    "description": "Lookup configured prompt template by name"
                }),
            ];
            for p in &state.config.prompts {
                prompts.push(json!({
                    "name": p.name,
                    "description": p.description
                }));
            }
            Some(rpc_ok(id, json!({ "prompts": prompts })))
        }
        "prompts/get" => {
            let prompt_name = req
                .params
                .as_ref()
                .and_then(|p| p.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let text = state.resolve_prompt(prompt_name);
            Some(rpc_ok(
                id,
                json!({
                    "description": prompt_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": text
                            }
                        }
                    ]
                }),
            ))
        }
        "resources/list" => {
            let templates: Vec<Value> = state
                .config
                .resources
                .iter()
                .map(|r| {
                    json!({
                        "uriTemplate": r.uri_pattern,
                        "name": r.uri_pattern,
                        "description": r.description,
                        "mimeType": "application/json"
                    })
                })
                .collect();

            Some(rpc_ok(
                id,
                json!({
                    "resources": [],
                    "resourceTemplates": templates
                }),
            ))
        }
        "resources/read" => {
            let uri = req
                .params
                .as_ref()
                .and_then(|p| p.get("uri"))
                .and_then(Value::as_str)
                .unwrap_or("");

            let res_text = if let Some(rest) = uri.strip_prefix("node://") {
                let parts: Vec<&str> = rest.split('/').collect();
                match parts.as_slice() {
                    [label, id_val] if !label.is_empty() && !id_val.is_empty() => {
                        state.read_node_resource(label, id_val).await
                    }
                    [label, id_val, "relationships"] if !label.is_empty() && !id_val.is_empty() => {
                        state.read_node_relationships_resource(label, id_val).await
                    }
                    _ => Err(ServerError::UnknownResource(uri.to_string())),
                }
            } else {
                Err(ServerError::UnknownResource(uri.to_string()))
            };

            match res_text {
                Ok(text) => Some(rpc_ok(
                    id,
                    json!({
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": text
                            }
                        ]
                    }),
                )),
                Err(e) => Some(rpc_err(id, -32602, e.to_string())),
            }
        }
        other => id.map(|req_id| rpc_err(Some(req_id), -32601, format!("Method not found: {other}"))),
    }
}

async fn sse_handler(
    State(state): State<GraphRagAppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let session_id = Uuid::new_v4().to_string();
    let (tx, rx) = mpsc::channel::<String>(32);

    {
        let mut guard = state.sessions.lock().await;
        guard.insert(session_id.clone(), tx);
    }

    let endpoint_uri = format!("/messages?sessionId={session_id}");
    let initial_event = Event::default().event("endpoint").data(endpoint_uri);
    let stream = tokio_stream::once(Ok(initial_event))
        .chain(ReceiverStream::new(rx).map(|msg| Ok(Event::default().event("message").data(msg))));

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

async fn messages_handler(
    State(state): State<GraphRagAppState>,
    Query(query): Query<MessageQuery>,
    Json(payload): Json<JsonRpcRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let session_id = query
        .session_id
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "Missing sessionId query parameter".into()))?;

    let sender = {
        let guard = state.sessions.lock().await;
        guard.get(&session_id).cloned()
    }
    .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Unknown sessionId: {session_id}")))?;

    if payload.jsonrpc.as_deref().unwrap_or("2.0") != "2.0" {
        return Err((StatusCode::BAD_REQUEST, "Invalid jsonrpc version".into()));
    }

    if let Some(response) = dispatch_json_rpc(&state, payload).await {
        let serialized = serde_json::to_string(&response).map_err(|err| {
            error!(error = %err, "Failed to serialize JSON-RPC response");
            (StatusCode::INTERNAL_SERVER_ERROR, "Serialization failed".into())
        })?;

        sender.send(serialized).await.map_err(|_| {
            (
                StatusCode::GONE,
                "SSE client disconnected before receiving response".into(),
            )
        })?;
    }

    Ok(StatusCode::ACCEPTED)
}

pub fn build_router(state: GraphRagAppState) -> Router {
    Router::new()
        .route("/sse", get(sse_handler))
        .route("/messages", post(messages_handler))
        .with_state(state)
}

pub async fn run_server_from_config(config_path: &Path, cli_read_only: bool) -> Result<()> {
    let cfg = load_server_config_from_path(config_path)?;
    let read_only = determine_read_only_mode(cli_read_only, cfg.read_only);

    let neo4j_cfg = ConfigBuilder::new()
        .uri(&cfg.neo4j.uri)
        .user(&cfg.neo4j.username)
        .password(&cfg.neo4j.password)
        .db(cfg.neo4j.database.as_str())
        .build()
        .context("Invalid Neo4j driver configuration")?;

    let graph = Arc::new(
        Graph::connect(neo4j_cfg)
            .await
            .context("Failed to connect to Neo4j")?,
    );

    graph
        .run(query("RETURN 1"))
        .await
        .context("Neo4j connectivity verification failed")?;

    let state = GraphRagAppState::new(graph, cfg, read_only);
    let transport = state.config.mcp_transport.to_lowercase();
    if transport != "sse" {
        bail!("Unsupported transport '{transport}' in this HTTP/SSE build (expected 'sse')");
    }

    let bind_addr = "127.0.0.1:8000";
    info!(%bind_addr, database = %state.database, read_only = %state.read_only, "Starting Universal Neo4j GraphRAG MCP server");

    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .context("Failed to bind TCP listener on 127.0.0.1:8000")?;

    axum::serve(listener, build_router(state))
        .await
        .context("HTTP server exited with error")?;

    Ok(())
}
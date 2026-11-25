# Movies GraphRAG Neo4j SSE MCP Server and Client (Rust + Neo4j)

An end-to-end Rust implementation of a **Model Context Protocol (MCP)** server and client over **HTTP / Server-Sent Events (SSE)** backed by **Neo4j**, migrated from Python (`FastMCP` / low-level `mcp.server.Server` + Starlette SSE) to **Rust (edition 2024)** using **`rmcp` (`3.11.0`)**, **`axum` (`0.8`)**, **`neo4rs` (`0.8`)**, **`reqwest` (`0.13`)**, and **`tokio`**.

The server exposes an SSE session stream (`GET /sse`) and a JSON-RPC message endpoint (`POST /messages`) on `0.0.0.0:8000` and provides the `graph_statistics` MCP tool to query node and relationship counts from Neo4j. The client connects to `http://localhost:8000/sse`, performs the MCP `initialize` handshake, invokes `graph_statistics`, and prints the formatted JSON result.

## Architecture & Project Structure

```
neo4j_connection/
├── neo4j_connection_server/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs          # AppState, MoviesGraphServer, Neo4j query logic & rmcp tool handler
│   │   └── main.rs         # Tokio runtime, Neo4j pool & health check, Axum /sse & /messages bridge
│   └── tests/
│       └── server_tests.rs # Metadata/capability unit tests & live Neo4j integration test
└── neo4j_connection_client/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs          # SSE parser, URL resolver, MCP handshake & graph_statistics client logic
    │   └── main.rs         # CLI entry point connecting to http://localhost:8000/sse
    └── tests/
        └── client_sse_tests.rs # SSE parser/formatter unit tests & ephemeral Axum SSE E2E test
```

### Exposed MCP Capabilities

#### Tools

| Tool Name | Arguments | Description |
| :--- | :--- | :--- |
| `graph_statistics` | None | Runs `RETURN COUNT {()} AS nodes, COUNT {()-[]->()} AS relationships` against Neo4j and returns `{"nodes": <int>, "relationships": <int>}`. |

#### HTTP / SSE Endpoints

| Method & Path | Description |
| :--- | :--- |
| `GET /sse` | Opens a `text/event-stream` connection, registers a UUID session, and emits an `endpoint` event containing `/messages?sessionId=<uuid>` followed by JSON-RPC `message` events. |
| `POST /messages?sessionId=<uuid>` | Receives JSON-RPC 2.0 requests (`initialize`, `notifications/initialized`, `tools/list`, `tools/call`) and pushes responses onto the matching SSE session channel. |

### Key Design Details

* **Shared Neo4j Connection Pool (`Arc<AppState>`):** Replaces Python's global driver pattern with an `Arc<AppState>` wrapping `Arc<neo4rs::Graph>` and the target database name, safely shared across concurrent Axum tasks.
* **Startup Connectivity Verification:** Before binding `0.0.0.0:8000`, the server executes `RETURN 1` against Neo4j to fail fast if the database is unreachable or misconfigured.
* **Dynamic Origin Resolution on the Client:** `origin_from_url` and `resolve_endpoint_url` derive the base origin from the input SSE URL, allowing the client to run seamlessly against both `http://localhost:8000/sse` and ephemeral test servers (`http://127.0.0.1:<port>/sse`).

## Prerequisites

* **Rust** (`cargo`, `rustc` supporting `edition = "2024"`)
* **Neo4j** instance running locally or remotely (default `neo4j://localhost:7687` or `bolt://localhost:7687`) with your target database (e.g., `moviedatabase` or `neo4j`).

### Environment Variables

Configure via environment variables or a `.env` file loaded via `dotenvy`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `NEO4J_URI` | `bolt://localhost:7687` | Bolt / Neo4j connection URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Target Neo4j database name (e.g., `moviedatabase`) |

## Building and Running

### 1. Start the SSE MCP Server

Ensure Neo4j is running and reachable, then start the server from `neo4j_connection/neo4j_connection_server`:

```bash
cargo run
```

The server will verify the Neo4j connection and start listening on `http://0.0.0.0:8000` (`/sse` and `/messages`).

### 2. Run the SSE MCP Client

In a second terminal, from `neo4j_connection/neo4j_connection_client`, run:

```bash
cargo run
```

Example output:

```text
Result from MCP Server:
----------------------------------------
{
  "nodes": 1132,
  "relationships": 8847
}
----------------------------------------
```

## Running Tests

### Client Unit & Ephemeral SSE End-to-End Tests

From `neo4j_connection/neo4j_connection_client`:

```bash
cargo test
```

This runs unit tests for SSE event parsing, endpoint URL resolution, and JSON formatting, as well as an end-to-end test against an ephemeral in-process Axum SSE server.

### Server Unit & Live Neo4j Integration Tests

From `neo4j_connection/neo4j_connection_server`, run the offline unit tests:

```bash
cargo test
```

To also run the `#[ignore]` integration test against a live Neo4j database configured via `NEO4J_*` environment variables:

```bash
cargo test -- --ignored
```

## Troubleshooting

* **Connection Refused on `http://localhost:8000/sse`:** Make sure `neo4j_connection_server` is running first and listening on port `8000`.
* **Neo4j Startup Failure:** Verify that Neo4j is running on port `7687` (e.g., `Test-NetConnection -ComputerName 127.0.0.1 -Port 7687`) and that `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `NEO4J_DATABASE` match your instance.
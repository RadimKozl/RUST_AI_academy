# Universal Neo4j GraphRAG MCP Server & Client (Rust)

A high-performance, modular Model Context Protocol (MCP) server and client implementation written in Rust (Edition 2024) using HTTP and Server-Sent Events (SSE) transport, backed by a Neo4j graph database.

---

## Architecture Overview

```text
+---------------------------+        HTTP GET /sse (Event Stream)        +-----------------------------+
|                           | <----------------------------------------- |                             |
|   universal_mcp_client    |                                            |    universal_mcp_server     |
|  (reqwest + tokio SSE)    | ---- HTTP POST /messages?sessionId=... --> |  (axum + neo4rs v0.8.0)     |
+---------------------------+          (JSON-RPC 2.0 Requests)           +--------------+--------------+
                                                                                        |
                                                                                        | Bolt Protocol
                                                                                        v
                                                                         +-----------------------------+
                                                                         |       Neo4j Database        |
                                                                         |  (e.g., moviedatabase)      |
                                                                         +-----------------------------+
```

### Project Structure

```text
universal_mcp/
├── universal_mcp_server/
│   ├── Cargo.toml
│   ├── config.json
│   ├── src/
│   │   ├── lib.rs          # Core MCP JSON-RPC dispatcher, SSE router, Neo4j Bolt <-> JSON mapping
│   │   └── main.rs         # CLI & environment entry point
│   └── tests/
│       └── server_tests.rs # Unit and HTTP/SSE integration tests
└── universal_mcp_client/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs          # SSE stream parser & JSON-RPC over HTTP/SSE client routines
    │   └── main.rs         # End-to-end verification runner
    └── tests/
        └── client_sse_tests.rs # Mock SSE server & client handshake tests
```

---

## Features

- **HTTP/SSE MCP Transport (`2025-11-25`):** Asynchronous SSE session management with unique `sessionId` routing via `axum`.
- **Dynamic Configuration (`config.json`):** Supports environment variable interpolation (`${VAR:-default}`) for database credentials, custom Cypher tools, prompt templates, and resource patterns.
- **Read-Only Guardrails:** Enforce read-only database access globally via `--read-only` CLI flag, `READ_ONLY=true` environment variable, or `"read_only": true` in `config.json`.
- **Built-in & Custom MCP Tools:**
  - `read_cypher` — Execute parameterized read-only Cypher queries.
  - `write_cypher` — Execute parameterized write Cypher queries (blocked when `read_only` is enabled).
  - `get_graph_schema` — Introspect node labels, relationship types, and node/relationship counts.
  - `run_configured_tool` — Invoke custom Cypher tools defined in `config.json`.
- **Graph Resource Templates:** Read node properties (`node://{label}/{id}`) and relationships (`node://{label}/{id}/relationships`) with automatic ID type matching (`id`, `tmdbId`, `movie_tmdbId`, `movieId`, `uid`, `name`, `title`).
- **System & Custom Prompts:** Expose `system_instructions` and user-defined prompt templates via `prompts/list` and `prompts/get`.

---

## Prerequisites

- **Rust Toolchain:** Rust 1.85+ (Edition 2024) with `cargo`
- **Neo4j Instance:** Neo4j 5.x (Desktop, Docker, or Aura) running and accessible over Bolt (`bolt://localhost:7687`)

---

## Configuration

Place `config.json` in the same directory as the compiled server binary (e.g., `target/debug/config.json` or `target/release/config.json`), or point to it explicitly using the `CONFIG_FILE` environment variable.

### Example `config.json`

```json
{
  "mcp_transport": "sse",
  "read_only": false,
  "neo4j": {
    "uri": "${NEO4J_URI:-bolt://localhost:7687}",
    "username": "${NEO4J_USERNAME:-neo4j}",
    "password": "${NEO4J_PASSWORD:-password}",
    "database": "${NEO4J_DATABASE:-moviedatabase}"
  },
  "agent_instructions": "You are an expert Neo4j database assistant. Always analyze the graph schema before writing complex queries and return results in clean Markdown.",
  "resources": [
    {
      "uri_pattern": "node://{label}/{id}",
      "description": "Fetch properties and labels of a node by label and identifier.",
      "cypher_query": "MATCH (n) WHERE $label IN labels(n) RETURN properties(n) AS props, labels(n) AS labels"
    }
  ],
  "custom_tools": [
    {
      "name": "find_nodes_by_property",
      "description": "Search graph nodes with a limit parameter.",
      "cypher_query": "MATCH (n) RETURN n LIMIT $limit",
      "parameters": ["limit"],
      "write": false
    }
  ],
  "prompts": [
    {
      "name": "analyze_graph_schema",
      "description": "Prompt guiding the assistant to inspect labels and relationship types first.",
      "template": "Inspect all node labels and relationship types in the database before answering user questions."
    }
  ]
}
```

### Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CONFIG_FILE` | `config.json` | Path to the server JSON configuration file |
| `READ_ONLY` | `false` | Set to `true`, `1`, or `yes` to disable all write queries |
| `NEO4J_URI` | `bolt://localhost:7687` | Bolt connection URI for Neo4j |
| `NEO4J_USERNAME` | `neo4j` | Neo4j database user |
| `NEO4J_PASSWORD` | `password` | Neo4j database password |
| `NEO4J_DATABASE` | `neo4j` | Target Neo4j database name |
| `MCP_SERVER_URL` | `http://localhost:8000/sse` | SSE endpoint URL used by `universal_mcp_client` |
| `MCP_RESOURCE_URI` | `node://Movie/11443` | Resource URI queried during client verification |

---

## Quick Start

### 1. Start the MCP Server

```powershell
cd universal_mcp_server
$env:CONFIG_FILE = "$PWD\config.json"
cargo run
```

To start the server in strict read-only mode:

```powershell
cargo run -- --read-only
```

The server listens on `http://127.0.0.1:8000` (`GET /sse` and `POST /messages`).

### 2. Run the MCP Client

In a separate terminal:

```powershell
cd universal_mcp_client
cargo run
```

The client will:
1. Establish an SSE connection to `http://localhost:8000/sse` and receive its session endpoint.
2. Perform the MCP `initialize` handshake.
3. List available tools (`tools/list`) and invoke `get_graph_schema`.
4. List prompts (`prompts/list`) and fetch `system_instructions`.
5. Read the graph resource `node://Movie/11443` via `resources/read`.

---

## Running Tests

Both crates include unit and integration test suites that run without requiring an active Neo4j instance:

```powershell
# Server unit & HTTP/SSE router tests
cd universal_mcp_server
cargo test

# Optional: run ignored live Neo4j integration test
cargo test -- --ignored

# Client SSE parser & end-to-end mock bridge tests
cd ..\universal_mcp_client
cargo test
```

---

## License

MIT

--------------------------------------------------------------------------
>NOTE: Copy `config.json` file to `./univerdsal_mcp_server/target/debug/`
--------------------------------------------------------------------------
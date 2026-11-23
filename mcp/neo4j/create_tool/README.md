# Movies GraphRAG MCP Tools Server & Client (Rust + Neo4j)

An end-to-end Rust implementation of a **Model Context Protocol (MCP)** tool server and client backed by **Neo4j** (`moviedatabase`), migrated from Python (`FastMCP` / low-level `mcp.server.Server`) to **Rust (edition 2024)** using **`rmcp` (`3.11.0`)**, **`neo4rs` (`0.8`)**, and **`tokio`**.

The server exposes graph analytics and genre-based movie search tools via `#[tool_router]` macros over `stdio` transport. The client spawns the compiled server binary as a child process via `TokioChildProcess` and executes JSON-RPC tool calls.

## Architecture & Project Structure

```text
create_tool/
├── create_tool_server/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs          # AppState, domain models, JSON schema params, MCP tool handlers
│   │   └── main.rs         # Tokio runtime, Neo4j connection pool, stdio MCP server entry point
│   └── tests/
│       └── server_tests.rs # Serialization, duplex capability/validation, and Neo4j integration tests
└── create_tool_client/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs          # Child process launcher, MCP client helpers, timeout wrappers
    │   └── main.rs         # CLI client calling graph_statistics and get_movies_by_genre
    └── tests/
        └── client_integration.rs # Binary path resolution, missing binary, and E2E integration tests
```

### Exposed MCP Tools

| Tool Name | Arguments | Description |
| :--- | :--- | :--- |
| `graph_statistics` | None | Counts the total number of nodes and relationships in the Neo4j graph (`{"nodes": ..., "relationships": ...}`). |
| `get_movies_by_genre` | `genre: String` (required), `limit: Option<i64>` (default: `10`, clamped to `1..=200`) | Queries `(:Movie)-[:IN_GENRE]->(:Genre {name: $genre})` ordered by `imdbRating DESC`. |

### Key Design Details

* **Strict `stdout` / `stderr` Separation:** The server writes `tracing` logs exclusively to `stderr` so `stdout` remains clean for MCP JSON-RPC frames.
* **Shared Neo4j Connection Pool:** `AppState` wraps `Arc<Graph>` and the target database name (`moviedatabase`), sharing a connection pool across tool invocations.
* **Automatic JSON Schema Generation:** `GetMoviesByGenreParams` derives `schemars::JsonSchema` so `rmcp`'s `#[tool_router]` automatically advertises parameter types and descriptions to clients.
* **Subprocess Binary Execution:** `create_tool_client` launches the compiled `create_tool_server` binary. Always rebuild `create_tool_server` after modifying server code before running the client or E2E tests.

## Prerequisites

* **Rust** (`cargo`, `rustc` supporting `edition = "2024"`)
* **Neo4j** instance running locally or remotely with the **`moviedatabase`** database populated with `Movie` and `Genre` nodes connected by `IN_GENRE` relationships.

### Environment Variables

Configure via environment variables or a `.env` file loaded via `dotenvy`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `NEO4J_URI` | `neo4j://127.0.0.1:7687` (or `neo4j://localhost:7687`) | Bolt / Neo4j connection URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `NEO4J_DATABASE` | `moviedatabase` | Target Neo4j database name |
| `MCP_SERVER_BIN` | `../create_tool_server/target/release/create_tool_server` (fallback `debug`) | Override path to the compiled server binary |

## Building and Running

### 1. Build the Server Binary

From `create_tool/create_tool_server`:

```powershell
cargo build
```

For optimized execution (automatically preferred by `default_server_binary_path()`):

```powershell
cargo build --release
```

### 2. Run the Client Against the Server

From `create_tool/create_tool_client`, pass optional `genre` and `limit` arguments (defaults to `Action` and `5`):

```powershell
cargo run
```

Or specify a custom genre and result limit:

```powershell
cargo run -- Sci-Fi 3
```

Example execution flow:

1. Spawns the compiled `create_tool_server` binary over `stdio` and completes the MCP handshake (`protocol_version: 2025-11-25`).
2. Lists available tools (`get_movies_by_genre`, `graph_statistics`).
3. Calls `graph_statistics` and prints node and relationship counts.
4. Calls `get_movies_by_genre` and pretty-prints the returned JSON array of movies.
5. Gracefully cancels the MCP client session and exits the server child process with exit code `0`.

## Running Tests

### Unit & Non-Ignored Integration Tests

From `create_tool/create_tool_server`:

```powershell
cargo test
```

From `create_tool/create_tool_client`:

```powershell
cargo test
```

### Live Neo4j & End-to-End (`#[ignore]`) Tests

Ensure Neo4j (`moviedatabase`) is reachable and the server binary is built:

```powershell
cargo build --manifest-path ..\create_tool_server\Cargo.toml
cargo test --test server_tests -- --ignored --nocapture
cargo test --test client_integration -- --ignored --nocapture
```

## Troubleshooting

* **Client executes outdated server logic:** Rebuild `create_tool_server` (`cargo build` or `cargo build --release`) so the child process binary reflects recent changes.
* **Unresolved `serde_json` crate in client:** Ensure `serde_json = "1.0"` is explicitly listed under `[dependencies]` in `create_tool_client/Cargo.toml`.
* **Empty results from `get_movies_by_genre`:** Verify that genre names match the case stored in Neo4j (e.g., `"Action"`, `"Comedy"`, `"Sci-Fi"`, `"Drama"`).
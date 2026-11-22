# Movies GraphRAG MCP Server & Client (Rust + Neo4j)

An end-to-end Rust implementation of a **Model Context Protocol (MCP)** server and client backed by **Neo4j** (`moviedatabase`), built with **Rust (edition 2024)**, **`rmcp` (`3.11.0`)**, **`neo4rs` (`0.8`)**, and **`tokio`**.

The server exposes graph analytics tools, genre-based movie queries, and a dynamic `movie://{tmdb_id}` resource template that renders structured Markdown details (including plot, genres, directors, and cast). The client spawns the server binary over `stdio` via `TokioChildProcess` and executes JSON-RPC tool and resource calls.

---

## Architecture & Project Structure

```text
create_resource/
├── create_resource_server/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs          # Domain models, Neo4j queries, Markdown formatter, MCP ServerHandler
│   │   └── main.rs         # Tokio runtime, Neo4j connection pool, stdio MCP server entry point
│   └── tests/
│       └── server_tests.rs # Markdown formatting, duplex capability/validation, and Neo4j integration tests
└── create_resource_client/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs          # Child process launcher, MCP client helpers, timeout wrappers
    │   └── main.rs         # CLI client querying tools, resource templates, and movie://{tmdb_id}
    └── tests/
        └── client_integration.rs # Binary path, error/timeout, and E2E integration tests
```

### Key Design Details

- **Separate `stdout` and `stderr`:** The server writes `tracing` logs exclusively to `stderr` so `stdout` stays dedicated to MCP JSON-RPC frames.
- **Shared Neo4j Pool:** `AppState` wraps `Arc<Graph>` and the target database name (`moviedatabase`), sharing a connection pool across tool and resource handlers.
- **Property Mapping (`movie_tmdbId`):** Movie nodes in `moviedatabase` store TMDB identifiers under `movie_tmdbId` (e.g., `16420` for *Othello*). Queries normalize values with `toString(m.movie_tmdbId)` so string and numeric values match reliably.
- **Subprocess Rebuild Reminder:** Because `create_resource_client` launches the compiled `create_resource_server` binary, rebuild the server whenever `create_resource_server` changes before running the client or E2E tests.

---

## Prerequisites

- **Rust** (`cargo`, `rustc` supporting `edition = "2024"`)
- **Neo4j** instance running locally or remotely with the **`moviedatabase`** database populated with `Movie`, `Genre`, and `Person` nodes and `IN_GENRE`, `ACTED_IN`, and `DIRECTED` relationships.

### Environment Variables

You can set these via environment variables or a `.env` file loaded with `dotenvy`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `NEO4J_URI` | `neo4j://127.0.0.1:7687` (or `neo4j://localhost:7687`) | Bolt / Neo4j connection URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `NEO4J_DATABASE` | `moviedatabase` | Target Neo4j database name |
| `MCP_SERVER_BIN` | `../create_resource_server/target/release/create_resource_server` (fallback `debug`) | Override path to compiled server binary |

---

## Building and Running

### 1. Build the Server Binary

From `create_resource/create_resource_server`:

```powershell
cargo build
```

For faster runtime execution (picked up automatically first by `default_server_binary_path()`):

```powershell
cargo build --release
```

### 2. Run the Client Against the Server

From `create_resource/create_resource_client`, pass a `movie_tmdbId` argument (defaults to `16420` if omitted):

```powershell
cargo run -- 16420
```

Example output flow:

1. Connects to the compiled `create_resource_server` binary over `stdio` and completes the MCP initialization handshake (`protocol_version: 2025-11-25`).
2. Lists available tools (`graph_statistics`, `get_movies_by_genre`).
3. Calls `graph_statistics` and `get_movies_by_genre` (`genre: "Sci-Fi"`, `limit: 3`).
4. Lists resource templates (`movie://{tmdb_id}`).
5. Reads `movie://16420` and prints the formatted Markdown movie details before cleanly cancelling the session.

---

## Running Tests

### Unit & Non-Ignored Integration Tests

Run server tests from `create_resource/create_resource_server`:

```powershell
cargo test
```

Run client tests from `create_resource/create_resource_client`:

```powershell
cargo test
```

### Live Neo4j & End-to-End (`#[ignore]`) Tests

Ensure Neo4j (`moviedatabase`) is running and the server binary is built:

```powershell
cargo build --manifest-path ..\create_resource_server\Cargo.toml
cargo test --test server_tests -- --ignored --nocapture
cargo test --test client_integration -- --ignored --nocapture
```

---

## Troubleshooting

- **`Movie with TMDB ID <id> not found in database`:** Verify that the target `Movie` node in `moviedatabase` has a `movie_tmdbId` property matching `<id>`.
- **Client runs old server code after edits:** Recompile `create_resource_server` (`cargo build` or `cargo build --release`) so the child process binary reflects your changes.
- **Generic type syntax in JSON argument helpers:** Use `Option<Map<String, Value>>` (with `serde_json::{Map, Value}`) when constructing tool argument maps.
# Movies GraphRAG Paginated MCP Tools & Resources Server and Client (Rust + Neo4j)

An end-to-end Rust implementation of a **Model Context Protocol (MCP)** server and client with pagination and resource template support backed by **Neo4j** (`moviedatabase`), migrated from Python (`FastMCP` / low-level `mcp.server.Server` + SSE) to **Rust (edition 2024)** using **`rmcp` (`3.11.0`)**, **`neo4rs` (`0.8`)**, and **`tokio`**.

The server exposes graph statistics, genre queries, cursor-based paginated genre browsing, and a `movie://{tmdb_id}` resource template over `stdio` transport. The client spawns the compiled server binary via `TokioChildProcess` and supports both CLI execution and an interactive pagination loop.

## Architecture & Project Structure

```
create_paginated_tool/
├── create_paginated_tool_server/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs          # AppState, pagination/movie models, MCP tools & resource handlers
│   │   └── main.rs         # Tokio runtime, Neo4j pool & health check, stdio MCP entry point
│   └── tests/
│       └── server_tests.rs # Parameter/Markdown tests, duplex capability tests & Neo4j integration tests
└── create_paginated_tool_client/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs          # Subprocess launcher, tool/template/resource helpers, timeout wrappers
    │   └── main.rs         # CLI & --interactive pagination client
    └── tests/
        └── client_integration.rs # Binary path override, missing binary & E2E integration tests
```

### Exposed MCP Capabilities

#### Tools

| Tool Name | Arguments | Description |
| :--- | :--- | :--- |
| `graph_statistics` | None | Counts total nodes and relationships in the Neo4j graph (`{"nodes": ..., "relationships": ...}`). |
| `get_movies_by_genre` | `genre: String` (required), `limit: Option<i64>` (default: `10`, clamped to `1..=200`) | Returns movies in `(:Movie)-[:IN_GENRE]->(:Genre {name: $genre})` ordered by `coalesce(m.imdbRating, 0) DESC`. |
| `list_movies_by_genre` | `genre: String` (required), `page_size: Option<i64>` (default: `10`, clamped to `1..=200`), `cursor: Option<i64>` (default: `0`, clamped `>= 0`) | Paginates movies ordered by `m.title ASC` with `SKIP $skip` and `LIMIT $limit`, returning `genre`, `movies`, `next_cursor`, `page`, `page_size`, and `has_more`. |

#### Resource Templates

| URI Template | Name | MIME Type | Description |
| :--- | :--- | :--- | :--- |
| `movie://{tmdb_id}` | `get_movie` | `text/plain` | Queries `Movie` where `toString(m.movie_tmdbId) = $tmdb_id` and returns formatted Markdown details (title, release date, runtime, genres, directors, plot, and cast). |

### Key Design Details

* **Strict `stdout` / `stderr` Separation:** Server `tracing` logs are written exclusively to `stderr` so `stdout` carries only MCP JSON-RPC protocol traffic.
* **Connection Health Check Before Handshake:** The server connects to Neo4j with a shared pool (`max size 16`) and runs `RETURN 1` with timeouts before serving `stdio`, failing fast if Neo4j is unreachable.
* **Subprocess Execution (`TokioChildProcess`):** `create_paginated_tool_client` spawns the compiled `create_paginated_tool_server` binary automatically. Rebuild `create_paginated_tool_server` after making server changes.

## Prerequisites

* **Rust** (`cargo`, `rustc` supporting `edition = "2024"`)
* **Neo4j** instance running locally or remotely with the **`moviedatabase`** database populated with `Movie` (including `movie_tmdbId`), `Genre`, and `Person` nodes and `IN_GENRE`, `ACTED_IN`, and `DIRECTED` relationships.

### Environment Variables

Configure via environment variables or a `.env` file loaded via `dotenvy`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `NEO4J_URI` | `neo4j://127.0.0.1:7687` | Bolt / Neo4j connection URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `NEO4J_DATABASE` | `moviedatabase` | Target Neo4j database name |
| `MCP_SERVER_BIN` | `../create_paginated_tool_server/target/release/create_paginated_tool_server` (fallback `debug`) | Override path to the compiled server binary |

## Building and Running

### 1. Build the Server Binary

From `create_paginated_tool/create_paginated_tool_server`:

```bash
cargo build
```

Or for an optimized release binary:

```bash
cargo build --release
```

### 2. Run the Client

Ensure Neo4j is running and reachable on `NEO4J_URI` (for example `127.0.0.1:7687`). From `create_paginated_tool/create_paginated_tool_client`, run with default positional arguments (`Action`, `page_size=5`, `cursor=0`, `tmdb_id=16420`):

```bash
cargo run
```

Or pass custom positional arguments (`<genre> <page_size> <cursor> <tmdb_id>`):

```bash
cargo run -- Action 5 5 16420
```

Or launch interactive pagination mode (`--interactive` or `-i`):

```bash
cargo run -- --interactive
```

## Running Tests

### Unit & Non-Ignored Tests

From `create_paginated_tool/create_paginated_tool_server`:

```bash
cargo test
```

From `create_paginated_tool/create_paginated_tool_client`:

```bash
cargo test
```

### Live Neo4j & End-to-End (`#[ignore]`) Tests

Build the server binary and run ignored live tests against Neo4j:

```bash
cargo build --manifest-path ..\create_paginated_tool_server\Cargo.toml
cargo test --test server_tests -- --ignored --nocapture
cargo test --test client_integration -- --ignored --nocapture
```

## Troubleshooting

* **`Timed out verifying Neo4j connection` / `ConnectionClosed("initialize response")`:** Confirm Neo4j is started and listening on `7687` (e.g., `Test-NetConnection -ComputerName 127.0.0.1 -Port 7687`) and verify credentials and `NEO4J_DATABASE` (`moviedatabase`).
* **Stale server behavior:** Rebuild `create_paginated_tool_server` (`cargo build` or `cargo build --release`) so the spawned child binary contains your latest changes.
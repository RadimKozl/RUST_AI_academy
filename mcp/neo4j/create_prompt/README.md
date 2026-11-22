# Movies GraphRAG MCP Server & Client (`create_prompt`)

An implementation of a **Model Context Protocol (MCP)** server and client in **Rust (edition 2024)** backed by a **Neo4j** graph database within the `RUST_AI_academy` repository.

## Architecture

- **`create_prompt_server`**: An MCP server built with `rmcp` (`3.11.0`) and `neo4rs` (`0.8`) running on the `tokio` async runtime. Communicates over `stdio`, with all logging directed exclusively to `stderr`.
  - **Tools**:
    - `graph_statistics` – counts `nodes` and `relationships` in the Neo4j graph.
    - `get_movies_by_genre` – returns movies for a given genre ordered by IMDb rating (`limit` from 1 to 100).
    - `list_movies_by_genre` – browses movies in a genre with cursor/offset pagination (`page_size`, `cursor`).
  - **Resource Templates**:
    - `movie://{tmdb_id}` – retrieves movie details by `tmdbId` (title, release year, tagline, runtime, plot/synopsis, directors, cast).
  - **Prompts**:
    - `movie_night_planner`
    - `discovery_prompt` (`genre`, optional `decade`, `mood`)
    - `analyze_preferences` (`favorite_movies`)
- **`create_prompt_client`**: An asynchronous MCP client that spawns the server binary via `TokioChildProcess`, verifies tools, resources, and prompts, and supports overriding the binary path via `MCP_SERVER_BIN`.

## Configuration (`.env`)

Create a `.env` file with your Neo4j connection details and target database:

```env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=moviedatabase
```

## Dependencies (`Cargo.toml`)

Both crates target Rust `edition = "2024"` and `rmcp = "3.11.0"`:

```toml
# create_prompt_server/Cargo.toml
[dependencies]
dotenvy = "0.15"
neo4rs = "0.8"
rmcp = { version = "3.11.0", features = ["server", "macros", "transport-io"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1", features = ["full"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

## Running & Testing

### 1. Build the Server

```powershell
cargo build --manifest-path .\create_prompt_server\Cargo.toml
```

### 2. Run the Client Against the Server and Neo4j

```powershell
cargo run --manifest-path .\create_prompt_client\Cargo.toml
```

### 3. Run Unit & Integration Tests

```powershell
cargo test --manifest-path .\create_prompt_server\Cargo.toml
cargo test --manifest-path .\create_prompt_client\Cargo.toml
```

End-to-end tests that require a running Neo4j instance and a compiled server binary are marked with `#[ignore]` and can be run with:

```powershell
cargo test --manifest-path .\create_prompt_client\Cargo.toml -- --ignored --nocapture
```

## Performance & Reliability Notes

- **Connection Pooling in `neo4rs`**: A connection pool (max size 16) configured with `ConfigBuilder` and targeted at `moviedatabase` reuses connections across async queries after initial startup verification (`RETURN 1`).
- **Clean `stdio` Transport**: Routing `tracing` logs to `stderr` keeps `stdout` free of non-protocol output so MCP JSON-RPC frames remain intact.
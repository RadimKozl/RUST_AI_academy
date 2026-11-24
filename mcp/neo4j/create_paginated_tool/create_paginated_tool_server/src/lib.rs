use std::sync::Arc;

use neo4rs::{query, BoltMap, Graph, Row};
use rmcp::{
    handler::server::{
        router::{
            prompt::PromptRouter,
            tool::ToolRouter,
        },
        wrapper::Parameters,
    },
    model::{
        CallToolResult, ErrorData as McpError, InitializeResult, ListResourceTemplatesResult,
        ListResourcesResult, PaginatedRequestParams, ReadResourceRequestParams,
        ReadResourceResult, ResourceContents, ResourceTemplate, ServerCapabilities,
    },
    schemars, service::RequestContext, tool, tool_handler, tool_router, RoleServer, ServerHandler,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{error, info, warn};

fn push_text_content<T>(out: &mut Vec<T>, text: String) -> Result<(), McpError>
where
    T: DeserializeOwned,
{
    let item = serde_json::from_value(json!({
        "type": "text",
        "text": text
    }))
    .map_err(|e| McpError::internal_error(format!("Failed building tool content: {e}"), None))?;
    out.push(item);
    Ok(())
}

#[derive(Clone)]
pub struct AppState {
    pub graph: Arc<Graph>,
    pub database: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct GraphStats {
    pub nodes: i64,
    pub relationships: i64,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct MovieSummary {
    pub title: Option<String>,
    #[serde(rename = "imdbRating")]
    pub imdb_rating: Option<f64>,
    pub released: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct PaginatedMovieItem {
    pub title: Option<String>,
    pub released: Option<i64>,
    pub rating: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct PaginatedMoviesResponse {
    pub genre: String,
    pub movies: Vec<PaginatedMovieItem>,
    pub next_cursor: Option<i64>,
    pub page: i64,
    pub page_size: i64,
    pub has_more: bool,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetMoviesByGenreParams {
    #[schemars(description = "The genre to search for (e.g., \"Action\", \"Drama\", \"Comedy\")")]
    pub genre: String,
    #[schemars(description = "Maximum number of movies to return (default: 10)")]
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListMoviesByGenreParams {
    #[schemars(description = "Genre name (e.g., \"Action\", \"Comedy\", \"Drama\")")]
    pub genre: String,
    #[schemars(description = "Number of movies to return per page (default: 10)")]
    pub page_size: Option<i64>,
    #[schemars(description = "Pagination cursor / offset index (default: 0)")]
    pub cursor: Option<i64>,
}

#[derive(Debug)]
pub struct CastMember {
    pub name: String,
    pub role: Option<String>,
}

#[derive(Debug)]
pub struct MovieDetails {
    pub title: String,
    pub released: String,
    pub tagline: Option<String>,
    pub runtime: String,
    pub plot: String,
    pub genres: Vec<String>,
    pub directors: Vec<String>,
    pub actors: Vec<CastMember>,
}

pub fn format_movie_markdown(details: &MovieDetails) -> String {
    let mut lines = Vec::new();
    lines.push(format!("# {} ({})", details.title, details.released));
    lines.push(String::new());

    if let Some(tagline) = details.tagline.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        lines.push(format!("_{tagline}_"));
        lines.push(String::new());
    }

    lines.push(format!("**Runtime:** {} minutes", details.runtime));

    let genres_str = if details.genres.is_empty() {
        "N/A".to_string()
    } else {
        details.genres.join(", ")
    };
    lines.push(format!("**Genres:** {genres_str}"));

    if !details.directors.is_empty() {
        lines.push(format!("**Director(s):** {}", details.directors.join(", ")));
    }

    lines.push(String::new());
    lines.push("## Plot".to_string());
    lines.push(details.plot.clone());

    if !details.actors.is_empty() {
        lines.push(String::new());
        lines.push("## Cast".to_string());
        for actor in &details.actors {
            match actor.role.as_deref().map(str::trim).filter(|r| !r.is_empty()) {
                Some(role) => lines.push(format!("- {} as {role}", actor.name)),
                None => lines.push(format!("- {}", actor.name)),
            }
        }
    }

    lines.join("\n")
}

pub async fn fetch_movie_markdown(
    graph: &Graph,
    database: &str,
    tmdb_id: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let cypher = r#"
        MATCH (m:Movie)
        WHERE toString(m.movie_tmdbId) = $tmdb_id
        RETURN m.title AS title,
               m.released AS released,
               coalesce(m.tagline, "") AS tagline,
               m.runtime AS runtime,
               m.plot AS plot,
               [ (m)-[:IN_GENRE]->(g:Genre) | g.name ] AS genres,
               [ (p)-[r:ACTED_IN]->(m) | {name: p.name, role: r.role} ] AS actors,
               [ (d)-[:DIRECTED]->(m) | d.name ] AS directors
    "#;

    let mut result = graph
        .execute(query(cypher).param("tmdb_id", tmdb_id))
        .await?;

    let Some(row): Option<Row> = result.next().await? else {
        warn!(database = %database, tmdb_id = %tmdb_id, "Movie not found in database");
        return Ok(format!("Movie with TMDB ID {tmdb_id} not found in database"));
    };

    let title: String = row
        .get("title")
        .unwrap_or_else(|_| "Unknown Title".to_string());
    let released: String = row
        .get::<String>("released")
        .or_else(|_| row.get::<i64>("released").map(|y| y.to_string()))
        .unwrap_or_else(|_| "N/A".to_string());
    let tagline: Option<String> = row.get::<String>("tagline").ok();
    let runtime: String = row
        .get::<String>("runtime")
        .or_else(|_| row.get::<i64>("runtime").map(|r| r.to_string()))
        .unwrap_or_else(|_| "N/A".to_string());
    let plot: String = row.get("plot").unwrap_or_default();
    let genres: Vec<String> = row.get("genres").unwrap_or_default();
    let directors: Vec<String> = row.get("directors").unwrap_or_default();

    let raw_actors: Vec<BoltMap> = row.get("actors").unwrap_or_default();
    let actors: Vec<CastMember> = raw_actors
        .into_iter()
        .map(|map| {
            let name = map
                .get::<String>("name")
                .unwrap_or_else(|_| "Unknown Actor".to_string());
            let role = map.get::<String>("role").ok();
            CastMember { name, role }
        })
        .collect();

    let details = MovieDetails {
        title,
        released,
        tagline,
        runtime,
        plot,
        genres,
        directors,
        actors,
    };

    Ok(format_movie_markdown(&details))
}

#[derive(Clone)]
pub struct MoviesPaginatedServer {
    state: Arc<AppState>,
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
    #[allow(dead_code)]
    prompt_router: PromptRouter<Self>,
}

#[tool_router]
impl MoviesPaginatedServer {
    pub fn new(state: Arc<AppState>) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
            prompt_router: PromptRouter::new(),
        }
    }

    #[tool(description = "Count the number of nodes and relationships in the Neo4j graph.")]
    async fn graph_statistics(&self) -> Result<CallToolResult, McpError> {
        let cypher = r#"RETURN COUNT {()} AS nodes, COUNT {()-[]-()} AS relationships"#;
        let mut result = self.state.graph.execute(query(cypher)).await.map_err(|e| {
            error!(error = %e, "graph_statistics query failed");
            McpError::internal_error(format!("Neo4j query failed: {e}"), None)
        })?;

        let stats = if let Some(row) = result
            .next()
            .await
            .map_err(|e| McpError::internal_error(format!("Neo4j row read failed: {e}"), None))?
        {
            GraphStats {
                nodes: row.get("nodes").unwrap_or(0),
                relationships: row.get("relationships").unwrap_or(0),
            }
        } else {
            GraphStats {
                nodes: 0,
                relationships: 0,
            }
        };

        let json_str = serde_json::to_string(&stats).map_err(|e| {
            McpError::internal_error(format!("JSON serialization failed: {e}"), None)
        })?;

        let mut out = Vec::new();
        push_text_content(&mut out, json_str)?;
        Ok(CallToolResult::success(out))
    }

    #[tool(description = "Get movies by genre from the Neo4j database ordered by IMDb rating.")]
    async fn get_movies_by_genre(
        &self,
        Parameters(params): Parameters<GetMoviesByGenreParams>,
    ) -> Result<CallToolResult, McpError> {
        let genre = params.genre.trim().to_string();
        if genre.is_empty() {
            return Err(McpError::invalid_params(
                "Missing or empty required argument: 'genre'".to_string(),
                None,
            ));
        }

        let limit = params.limit.unwrap_or(10).clamp(1, 200);
        info!(genre = %genre, limit = limit, "Searching for movies by genre");

        let cypher = r#"
            MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: $genre})
            RETURN m.title AS title,
                   m.imdbRating AS imdbRating,
                   m.released AS released
            ORDER BY coalesce(m.imdbRating, 0) DESC
            LIMIT $limit
        "#;

        let mut result = self
            .state
            .graph
            .execute(query(cypher).param("genre", genre.clone()).param("limit", limit))
            .await
            .map_err(|e| {
                error!(error = %e, genre = %genre, "Query failed for get_movies_by_genre");
                McpError::internal_error(format!("Neo4j query failed: {e}"), None)
            })?;

        let mut movies = Vec::new();
        while let Some(row) = result
            .next()
            .await
            .map_err(|e| McpError::internal_error(format!("Neo4j row read failed: {e}"), None))?
        {
            movies.push(MovieSummary {
                title: row.get("title").ok(),
                imdb_rating: row.get("imdbRating").ok(),
                released: row.get("released").ok(),
            });
        }

        if movies.is_empty() {
            warn!(genre = %genre, "No movies found for genre");
        } else {
            info!(genre = %genre, count = movies.len(), "Found movies for genre");
        }

        let json_str = serde_json::to_string(&movies).map_err(|e| {
            McpError::internal_error(format!("JSON serialization failed: {e}"), None)
        })?;

        let mut out = Vec::new();
        push_text_content(&mut out, json_str)?;
        Ok(CallToolResult::success(out))
    }

    #[tool(description = "Browse movies in a genre with pagination support.")]
    async fn list_movies_by_genre(
        &self,
        Parameters(params): Parameters<ListMoviesByGenreParams>,
    ) -> Result<CallToolResult, McpError> {
        let genre = params.genre.trim().to_string();
        if genre.is_empty() {
            return Err(McpError::invalid_params(
                "Missing or empty required argument: 'genre'".to_string(),
                None,
            ));
        }

        let page_size = params.page_size.unwrap_or(10).clamp(1, 200);
        let skip = params.cursor.unwrap_or(0).max(0);
        let page_num = (skip / page_size) + 1;

        info!(
            genre = %genre,
            page = page_num,
            page_size = page_size,
            skip = skip,
            "Fetching paginated movies by genre"
        );

        let cypher = r#"
            MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: $genre})
            RETURN m.title AS title,
                   m.released AS released,
                   m.imdbRating AS rating
            ORDER BY m.title ASC
            SKIP $skip
            LIMIT $limit
        "#;

        let mut result = self
            .state
            .graph
            .execute(
                query(cypher)
                    .param("genre", genre.clone())
                    .param("skip", skip)
                    .param("limit", page_size),
            )
            .await
            .map_err(|e| {
                error!(error = %e, genre = %genre, "Query failed for list_movies_by_genre");
                McpError::internal_error(format!("Neo4j query failed: {e}"), None)
            })?;

        let mut movies = Vec::new();
        while let Some(row) = result
            .next()
            .await
            .map_err(|e| McpError::internal_error(format!("Neo4j row read failed: {e}"), None))?
        {
            movies.push(PaginatedMovieItem {
                title: row.get("title").ok(),
                released: row.get("released").ok(),
                rating: row.get("rating").ok(),
            });
        }

        let has_more = (movies.len() as i64) == page_size;
        let next_cursor = if has_more { Some(skip + page_size) } else { None };

        info!(
            genre = %genre,
            page = page_num,
            count = movies.len(),
            has_more = has_more,
            "Returned paginated movies"
        );

        let payload = PaginatedMoviesResponse {
            genre,
            movies,
            next_cursor,
            page: page_num,
            page_size,
            has_more,
        };

        let json_str = serde_json::to_string_pretty(&payload).map_err(|e| {
            McpError::internal_error(format!("JSON serialization failed: {e}"), None)
        })?;

        let mut out = Vec::new();
        push_text_content(&mut out, json_str)?;
        Ok(CallToolResult::success(out))
    }
}

#[tool_handler]
impl ServerHandler for MoviesPaginatedServer {
    fn get_info(&self) -> InitializeResult {
        let capabilities = ServerCapabilities::builder()
            .enable_tools()
            .enable_resources()
            .build();

        InitializeResult::new(capabilities)
            .with_instructions("RUST_AI_academy Movies GraphRAG MCP server with pagination backed by Neo4j.")
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult::with_all_items(Vec::new()))
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        let template = ResourceTemplate::new("movie://{tmdb_id}", "get_movie")
            .with_mime_type("text/plain");

        Ok(ListResourceTemplatesResult::with_all_items(vec![template]))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::ReadResourceResponse, McpError> {
        let uri_str = request.uri.as_str();
        let tmdb_id = uri_str.strip_prefix("movie://").ok_or_else(|| {
            McpError::invalid_params(format!("Unsupported URI scheme: {uri_str}"), None)
        })?;

        info!(uri = %uri_str, tmdb_id = %tmdb_id, "Fetching movie resource");

        let markdown = fetch_movie_markdown(&self.state.graph, &self.state.database, tmdb_id)
            .await
            .map_err(|e| {
                error!(error = %e, uri = %uri_str, "Failed reading movie resource");
                McpError::internal_error(format!("Failed reading resource '{uri_str}': {e}"), None)
            })?;

        let contents = vec![ResourceContents::text(markdown, uri_str.to_string())];
        Ok(rmcp::model::ReadResourceResponse::Complete(
            ReadResourceResult::new(contents),
        ))
    }
}
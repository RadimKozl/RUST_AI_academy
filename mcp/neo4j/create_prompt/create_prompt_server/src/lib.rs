use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use neo4rs::{query, Graph};
use rmcp::{
    handler::server::{
        router::{prompt::PromptRouter, tool::ToolRouter},
        wrapper::Parameters,
    },
    model::{
        CallToolResult, ErrorData as McpError, GetPromptResult, InitializeResult,
        ListResourceTemplatesResult, ListResourcesResult, PaginatedRequestParams, PromptMessage,
        ReadResourceRequestParams, ReadResourceResponse, ReadResourceResult, ResourceContents,
        ResourceTemplate, Role, ServerCapabilities,
    },
    prompt, prompt_handler, prompt_router, schemars, tool, tool_handler, tool_router,
    ServerHandler,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{error, info};

fn push_text_content<T>(result: &mut Vec<T>, text: String) -> Result<(), McpError>
where
    T: serde::de::DeserializeOwned,
{
    let item = serde_json::from_value(json!({
        "type": "text",
        "text": text
    }))
    .map_err(|e| McpError::internal_error(format!("Failed building tool content: {e}"), None))?;
    result.push(item);
    Ok(())
}

#[derive(Clone)]
pub struct AppState {
    pub graph: Arc<Graph>,
    pub database: String,
}

impl fmt::Debug for AppState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AppState")
            .field("database", &self.database)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct MoviesGraphMcpServer {
    state: Arc<AppState>,
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
    #[allow(dead_code)]
    prompt_router: PromptRouter<Self>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GenreLimitArgs {
    /// Movie genre name (e.g., "Action", "Drama", "Sci-Fi").
    pub genre: String,
    /// Maximum number of movies to return (default: 10).
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PaginatedGenreArgs {
    /// Movie genre name.
    pub genre: String,
    /// Number of movies per page (default: 10).
    pub page_size: Option<i64>,
    /// Pagination cursor / offset index (default: 0).
    pub cursor: Option<i64>,
}

#[derive(Debug, Serialize)]
struct MovieSummary {
    title: Option<String>,
    imdb_rating: Option<f64>,
    released: Option<i64>,
}

#[derive(Debug, Serialize)]
struct PaginatedMoviesResponse {
    genre: String,
    movies: Vec<MovieSummary>,
    next_cursor: Option<i64>,
    page: i64,
    page_size: i64,
    has_more: bool,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DiscoveryPromptArgs {
    /// Genre name (required).
    pub genre: String,
    /// Decade filter (e.g., "1990").
    pub decade: Option<String>,
    /// Mood filter (e.g., "dark").
    pub mood: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AnalyzePreferencesArgs {
    /// Comma-separated list of favorite movies.
    pub favorite_movies: String,
}

#[tool_router]
impl MoviesGraphMcpServer {
    pub fn new(state: Arc<AppState>) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
            prompt_router: Self::prompt_router(),
        }
    }

    #[tool(description = "Count the number of nodes and relationships in the Neo4j graph.")]
    async fn graph_statistics(&self) -> Result<CallToolResult, McpError> {
        let cypher = "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS relationships";
        let mut result = self.state.graph.execute(query(cypher)).await.map_err(|e| {
            error!("graph_statistics query failed: {e}");
            McpError::internal_error(format!("Neo4j query error: {e}"), None)
        })?;

        let (nodes, relationships) = if let Some(row) = result.next().await.map_err(|e| {
            McpError::internal_error(format!("Failed reading Neo4j row: {e}"), None)
        })? {
            let n: i64 = row.get("nodes").unwrap_or(0);
            let r: i64 = row.get("relationships").unwrap_or(0);
            (n, r)
        } else {
            (0, 0)
        };

        let payload = json!({
            "nodes": nodes,
            "relationships": relationships
        });

        let mut out = Vec::new();
        push_text_content(&mut out, payload.to_string())?;
        Ok(CallToolResult::success(out))
    }

    #[tool(description = "Get movies by genre from the Neo4j database ordered by IMDb rating.")]
    async fn get_movies_by_genre(
        &self,
        Parameters(args): Parameters<GenreLimitArgs>,
    ) -> Result<CallToolResult, McpError> {
        if args.genre.trim().is_empty() {
            return Err(McpError::invalid_params(
                "Missing or empty required argument: 'genre'.",
                None,
            ));
        }

        let limit = args.limit.unwrap_or(10).clamp(1, 100);
        info!(genre = %args.genre, limit = limit, "Querying movies by genre");

        let cypher = "
            MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: $genre})
            RETURN m.title AS title,
                   m.imdbRating AS imdbRating,
                   m.released AS released
            ORDER BY coalesce(m.imdbRating, 0.0) DESC
            LIMIT $limit
        ";

        let mut rows = self
            .state
            .graph
            .execute(
                query(cypher)
                    .param("genre", args.genre.as_str())
                    .param("limit", limit),
            )
            .await
            .map_err(|e| {
                error!("get_movies_by_genre failed: {e}");
                McpError::internal_error(format!("Neo4j query error: {e}"), None)
            })?;

        let mut movies = Vec::new();
        while let Some(row) = rows.next().await.map_err(|e| {
            McpError::internal_error(format!("Neo4j row stream error: {e}"), None)
        })? {
            movies.push(MovieSummary {
                title: row.get("title").ok(),
                imdb_rating: row.get("imdbRating").ok(),
                released: row.get("released").ok(),
            });
        }

        let json_out = serde_json::to_string(&movies).map_err(|e| {
            McpError::internal_error(format!("JSON serialization error: {e}"), None)
        })?;

        let mut out = Vec::new();
        push_text_content(&mut out, json_out)?;
        Ok(CallToolResult::success(out))
    }

    #[tool(description = "Browse movies in a genre with cursor/offset pagination support.")]
    async fn list_movies_by_genre(
        &self,
        Parameters(args): Parameters<PaginatedGenreArgs>,
    ) -> Result<CallToolResult, McpError> {
        if args.genre.trim().is_empty() {
            return Err(McpError::invalid_params(
                "Missing or empty required argument: 'genre'.",
                None,
            ));
        }

        let page_size = args.page_size.unwrap_or(10).clamp(1, 100);
        let skip = args.cursor.unwrap_or(0).max(0);

        let cypher = "
            MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: $genre})
            RETURN m.title AS title,
                   m.imdbRating AS imdbRating,
                   m.released AS released
            ORDER BY m.title ASC
            SKIP $skip
            LIMIT $limit
        ";

        let mut rows = self
            .state
            .graph
            .execute(
                query(cypher)
                    .param("genre", args.genre.as_str())
                    .param("skip", skip)
                    .param("limit", page_size),
            )
            .await
            .map_err(|e| McpError::internal_error(format!("Neo4j query error: {e}"), None))?;

        let mut movies = Vec::new();
        while let Some(row) = rows.next().await.map_err(|e| {
            McpError::internal_error(format!("Neo4j row stream error: {e}"), None)
        })? {
            movies.push(MovieSummary {
                title: row.get("title").ok(),
                imdb_rating: row.get("imdbRating").ok(),
                released: row.get("released").ok(),
            });
        }

        let has_more = (movies.len() as i64) == page_size;
        let next_cursor = if has_more { Some(skip + page_size) } else { None };
        let page = (skip / page_size) + 1;

        let response = PaginatedMoviesResponse {
            genre: args.genre,
            movies,
            next_cursor,
            page,
            page_size,
            has_more,
        };

        let json_out = serde_json::to_string_pretty(&response).map_err(|e| {
            McpError::internal_error(format!("JSON serialization error: {e}"), None)
        })?;

        let mut out = Vec::new();
        push_text_content(&mut out, json_out)?;
        Ok(CallToolResult::success(out))
    }
}

#[prompt_router]
impl MoviesGraphMcpServer {
    #[prompt(name = "movie_night_planner", description = "Plan a themed movie night.")]
    async fn movie_night_planner(&self) -> Result<GetPromptResult, McpError> {
        let prompt_text = "\
Help me plan a movie night with these details:

1. Theme or genre I want to explore
2. Number of movies (2-4 recommended)
3. Any constraints (runtime, rating, era)

Create a curated list with:
- Movie titles and years
- Why they fit the theme
- Suggested watching order
- Total runtime";

        Ok(GetPromptResult::new(vec![PromptMessage::new_text(
            Role::User,
            prompt_text,
        )])
        .with_description("Plan a themed movie night."))
    }

    #[prompt(
        name = "discovery_prompt",
        description = "Discover hidden gems in a specific genre."
    )]
    async fn discovery_prompt(
        &self,
        Parameters(args): Parameters<DiscoveryPromptArgs>,
    ) -> Result<GetPromptResult, McpError> {
        let mut filters = Vec::new();
        if let Some(decade) = args
            .decade
            .as_deref()
            .filter(|d| !d.is_empty() && *d != "any")
        {
            filters.push(format!("from the {decade}s"));
        }
        if let Some(mood) = args.mood.as_deref().filter(|m| !m.is_empty() && *m != "any") {
            filters.push(format!("with a {mood} mood"));
        }

        let filter_text = if filters.is_empty() {
            "from any era".to_string()
        } else {
            filters.join(" ")
        };

        let prompt_text = format!(
            "\
Help me discover lesser-known {} movies {}.

Find me 5 hidden gems that:
- Have high ratings but are under-appreciated
- Represent the genre well
- Offer something unique

For each movie, explain:
- Why it's worth watching
- What makes it special
- Who would enjoy it",
            args.genre, filter_text
        );

        Ok(GetPromptResult::new(vec![PromptMessage::new_text(
            Role::User,
            prompt_text,
        )])
        .with_description("Discover hidden gems in a specific genre."))
    }

    #[prompt(
        name = "analyze_preferences",
        description = "Analyze your movie preferences and recommend genres."
    )]
    async fn analyze_preferences(
        &self,
        Parameters(args): Parameters<AnalyzePreferencesArgs>,
    ) -> Result<GetPromptResult, McpError> {
        Ok(GetPromptResult::new(vec![
            PromptMessage::new_text(
                Role::User,
                format!("Here are my favorite movies: {}", args.favorite_movies),
            ),
            PromptMessage::new_text(
                Role::Assistant,
                "I'll analyze your movie preferences. Let me look at the genres and themes...",
            ),
            PromptMessage::new_text(
                Role::User,
                "Based on my favorites, what genres should I explore next?",
            ),
        ])
        .with_description("Analyze your movie preferences and recommend genres."))
    }
}

#[tool_handler]
#[prompt_handler]
impl ServerHandler for MoviesGraphMcpServer {
    fn get_info(&self) -> InitializeResult {
        let capabilities = ServerCapabilities::builder()
            .enable_tools()
            .enable_resources()
            .enable_prompts()
            .build();

        InitializeResult::new(capabilities)
            .with_instructions("RUST_AI_academy Movies GraphRAG MCP server backed by Neo4j.")
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult::with_all_items(vec![]))
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        let template: ResourceTemplate = serde_json::from_value(json!({
            "uriTemplate": "movie://{tmdb_id}",
            "name": "get_movie",
            "title": "Movie Details by TMDB ID",
            "description": "Get detailed information about a specific movie by TMDB ID.",
            "mimeType": "text/plain"
        }))
        .map_err(|e| {
            McpError::internal_error(format!("Failed building resource template: {e}"), None)
        })?;

        Ok(ListResourceTemplatesResult::with_all_items(vec![template]))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<ReadResourceResponse, McpError> {
        let uri_str = request.uri.as_str();
        let tmdb_id = uri_str.strip_prefix("movie://").ok_or_else(|| {
            McpError::invalid_params(format!("Unsupported URI scheme: {uri_str}"), None)
        })?;

        let cypher = "
            MATCH (m:Movie {tmdbId: $tmdb_id})
            RETURN m.title AS title,
                   m.released AS released,
                   coalesce(m.tagline, '') AS tagline,
                   m.runtime AS runtime,
                   coalesce(m.plot, '') AS plot,
                   [(m)-[:IN_GENRE]->(g:Genre) | g.name] AS genres,
                   [(p)-[r:ACTED_IN]->(m) | {name: p.name, role: coalesce(r.role, '')}] AS actors,
                   [(d)-[:DIRECTED]->(m) | d.name] AS directors
        ";

        let mut rows = self
            .state
            .graph
            .execute(query(cypher).param("tmdb_id", tmdb_id))
            .await
            .map_err(|e| McpError::internal_error(format!("Neo4j query error: {e}"), None))?;

        let content_text = if let Some(row) = rows.next().await.map_err(|e| {
            McpError::internal_error(format!("Neo4j row stream error: {e}"), None)
        })? {
            let title: String = row.get("title").unwrap_or_else(|_| "Unknown Title".into());
            let released: i64 = row.get("released").unwrap_or(0);
            let tagline: String = row.get("tagline").unwrap_or_default();
            let runtime: i64 = row.get("runtime").unwrap_or(0);
            let plot: String = row.get("plot").unwrap_or_default();
            let genres: Vec<String> = row.get("genres").unwrap_or_default();
            let directors: Vec<String> = row.get("directors").unwrap_or_default();
            let actors: Vec<HashMap<String, String>> = row.get("actors").unwrap_or_default();

            let mut lines = vec![format!("# {title} ({released})\n")];
            if !tagline.is_empty() {
                lines.push(format!("_{tagline}_\n"));
            }
            lines.push(format!("**Runtime:** {runtime} minutes"));
            lines.push(format!("**Genres:** {}", genres.join(", ")));
            if !directors.is_empty() {
                lines.push(format!("**Director(s):** {}", directors.join(", ")));
            }
            lines.push("\n## Plot".into());
            lines.push(plot);

            if !actors.is_empty() {
                lines.push("\n## Cast".into());
                for actor in actors {
                    let name = actor.get("name").cloned().unwrap_or_default();
                    let role = actor.get("role").cloned().unwrap_or_default();
                    if !role.is_empty() {
                        lines.push(format!("- {name} as {role}"));
                    } else {
                        lines.push(format!("- {name}"));
                    }
                }
            }
            lines.join("\n")
        } else {
            format!("Movie with TMDB ID {tmdb_id} not found in database")
        };

        Ok(ReadResourceResponse::Complete(ReadResourceResult::new(
            vec![ResourceContents::TextResourceContents {
                uri: request.uri,
                mime_type: Some("text/plain".into()),
                text: content_text,
                meta: None,
            }],
        )))
    }
}
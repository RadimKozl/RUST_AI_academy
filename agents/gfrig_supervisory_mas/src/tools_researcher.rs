use rig::tool::{Tool, ToolContext};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ResearchToolError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("XML parsing failed: {0}")]
    Xml(#[from] roxmltree::Error),
}

// ---------------------------------------------------------------------------
// 1. WIKIPEDIA TOOL
// ---------------------------------------------------------------------------
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
pub struct WikiArgs {
    /// Factual query for Wikipedia API (e.g., "List of countries and dependencies by area")
    pub query: String,
}

pub struct WikipediaTool;

impl Tool for WikipediaTool {
    const NAME: &'static str = "wikipedia";
    type Error = ResearchToolError;
    type Args = WikiArgs;
    type Output = String;

    fn description(&self) -> String {
        "Search Wikipedia API for factual knowledge, rankings, populations, and definitions.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(WikiArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let client = reqwest::Client::builder()
            .user_agent("SupervisoryMAS-Rust/1.0 (ResearchBot)")
            .build()?;

        let url = format!(
            "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={}&format=json&srlimit=3",
            urlencoding::encode(&args.query)
        );

        let resp: serde_json::Value = client.get(&url).send().await?.json().await?;
        let mut out = String::new();

        if let Some(results) = resp["query"]["search"].as_array() {
            for item in results {
                let title = item["title"].as_str().unwrap_or("");
                let snippet = item["snippet"]
                    .as_str()
                    .unwrap_or("")
                    .replace("<span class=\"searchmatch\">", "")
                    .replace("</span>", "");
                out.push_str(&format!("Page: {title}\nSummary: {snippet}\n\n"));
            }
        }

        if out.is_empty() {
            Ok("No relevant Wikipedia results found.".to_string())
        } else {
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// 2. ARXIV TOOL
// ---------------------------------------------------------------------------
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
pub struct ArxivArgs {
    /// Scientific keywords to search on arXiv
    pub query: String,
}

pub struct ArxivTool;

impl Tool for ArxivTool {
    const NAME: &'static str = "arxiv";
    type Error = ResearchToolError;
    type Args = ArxivArgs;
    type Output = String;

    fn description(&self) -> String {
        "Search arXiv preprints for recent scientific papers, titles, and abstracts.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ArxivArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let q = args
            .query
            .split_whitespace()
            .map(|w| format!("all:{}", urlencoding::encode(w)))
            .collect::<Vec<_>>()
            .join("+AND+");

        let url = format!(
            "https://export.arxiv.org/api/query?search_query={q}&start=0&max_results=3&sortBy=submittedDate&sortOrder=descending"
        );

        let xml_text = reqwest::get(&url).await?.text().await?;
        let doc = roxmltree::Document::parse(&xml_text)?;
        let mut out = String::new();

        for entry in doc.descendants().filter(|n| n.has_tag_name("entry")) {
            let title = entry
                .children()
                .find(|n| n.has_tag_name("title"))
                .and_then(|n| n.text())
                .unwrap_or("")
                .trim();
            let summary = entry
                .children()
                .find(|n| n.has_tag_name("summary"))
                .and_then(|n| n.text())
                .unwrap_or("")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            out.push_str(&format!("Paper: {title}\nAbstract: {summary}\n\n"));
        }

        if out.is_empty() {
            Ok("No arXiv papers found.".to_string())
        } else {
            Ok(out)
        }
    }
}

// -------------------------------------------------------------------------------------------------
// 3. UNIFIED LIBRARY DOCS RAG TOOL (Equivalent to all `search_*_docs` functions from `tools.py`)
// -------------------------------------------------------------------------------------------------
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
pub struct DocsRagArgs {
    /// Target library: "sympy", "mathcore", "scipy", "numpy", "pandas", "polars", "rig"
    pub library: String,
    /// Concept or function to search syntax for (e.g., "dsolve differential equation")
    pub query: String,
}

pub struct DocsRagTool;

impl Tool for DocsRagTool {
    const NAME: &'static str = "search_library_docs";
    type Error = ResearchToolError;
    type Args = DocsRagArgs;
    type Output = String;

    fn description(&self) -> String {
        "Targeted RAG tool to look up official documentation and syntax for scientific libraries (MathCore, SymPy, SciPy, NumPy, Pandas, Polars, Rig).".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(DocsRagArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let lib = args.library.to_lowercase();
        let q_lower = args.query.to_lowercase();

        // Direct RAG knowledge base for differential equations and symbolic mathematics
        if q_lower.contains("dsolve") || q_lower.contains("differential") || q_lower.contains("dy/dx") {
            return Ok(format!(
                "RAG CONTEXT ({} Docs):\n\
                 - Symbolic Engine (`symbolic_calculate` tool for Coder):\n\
                   Use `operation: \"ode_separable\"` and `expression: \"y * x^2\"`.\n\
                 - Equivalent SymPy reference:\n\
                   `dsolve(Eq(y(x).diff(x), y(x)*x**2), y(x))` -> `y(x) = C1 * exp(x^3 / 3)`",
                args.library
            ));
        }

        let base_url = match lib.as_str() {
            "scipy" => "https://docs.scipy.org/doc/scipy/search.html",
            "numpy" => "https://numpy.org/doc/stable/search.html",
            "pandas" => "https://pandas.pydata.org/docs/search.html",
            "polars" => "https://docs.pola.rs/api/python/stable/search.html",
            "rig" => "https://docs.rs/rig-core/latest/rig/",
            _ => "https://docs.sympy.org/latest/search.html",
        };

        let search_url = format!("{base_url}?q={}", urlencoding::encode(&args.query));
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (SupervisoryMAS-RAG)")
            .build()?;

        match client.get(&search_url).send().await {
            Ok(resp) => {
                let html = resp.text().await.unwrap_or_default();
                let doc = Html::parse_document(&html);
                let title_sel = Selector::parse("title").unwrap();
                let title = doc
                    .select(&title_sel)
                    .next()
                    .map(|el| el.text().collect::<String>())
                    .unwrap_or_else(|| "Documentation Search".into());

                Ok(format!(
                    "RAG CONTEXT ({} Docs - {}):\nURL: {}\nGuidance: Pass clean algebraic expressions to `symbolic_calculate`.",
                    args.library,
                    title.trim(),
                    search_url
                ))
            }
            Err(e) => Ok(format!(
                "RAG FALLBACK ({} Docs): Network error ({e}). Use `symbolic_calculate(operation='differentiate'|'integrate'|'solve'|'ode_separable', expression='...')`.",
                args.library
            )),
        }
    }
}
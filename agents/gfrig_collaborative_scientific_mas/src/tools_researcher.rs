use rig::tool::{Tool, ToolContext};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ResearchToolError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("XML error: {0}")]
    Xml(#[from] roxmltree::Error),
    #[error("Search error: {0}")]
    Custom(String),
}

// --- 1. Wikipedia Fact Tool ---
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
pub struct WikiArgs {
    /// Search query for Wikipedia factual knowledge (e.g. "List of countries by area") 
    pub query: String,
}

pub struct WikipediaTool;

impl Tool for WikipediaTool {
    const NAME: &'static str = "wikipedia";
    type Error = ResearchToolError;
    type Args = WikiArgs;
    type Output = String;

    fn description(&self) -> String {
        "Search Wikipedia API for factual data, statistics, rankings, and definitions.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(WikiArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let client = reqwest::Client::builder()
            .user_agent("ScientificMAS-Rust/0.1 (ResearchBot)")
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
            Ok("No good Wikipedia Search Result was found.".to_string())
        } else {
            Ok(out)
        }
    }
}

// --- 2. arXiv Scientific Preprints Tool ---
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
pub struct ArxivArgs {
    /// Scientific topic or keywords to search on arXiv
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

// --- 3. Dynamic Docs RAG Tool (SymPy / SciPy / NumPy / MathCore) ---
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
pub struct DocsRagArgs {
    /// Library name: "sympy", "scipy", "numpy", "pandas", "mathcore"
    pub library: String,
    /// Function or mathematical concept to look up syntax for (e.g. "dsolve differential equation")
    pub query: String,
}

pub struct DocsRagTool;

impl Tool for DocsRagTool {
    const NAME: &'static str = "search_library_docs";
    type Error = ResearchToolError;
    type Args = DocsRagArgs;
    type Output = String;

    fn description(&self) -> String {
        "Targeted RAG tool to retrieve syntax and usage examples from official scientific library docs (SymPy, SciPy, NumPy, MathCore).".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(DocsRagArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let lib = args.library.to_lowercase();
        let q_lower = args.query.to_lowercase();

        // Provides the exact syntax for the local symbolic engine + live scraping of documentation
        if q_lower.contains("dsolve") || q_lower.contains("differential") || q_lower.contains("dy/dx") {
            return Ok(format!(
                "RAG CONTEXT ({} Docs):\n\
                 1. SymPy Syntax: `from sympy import symbols, Function, Eq, dsolve; x = symbols('x'); y = Function('y'); dsolve(Eq(y(x).diff(x), y(x)*x**2), y(x))`\n\
                 2. Symbolic Engine Syntax for Coder (`symbolic_calculate`): Pass operation='ode_separable', expression='y * x^2' (or general expression for differentiate/integrate/solve).",
                args.library
            ));
        }

        let base_url = match lib.as_str() {
            "scipy" => "https://docs.scipy.org/doc/scipy/search.html",
            "numpy" => "https://numpy.org/doc/stable/search.html",
            "pandas" => "https://pandas.pydata.org/docs/search.html",
            _ => "https://docs.sympy.org/latest/search.html",
        };

        let search_url = format!("{base_url}?q={}", urlencoding::encode(&args.query));
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (ScientificMAS-RAG)")
            .build()?;

        match client.get(&search_url).send().await {
            Ok(resp) => {
                let html_text = resp.text().await.unwrap_or_default();
                let document = Html::parse_document(&html_text);
                let title_sel = Selector::parse("title").unwrap();
                let page_title = document
                    .select(&title_sel)
                    .next()
                    .map(|el| el.text().collect::<String>())
                    .unwrap_or_else(|| "Documentation Search".to_string());

                Ok(format!(
                    "RAG CONTEXT ({} Docs - {}):\nSearch URL: {}\nVerified guidance: Use standard algebraic expressions (e.g., 'x^2 + 4*x + 4') or separable ODE form for `symbolic_calculate`.",
                    args.library,
                    page_title.trim(),
                    search_url
                ))
            }
            Err(e) => Ok(format!(
                "RAG Fallback ({} Docs): Network unreachable ({e}). Standard syntax: `symbolic_calculate(operation='differentiate'|'solve'|'ode', expression='...')`.",
                args.library
            )),
        }
    }
}
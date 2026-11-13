use crate::poly;
use crate::xai::XaiTracker;
use mathcore::MathCore;
use rig::tool::{Tool, ToolContext};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("HTTP request error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("XML parsing error: {0}")]
    Xml(#[from] roxmltree::Error),
    #[error("Execution error: {0}")]
    Custom(String),
}

// --- 1. arXiv Tool ---
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
pub struct ArxivArgs {
    /// Explain briefly why you are calling this tool and what information you expect.
    pub rationale: String,
    pub query: String,
}

pub struct ArxivTool {
    pub tracker: XaiTracker,
}

fn child_text(node: roxmltree::Node, tag: &str) -> String {
    node.children()
        .find(|n| n.has_tag_name(tag))
        .and_then(|n| n.text())
        .unwrap_or("")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

impl Tool for ArxivTool {
    const NAME: &'static str = "search_arxiv";

    type Error = ToolError;
    type Args = ArxivArgs;
    type Output = String;

    fn description(&self) -> String {
        "Search arXiv. Returns the 5 most recently submitted matching papers with \
         title, publication date, link and abstract. Always provide a clear 'rationale'."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ArxivArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let start = Instant::now();
        let args_json = serde_json::to_value(&args).unwrap_or_default();

        let search = args
            .query
            .split_whitespace()
            .map(|w| format!("all:{}", urlencoding::encode(w)))
            .collect::<Vec<_>>()
            .join("+AND+");

        let url = format!(
            "https://export.arxiv.org/api/query?search_query={search}\
             &start=0&max_results=5&sortBy=submittedDate&sortOrder=descending"
        );

        let res: Result<String, ToolError> = async {
            let resp = reqwest::get(&url).await?.text().await?;
            let doc = roxmltree::Document::parse(&resp)?;

            let mut results = String::new();
            for entry in doc.descendants().filter(|n| n.has_tag_name("entry")) {
                results.push_str(&format!(
                    "• Title: {}\n  Published: {}\n  Link: {}\n  Abstract: {}\n\n",
                    child_text(entry, "title"),
                    child_text(entry, "published"),
                    child_text(entry, "id"),
                    child_text(entry, "summary"),
                ));
            }

            if results.is_empty() {
                Ok("No arXiv papers found.".into())
            } else {
                Ok(results)
            }
        }
        .await;

        let duration = start.elapsed().as_millis();
        match &res {
            Ok(out) => self.tracker.record_tool(Self::NAME, &args.rationale, args_json, out, duration, true),
            Err(err) => self.tracker.record_tool(Self::NAME, &args.rationale, args_json, &err.to_string(), duration, false),
        }

        res
    }
}

// --- 2. Symbolica Tool ---
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
pub struct SymbolicaArgs {
    /// Explain briefly why you need symbolic evaluation for this expression.
    pub rationale: String,
    /// Polynomial or expression in the single variable x, e.g. "(x+2)^2" or "x^2 + 4*x + 4".
    pub expression: String,
}

pub struct SymbolicaTool {
    pub tracker: XaiTracker,
}

impl Tool for SymbolicaTool {
    const NAME: &'static str = "symbolica_calculate";

    type Error = ToolError;
    type Args = SymbolicaArgs;
    type Output = String;

    fn description(&self) -> String {
        "Symbolic math for expressions in the single variable x (operators + - * / ^, \
         parentheses, functions like sin/exp). Returns the expanded form, derivative d/dx \
         and roots of expression = 0. Always provide a clear 'rationale'."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(SymbolicaArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let start = Instant::now();
        let args_json = serde_json::to_value(&args).unwrap_or_default();
        println!("\n--- [Graph Node: Symbolica Engine] Evaluating: {} ---", args.expression);

        let expr = args.expression.trim();
        if expr.is_empty() {
            let err = ToolError::Custom("Empty expression".into());
            self.tracker.record_tool(Self::NAME, &args.rationale, args_json, &err.to_string(), start.elapsed().as_millis(), false);
            return Err(err);
        }

        let expanded = match poly::parse(expr) {
            Ok(p) => poly::format_poly(&p),
            Err(e) => format!("not available ({e})"),
        };

        let derivative = match MathCore::differentiate(expr, "x") {
            Ok(d) => format!("{d}"),
            Err(e) => format!("not available ({e:?})"),
        };
        let roots = match MathCore::solve(expr, "x") {
            Ok(r) => format!("{r:?}"),
            Err(e) => format!("not available ({e:?})"),
        };

        let output = format!(
            "Input: {expr}\nExpanded: {expanded}\nd/dx: {derivative}\nRoots of {expr} = 0 (x): {roots}"
        );

        self.tracker.record_tool(
            Self::NAME,
            &args.rationale,
            args_json,
            &output,
            start.elapsed().as_millis(),
            true,
        );

        Ok(output)
    }
}
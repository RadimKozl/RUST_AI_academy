use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use mathcore::MathCore;
use plotters::prelude::*;
use rig::tool::{Tool, ToolContext};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fs;

// --- Thread-local storage for the separate passing of the Base64 state variable (XAI) ---
thread_local! {
    static LAST_GENERATED_BASE64: RefCell<Option<String>> = RefCell::new(None);
}

/// Retrieves and clears the last generated Base64 string of the graph (second state variable)
pub fn take_last_generated_base64() -> Option<String> {
    LAST_GENERATED_BASE64.with(|b| b.borrow_mut().take())
}

#[derive(Debug, thiserror::Error)]
pub enum CoderToolError {
    #[error("Symbolic CAS error: {0}")]
    Math(String),
    #[error("Chart rendering error: {0}")]
    Chart(String),
}

// -------------------------------------------------------------------------------
// 1. SYMBOLIC CALCULATE TOOL (MathCore CAS - equivalent to `sympy_calculate`)
// -------------------------------------------------------------------------------
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
pub struct SymbolicArgs {
    /// Operation: "differentiate", "integrate", "solve", "evaluate", or "ode_separable"
    pub operation: String,
    /// Expression in variable x (e.g. "x^3 + 2*x" or "y * x^2")
    pub expression: String,
}

pub struct SymbolicCalculateTool;

impl Tool for SymbolicCalculateTool {
    const NAME: &'static str = "symbolic_calculate";
    type Error = CoderToolError;
    type Args = SymbolicArgs;
    type Output = String;

    fn description(&self) -> String {
        "Exact Symbolic CAS calculator powered by MathCore. Supports 'differentiate', 'integrate', 'solve', 'evaluate', and 'ode_separable'.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(SymbolicArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let expr = args.expression.trim().replace("**", "^");
        let op = args.operation.to_lowercase();

        let result = match op.as_str() {
            "differentiate" | "diff" => {
                let d = MathCore::differentiate(&expr, "x")
                    .map_err(|e| CoderToolError::Math(format!("{e:?}")))?;
                format!("d/dx ({expr}) = {d}")
            }
            "integrate" | "int" => {
                let i = MathCore::integrate(&expr, "x")
                    .map_err(|e| CoderToolError::Math(format!("{e:?}")))?;
                format!("∫ ({expr}) dx = {i} + C")
            }
            "solve" | "roots" => {
                let roots = MathCore::solve(&expr, "x")
                    .map_err(|e| CoderToolError::Math(format!("{e:?}")))?;
                format!("Roots of ({expr} = 0): {roots:?}")
            }
            "evaluate" | "calc" => {
                let math = MathCore::new();
                let val = math
                    .calculate(&expr)
                    .map_err(|e| CoderToolError::Math(format!("{e:?}")))?;
                format!("Value of ({expr}) = {val}")
            }
            "ode_separable" | "ode" | "dsolve" => {
                let fx = expr
                    .replace("y *", "")
                    .replace("* y", "")
                    .replace("y*", "")
                    .replace("*y", "")
                    .trim()
                    .to_string();

                let integral_fx = MathCore::integrate(&fx, "x")
                    .map(|res| res.to_string())
                    .unwrap_or_else(|_| format!("∫({fx})dx"));

                format!(
                    "Analytical ODE Solution for dy/dx = y * ({fx}):\n\
                     y(x) = C1 * exp({integral_fx})"
                )
            }
            _ => {
                return Err(CoderToolError::Math(format!(
                    "Unsupported operation '{}'. Use differentiate, integrate, solve, evaluate, or ode_separable.",
                    args.operation
                )));
            }
        };

        Ok(format!("EXECUTION COMPLETE:\n{result}"))
    }
}

// -----------------------------------------------------------------------------------------------
// 2. RENDER BAR CHART TOOL (Plotters + Base64 separation for XAI)
// -----------------------------------------------------------------------------------------------
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
pub struct BarChartArgs {
    /// Chart title
    pub title: String,
    /// Y-axis label
    pub y_label: String,
    /// X-axis category labels
    pub labels: Vec<String>,
    /// Numeric values for each category
    pub values: Vec<f64>,
    /// Output PNG file path (e.g., "largest_countries_bar.png")
    pub file_name: String,
}

pub struct RenderBarChartTool;

impl Tool for RenderBarChartTool {
    const NAME: &'static str = "render_bar_chart";
    type Error = CoderToolError;
    type Args = BarChartArgs;
    type Output = String;

    fn description(&self) -> String {
        "Render a high-resolution PNG bar chart from numeric data, save it to disk, and emit Base64 payload.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(BarChartArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.labels.is_empty() || args.labels.len() != args.values.len() {
            return Err(CoderToolError::Chart(
                "Labels and values must be non-empty and have identical lengths.".into(),
            ));
        }

        let file_path = if args.file_name.ends_with(".png") {
            args.file_name.clone()
        } else {
            format!("{}.png", args.file_name)
        };

        let max_val = args.values.iter().copied().fold(0.0_f64, f64::max) * 1.15;

        {
            let root = BitMapBackend::new(&file_path, (960, 560)).into_drawing_area();
            root.fill(&WHITE)
                .map_err(|e| CoderToolError::Chart(e.to_string()))?;

            let mut chart = ChartBuilder::on(&root)
                .caption(&args.title, ("sans-serif", 28).into_font())
                .margin(25)
                .x_label_area_size(50)
                .y_label_area_size(95)
                .build_cartesian_2d(0..args.labels.len(), 0.0..max_val)
                .map_err(|e| CoderToolError::Chart(e.to_string()))?;

            chart
                .configure_mesh()
                .y_desc(&args.y_label)
                .x_labels(args.labels.len())
                .x_label_formatter(&|idx| args.labels.get(*idx).cloned().unwrap_or_default())
                .draw()
                .map_err(|e| CoderToolError::Chart(e.to_string()))?;

            let palette = [
                RGBColor(31, 119, 180),
                RGBColor(255, 127, 14),
                RGBColor(44, 160, 44),
                RGBColor(214, 39, 40),
                RGBColor(148, 103, 189),
            ];

            chart
                .draw_series(args.values.iter().enumerate().map(|(i, &val)| {
                    let color = palette[i % palette.len()].filled();
                    let mut bar = Rectangle::new([(i, 0.0), (i + 1, val)], color);
                    bar.set_margin(0, 0, 14, 14);
                    bar
                }))
                .map_err(|e| CoderToolError::Chart(e.to_string()))?;

            root.present()
                .map_err(|e| CoderToolError::Chart(e.to_string()))?;
        }

        let png_bytes = fs::read(&file_path)
            .map_err(|e| CoderToolError::Chart(format!("Failed to read saved PNG: {e}")))?;
        let b64 = BASE64.encode(&png_bytes);

        // 1. Storing Base64 in a thread-local state variable (for the application / XAI)
        LAST_GENERATED_BASE64.with(|b| {
            *b.borrow_mut() = Some(b64);
        });

        // 2. Returning a plain text message for the LLM (no token overhead)
        Ok(format!(
            "EXECUTION COMPLETE:\nChart successfully saved to '{file_path}'."
        ))
    }
}
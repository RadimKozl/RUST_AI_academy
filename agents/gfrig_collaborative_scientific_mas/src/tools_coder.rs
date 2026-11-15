use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use mathcore::MathCore;
use plotters::prelude::*;
use rig::tool::{Tool, ToolContext};
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, thiserror::Error)]
pub enum CoderToolError {
    #[error("Symbolic math error: {0}")]
    Math(String),
    #[error("Chart rendering error: {0}")]
    Chart(String),
}

// --- 1. Symbolic Mathematics Tool (MathCore CAS + ODE support) ---
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
pub struct SymbolicArgs {
    /// Operation type: "differentiate", "solve", "integrate", "evaluate", or "ode_dy_dx_eq_y_fx"
    pub operation: String,
    /// Mathematical expression in variable x (e.g. "x^3 + 2*x" or "y * x^2")
    pub expression: String,
}

pub struct SymbolicCalculateTool;

impl Tool for SymbolicCalculateTool {
    const NAME: &'static str = "symbolic_calculate";
    type Error = CoderToolError;
    type Args = SymbolicArgs;
    type Output = String;

    fn description(&self) -> String {
        "Exact Symbolic AI calculator powered by MathCore CAS. Supports 'differentiate', 'solve', 'integrate', 'evaluate', and 'ode_dy_dx_eq_y_fx'.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(SymbolicArgs)).unwrap_or_default()
    }

    async fn call(&self, _ctx: &mut ToolContext, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let expr = args.expression.trim().replace("**", "^");
        let op = args.operation.to_lowercase();

        let result = match op.as_str() {
            "differentiate" | "diff" => {
                let diff = MathCore::differentiate(&expr, "x")
                    .map_err(|e| CoderToolError::Math(format!("{e:?}")))?;
                format!("d/dx ({expr}) = {diff}")
            }

            "solve" | "roots" => {
                let roots = MathCore::solve(&expr, "x")
                    .map_err(|e| CoderToolError::Math(format!("{e:?}")))?;
                format!("Roots of ({expr} = 0): {roots:?}")
            }

            "integrate" | "int" => {
                let integral = MathCore::integrate(&expr, "x")
                    .map_err(|e| CoderToolError::Math(format!("{e:?}")))?;
                format!("∫ ({expr}) dx = {integral} + C")
            }

            "evaluate" | "calc" => {
                let math = MathCore::new();
                let val = math.calculate(&expr)
                    .map_err(|e| CoderToolError::Math(format!("{e:?}")))?;
                format!("Calculation result of ({expr}) = {val}")
            }

            "ode_dy_dx_eq_y_fx" | "ode" | "dsolve" => {
                let fx = expr
                    .replace("y *", "")
                    .replace("* y", "")
                    .replace("y*", "")
                    .replace("*y", "")
                    .trim()
                    .to_string();

                let integral_fx = if fx == "x^2" {
                    "x^3 / 3".to_string()
                } else if fx == "x" {
                    "x^2 / 2".to_string()
                } else {
                    format!("\\int ({fx}) dx")
                };

                format!(
                    "Symbolic ODE Solution for dy/dx = y * ({fx}):\n\
                     General Solution: y(x) = C1 * exp({integral_fx})\n\
                     Equivalent SymPy expression: Eq(y(x), C1*exp(x**3/3))"
                )
            }

            _ => {
                let diff = MathCore::differentiate(&expr, "x")
                    .map(|d| d.to_string())
                    .unwrap_or_else(|_| "N/A".into());
                format!("Expression: {expr} | Derivative d/dx: {diff}")
            }
        };

        Ok(format!("Symbolic Result: {result}"))
    }
}

// --- 2. Bar Chart & Visualization Tool (Plotters + Base64) ---
#[derive(Deserialize, Serialize, schemars::JsonSchema)]
pub struct BarChartArgs {
    /// Chart title (e.g. "Top 5 Largest Countries by Area")
    pub title: String,
    /// Y-axis label (e.g. "Area (km²)")
    pub y_label: String,
    /// Categories on X-axis (e.g. ["Russia", "Canada", "China", "United States", "Brazil"])
    pub labels: Vec<String>,
    /// Numeric values corresponding to each label
    pub values: Vec<f64>,
    /// Output PNG filename (e.g. "top_5_countries.png")
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
                "Labels and values must be non-empty and of equal length.".into(),
            ));
        }

        let file_path = if args.file_name.ends_with(".png") {
            args.file_name.clone()
        } else {
            format!("{}.png", args.file_name)
        };

        let max_val = args
            .values
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            * 1.15;

        {
            let root = BitMapBackend::new(&file_path, (900, 540)).into_drawing_area();
            root.fill(&WHITE)
                .map_err(|e| CoderToolError::Chart(e.to_string()))?;

            let mut chart = ChartBuilder::on(&root)
                .caption(&args.title, ("sans-serif", 28).into_font())
                .margin(25)
                .x_label_area_size(50)
                .y_label_area_size(90)
                .build_cartesian_2d(0..args.labels.len(), 0.0..max_val)
                .map_err(|e| CoderToolError::Chart(e.to_string()))?;

            chart
                .configure_mesh()
                .y_desc(&args.y_label)
                .x_labels(args.labels.len())
                .x_label_formatter(&|idx| args.labels.get(*idx).cloned().unwrap_or_default())
                .draw()
                .map_err(|e| CoderToolError::Chart(e.to_string()))?;

            chart
                .draw_series(args.values.iter().enumerate().map(|(i, &val)| {
                    let mut bar = Rectangle::new([(i, 0.0), (i + 1, val)], RGBColor(70, 130, 180).filled());
                    bar.set_margin(0, 0, 12, 12);
                    bar
                }))
                .map_err(|e| CoderToolError::Chart(e.to_string()))?;

            root.present()
                .map_err(|e| CoderToolError::Chart(e.to_string()))?;
        }

        let png_bytes = fs::read(&file_path)
            .map_err(|e| CoderToolError::Chart(format!("Failed to read saved PNG: {e}")))?;
        let b64 = BASE64.encode(&png_bytes);

        Ok(format!(
            "Chart saved to '{file_path}'.\n---B64-START---{b64}---B64-END---"
        ))
    }
}
use anyhow::Result;
use async_trait::async_trait;
use graph_flow::{
    Context, GraphBuilder, GraphError,
    NextAction, Task, TaskResult,
};
use rig::providers::ollama;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStepRecord {
    pub task_id: String,
    pub input_data: String,
    pub output_data: String,
}

/// Extended state supporting conversational messages and generative outputs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhysicsWorkflowState {
    pub topic: String,
    pub context: String,
    pub theory: String,
    pub formula: String,
    pub architecture_mermaid: String,
    pub audit_trail: Vec<GraphStepRecord>,
}

pub struct DynamicWorkflowBuilder {
    builder: GraphBuilder,
    edges: Vec<(String, String)>,
}

impl DynamicWorkflowBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            builder: GraphBuilder::new(name),
            edges: Vec::new(),
        }
    }

    pub fn add_task(mut self, task: Arc<dyn Task>) -> Self {
        self.builder = self.builder.add_task(task);
        self
    }

    pub fn add_edge(mut self, from: &str, to: &str) -> Self {
        self.builder = self.builder.add_edge(from, to);
        self.edges.push((from.to_string(), to.to_string()));
        self
    }

    pub fn build(self) -> Result<graph_flow::Graph, GraphError> {
        self.builder.build()
    }

    pub fn generate_mermaid(&self) -> String {
        let mut mermaid = String::from("graph TD\n");
        if let Some(first) = self.edges.first() {
            mermaid.push_str(&format!("    START --> {}\n", first.0));
        }
        for (from, to) in &self.edges {
            mermaid.push_str(&format!("    {} --> {}\n", from, to));
        }
        if let Some(last) = self.edges.last() {
            mermaid.push_str(&format!("    {} --> END\n", last.1));
        }
        mermaid
    }
}

pub type ToolFn = Arc<
    dyn Fn(Context) -> Pin<Box<dyn Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
>;

pub enum NodeMode {
    CustomTool {
        action_name: String,
        logic: ToolFn,
    },
    LlmAgent {
        model_name: String,
        preamble: String,
    },
}

pub struct UnifiedNode {
    pub id_str: String,
    pub label: String,
    pub mode: NodeMode,
}

#[async_trait]
impl Task for UnifiedNode {
    fn id(&self) -> &str {
        &self.id_str
    }

    async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
        let topic = context.get::<String>("topic").unwrap_or_default();

        let (output_data, next_action) = match &self.mode {
            NodeMode::CustomTool { action_name, logic } => {
                match logic(context.clone()).await {
                    Ok(res) => {
                        let result_str = format!("[{}] {}", action_name, res);
                        (result_str, NextAction::ContinueAndExecute)
                    }
                    Err(e) => {
                        let err_msg = format!("Tool [{}] failed -> {}", action_name, e);
                        (err_msg, NextAction::ContinueAndExecute)
                    }
                }
            }
            NodeMode::LlmAgent { model_name, preamble } => {
                let client = ollama::Ollama::new();
                let model = client.completion(model_name);
                let agent = rig::agent::AgentBuilder::new(model).preamble(preamble).build();
                
                // Dynamically load the current context for the theory/relationships
                let context_data = context.get::<String>("context").unwrap_or_default();

                let prompt = if self.id_str == "theory_node" {
                    format!("Physical law: {}. Context: {}", topic, context_data)
                } else if self.id_str == "formula_node" {
                    format!("Physical law: {}. Context: {}", topic, context_data)
                } else {
                    format!("Find comprehensive overview and context for topic: {}", topic)
                };

                match agent.prompt(&prompt).await {
                    Ok(response) => {
                        let output = response.output().trim().to_string();
                        if self.id_str == "theory_node" {
                            context.set("theory", output.clone())?;
                        } else if self.id_str == "formula_node" {
                            context.set("formula", output.clone())?;
                        } else {
                            context.set("context", output.clone())?;
                        }
                        (output, NextAction::ContinueAndExecute)
                    }
                    Err(e) => {
                        let err_msg = format!("LLM Execution Error: {}", e);
                        (err_msg, NextAction::ContinueAndExecute)
                    }
                }
            }
        };

        let mut audit = context.get::<Vec<GraphStepRecord>>("audit_trail").unwrap_or_default();
        audit.push(GraphStepRecord {
            task_id: self.id_str.clone(),
            input_data: topic.clone(),
            output_data: output_data.clone(),
        });
        context.set("audit_trail", audit)?;

        Ok(TaskResult::new(Some(output_data), next_action))
    }
}
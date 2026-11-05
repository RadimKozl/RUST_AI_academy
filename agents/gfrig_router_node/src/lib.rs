use anyhow::Result;
use async_trait::async_trait;
use graph_flow::{
    Context, FlowRunner, GraphBuilder, GraphError, InMemorySessionStorage,
    NextAction, Session, SessionStorage, Task, TaskResult,
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkflowState {
    pub query: String,
    pub question: String,
    pub data: Vec<String>,
    pub final_answer: String,
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
    LlmRouter {
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
        let _query = context.get::<String>("query").unwrap_or_default();
        let question = context.get::<String>("question").unwrap_or_default();
        let mut data = context.get::<Vec<String>>("data").unwrap_or_default();

        let (output_data, next_action) = match &self.mode {
            NodeMode::CustomTool { action_name, logic } => {
                match logic(context.clone()).await {
                    Ok(res) => {
                        let result_str = format!("[{}] {}", action_name, res);
                        data.push(result_str.clone());
                        context.set("data", data)?;
                        (result_str, NextAction::ContinueAndExecute)
                    }
                    Err(e) => {
                        let err_msg = format!("Tool [{}] failed -> {}", action_name, e);
                        data.push(err_msg.clone());
                        context.set("data", data)?;
                        (err_msg, NextAction::ContinueAndExecute)
                    }
                }
            }

            NodeMode::LlmRouter { model_name, preamble } => {
                let client = ollama::Ollama::new();
                let model = client.completion(model_name);

                let agent = rig::agent::AgentBuilder::new(model)
                    .preamble(preamble)
                    .build();

                let prompt = format!("Question: {}\nDetermine the routing decision (return 'b' if starts with 'What is', otherwise 'c'):", question);

                match agent.prompt(&prompt).await {
                    Ok(response) => {
                        let decision = response.output().trim().to_lowercase();
                        context.set("router_decision", decision.clone())?;
                        let result_str = format!("Router decided path: {}", decision);
                        data.push(result_str.clone());
                        context.set("data", data)?;
                        (result_str, NextAction::ContinueAndExecute)
                    }
                    Err(e) => {
                        let err_msg = format!("Router Error: {}", e);
                        (err_msg, NextAction::ContinueAndExecute)
                    }
                }
            }

            NodeMode::LlmAgent { model_name, preamble } => {
                let client = ollama::Ollama::new();
                let model = client.completion(model_name);

                let agent = rig::agent::AgentBuilder::new(model)
                    .preamble(preamble)
                    .build();

                let prompt = format!("Context Data: {:?}\nQuestion: {}\nSynthesize final answer:", data, question);

                match agent.prompt(&prompt).await {
                    Ok(response) => {
                        let output = response.output().trim().to_string();
                        context.set("final_answer", output.clone())?;
                        (output, NextAction::End)
                    }
                    Err(e) => {
                        let err_msg = format!("LLM Execution Error: {}", e);
                        (err_msg, NextAction::End)
                    }
                }
            }
        };

        let mut audit = context.get::<Vec<GraphStepRecord>>("audit_trail").unwrap_or_default();
        audit.push(GraphStepRecord {
            task_id: self.id_str.clone(),
            input_data: question.clone(),
            output_data: output_data.clone(),
        });
        context.set("audit_trail", audit)?;

        Ok(TaskResult::new(Some(output_data), next_action))
    }
}

pub async fn run_conditional_workflow(
    user_query: &str,
    question: &str,
    nodes: Vec<UnifiedNode>,
    edges: Vec<(&str, &str)>,
    entry_node_id: &str,
) -> Result<WorkflowState> {
    if nodes.is_empty() {
        anyhow::bail!("Nodes list cannot be empty!");
    }

    let mut workflow_builder = DynamicWorkflowBuilder::new("conditional_workflow");

    for node in nodes {
        workflow_builder = workflow_builder.add_task(Arc::new(node));
    }

    for (from, to) in edges {
        workflow_builder = workflow_builder.add_edge(from, to);
    }

    let mermaid_code = workflow_builder.generate_mermaid();
    let graph = workflow_builder.build()?;

    let storage = Arc::new(InMemorySessionStorage::new());
    let session_id = "session_conditional_1".to_string();

    let session = Session::new_from_task(session_id.clone(), entry_node_id);

    session.context.set("query", user_query.to_string())?;
    session.context.set("question", question.to_string())?;
    session.context.set("data", Vec::<String>::new())?;
    session.context.set("final_answer", "".to_string())?;
    session.context.set("router_decision", "".to_string())?;
    session.context.set("audit_trail", Vec::<GraphStepRecord>::new())?;
    session.context.set("architecture_mermaid", mermaid_code)?;

    SessionStorage::save(&*storage, session).await?;

    let runner = FlowRunner::new(Arc::new(graph), storage.clone());
    runner.run(&session_id).await?;

    let final_session = SessionStorage::get(&*storage, &session_id).await.unwrap().unwrap();
    let ctx = &final_session.context;

    let final_state = WorkflowState {
        query: ctx.get::<String>("query").unwrap_or_default(),
        question: ctx.get::<String>("question").unwrap_or_default(),
        data: ctx.get::<Vec<String>>("data").unwrap_or_default(),
        final_answer: ctx.get::<String>("final_answer").unwrap_or_default(),
        architecture_mermaid: ctx.get::<String>("architecture_mermaid").unwrap_or_default(),
        audit_trail: ctx.get::<Vec<GraphStepRecord>>("audit_trail").unwrap_or_default(),
    };

    Ok(final_state)
}

/// Helper alias for backward compatibility with run_graph_workflow
pub async fn run_graph_workflow(
    question: &str,
    nodes: Vec<UnifiedNode>,
    edges: Vec<(&str, &str)>,
    entry_node_id: &str,
) -> Result<WorkflowState> {
    run_conditional_workflow("Workflow Execution", question, nodes, edges, entry_node_id).await
}
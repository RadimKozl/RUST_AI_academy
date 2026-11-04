use anyhow::Result;
use async_trait::async_trait;
use graph_flow::{
    Context, FlowRunner, GraphBuilder, GraphError, InMemorySessionStorage,
    NextAction, Session, SessionStorage, Task, TaskResult,
};
use rig::providers::ollama;
use serde::{Deserialize, Serialize};
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

pub struct InputTask {
    pub id_str: String,
    pub label: String,
}

#[async_trait]
impl Task for InputTask {
    fn id(&self) -> &str {
        &self.id_str
    }

    async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
        let query = context.get::<String>("query").unwrap_or_default();
        let mut data = context.get::<Vec<String>>("data").unwrap_or_default();

        let val = format!("Processed Input for label [{}]", self.label);
        data.push(val.clone());
        context.set("data", data)?;

        let mut audit = context.get::<Vec<GraphStepRecord>>("audit_trail").unwrap_or_default();
        audit.push(GraphStepRecord {
            task_id: self.id_str.clone(),
            input_data: query,
            output_data: val,
        });
        context.set("audit_trail", audit)?;

        // Change from Continue to ContinueAndExecute to move the string to node_b
        Ok(TaskResult::new(
            Some("Input processed".to_string()),
            NextAction::ContinueAndExecute,
        ))
    }
}

pub struct LlmAgentTask {
    pub id_str: String,
    pub model_name: String,
}

#[async_trait]
impl Task for LlmAgentTask {
    fn id(&self) -> &str {
        &self.id_str
    }

    async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
        let query = context.get::<String>("query").unwrap_or_default();
        let current_data = context.get::<Vec<String>>("data").unwrap_or_default();

        let client = ollama::Ollama::new();
        let model = client.completion(&self.model_name);

        let agent = rig::agent::AgentBuilder::new(model)
            .preamble("You are a helpful assistant executing inside a Rust GraphFlow node.")
            .build();

        let prompt = format!(
            "Context Data: {:?}\nUser Query: {}\nProvide a concise answer:",
            current_data, query
        );

        match agent.prompt(&prompt).await {
            Ok(response) => {
                let output = response.output().trim().to_string();
                context.set("final_answer", output.clone())?;

                let mut audit = context.get::<Vec<GraphStepRecord>>("audit_trail").unwrap_or_default();
                audit.push(GraphStepRecord {
                    task_id: self.id_str.clone(),
                    input_data: prompt,
                    output_data: output.clone(),
                });
                context.set("audit_trail", audit)?;

                Ok(TaskResult::new(Some(output), NextAction::End))
            }
            Err(e) => {
                eprintln!("LLM Task Execution Error: {}", e);
                let err_msg = format!("Error: {}", e);
                Ok(TaskResult::new(Some(err_msg), NextAction::End))
            }
        }
    }
}

pub async fn run_langgraph_workflow(
    model_name: &str,
    user_query: &str,
) -> Result<WorkflowState> {
    let workflow_builder = DynamicWorkflowBuilder::new("langgraph_workflow")
        .add_task(Arc::new(InputTask {
            id_str: "node_a".to_string(),
            label: "Node_A".to_string(),
        }))
        .add_task(Arc::new(LlmAgentTask {
            id_str: "node_b".to_string(),
            model_name: model_name.to_string(),
        }))
        .add_edge("node_a", "node_b");

    let mermaid_code = workflow_builder.generate_mermaid();
    let graph = workflow_builder.build()?;

    let storage = Arc::new(InMemorySessionStorage::new());
    let session_id = "session_1".to_string();
    
    let session = Session::new_from_task(session_id.clone(), "node_a");
    
    session.context.set("query", user_query.to_string())?;
    session.context.set("data", Vec::<String>::new())?;
    session.context.set("final_answer", "".to_string())?;
    session.context.set("audit_trail", Vec::<GraphStepRecord>::new())?;
    session.context.set("architecture_mermaid", mermaid_code)?;

    SessionStorage::save(&*storage, session).await?;

    let runner = FlowRunner::new(Arc::new(graph), storage.clone());
    runner.run(&session_id).await?;

    let final_session = SessionStorage::get(&*storage, &session_id).await.unwrap().unwrap();
    let ctx = &final_session.context;

    let final_state = WorkflowState {
        query: ctx.get::<String>("query").unwrap_or_default(),
        data: ctx.get::<Vec<String>>("data").unwrap_or_default(),
        final_answer: ctx.get::<String>("final_answer").unwrap_or_default(),
        architecture_mermaid: ctx.get::<String>("architecture_mermaid").unwrap_or_default(),
        audit_trail: ctx.get::<Vec<GraphStepRecord>>("audit_trail").unwrap_or_default(),
    };

    Ok(final_state)
}
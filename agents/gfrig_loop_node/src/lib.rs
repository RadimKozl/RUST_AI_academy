use anyhow::Result;
use async_trait::async_trait;
use graph_flow::{
    Context, GraphBuilder, GraphError, InMemorySessionStorage,
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
    pub counter: i32,
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
                let agent = rig::agent::AgentBuilder::new(model).preamble(preamble).build();
                let prompt = format!("Question: {}\nDetermine the routing decision:", question);

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
                let agent = rig::agent::AgentBuilder::new(model).preamble(preamble).build();
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

/// Robust iterative workflow runner for the secure management of cycles (Loop Pattern)
pub async fn run_loop_workflow(
    user_query: &str,
    question: &str,
    nodes: Vec<UnifiedNode>,
    edges: Vec<(&str, &str)>,
    _entry_node_id: &str,
    max_loops: i32,
) -> Result<WorkflowState> {
    if nodes.is_empty() {
        anyhow::bail!("Nodes list cannot be empty!");
    }

    let mut workflow_builder = DynamicWorkflowBuilder::new("loop_workflow");
    
    // Storing nodes in a map for direct execution within an iterative loop
    let mut task_map = std::collections::HashMap::new();
    for node in nodes {
        let task_arc = Arc::new(node);
        task_map.insert(task_arc.id().to_string(), task_arc.clone());
        workflow_builder = workflow_builder.add_task(task_arc);
    }

    for (from, to) in edges {
        workflow_builder = workflow_builder.add_edge(from, to);
    }

    let mermaid_code = workflow_builder.generate_mermaid();
    let _graph = workflow_builder.build()?;

    let storage = Arc::new(InMemorySessionStorage::new());
    let session_id = "session_loop_1".to_string();
    let session = Session::new_from_task(session_id.clone(), "node_a");

    session.context.set("query", user_query.to_string())?;
    session.context.set("question", question.to_string())?;
    session.context.set("counter", 0)?;
    session.context.set("max_loops", max_loops)?;
    session.context.set("data", Vec::<String>::new())?;
    session.context.set("final_answer", "".to_string())?;
    session.context.set("router_decision", "".to_string())?;
    session.context.set("audit_trail", Vec::<GraphStepRecord>::new())?;
    session.context.set("architecture_mermaid", mermaid_code.clone())?;

    SessionStorage::save(&*storage, session).await?;

    // 1. Launching the starting node A
    let sess = SessionStorage::get(&*storage, &session_id).await?.unwrap();
    if let Some(node_a) = task_map.get("node_a") {
        node_a.run(sess.context.clone()).await?;
    }
    SessionStorage::save(&*storage, sess).await?;

    // 2. Iterative cycle loop (B -> C -> Router until counter < max_loops)
    loop {
        let sess = SessionStorage::get(&*storage, &session_id).await?.unwrap();
        let counter = sess.context.get::<i32>("counter").unwrap_or_default();

        if counter >= max_loops {
            break;
        }

        // Start Node B
        if let Some(node_b) = task_map.get("node_b") {
            node_b.run(sess.context.clone()).await?;
        }
        // Launching Node C (increments the counter)
        if let Some(node_c) = task_map.get("node_c") {
            node_c.run(sess.context.clone()).await?;
        }
        // Starting the Router
        if let Some(node_router) = task_map.get("node_router") {
            node_router.run(sess.context.clone()).await?;
        }

        SessionStorage::save(&*storage, sess).await?;
    }

    // 3. Execution of end node D after the loop finishes
    let final_sess = SessionStorage::get(&*storage, &session_id).await?.unwrap();
    if let Some(node_d) = task_map.get("node_d") {
        node_d.run(final_sess.context.clone()).await?;
    }
    SessionStorage::save(&*storage, final_sess).await?;

    let updated_session = SessionStorage::get(&*storage, &session_id).await.unwrap().unwrap();
    let ctx = &updated_session.context;

    let final_state = WorkflowState {
        query: ctx.get::<String>("query").unwrap_or_default(),
        question: ctx.get::<String>("question").unwrap_or_default(),
        counter: ctx.get::<i32>("counter").unwrap_or_default(),
        data: ctx.get::<Vec<String>>("data").unwrap_or_default(),
        final_answer: ctx.get::<String>("final_answer").unwrap_or_default(),
        architecture_mermaid: ctx.get::<String>("architecture_mermaid").unwrap_or_default(),
        audit_trail: ctx.get::<Vec<GraphStepRecord>>("audit_trail").unwrap_or_default(),
    };

    Ok(final_state)
}
use async_trait::async_trait;
use graph_flow::{Context as GraphContext, GraphBuilder, Task, TaskResult};
use gfrig_human_loop::HitlState;
use std::sync::Arc;

pub struct MockHumanTask {
    id_str: String,
    should_approve: bool,
}

impl MockHumanTask {
    pub fn new(id: impl Into<String>, should_approve: bool) -> Self {
        Self {
            id_str: id.into(),
            should_approve,
        }
    }
}

#[async_trait]
impl Task for MockHumanTask {
    fn id(&self) -> &str {
        &self.id_str
    }

    async fn run(&self, context: GraphContext) -> graph_flow::Result<TaskResult> {
        let iteration = context.get::<i32>("iteration").unwrap_or(0);

        if self.should_approve || iteration >= 2 {
            let _ = context.set("approved", true);
        } else {
            let _ = context.set("approved", false);
        }

        Ok(TaskResult::move_to_next())
    }
}

pub struct MockJokeTask {
    id_str: String,
}

impl MockJokeTask {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id_str: id.into() }
    }
}

#[async_trait]
impl Task for MockJokeTask {
    fn id(&self) -> &str {
        &self.id_str
    }

    async fn run(&self, context: GraphContext) -> graph_flow::Result<TaskResult> {
        let iteration = context.get::<i32>("iteration").unwrap_or(0) + 1;
        let joke = format!("Mock joke no. {}", iteration);

        let _ = context.set("joke_text", joke.clone());
        let _ = context.set("iteration", iteration);

        Ok(TaskResult::move_to_next())
    }
}

pub struct MockStoryTask {
    id_str: String,
}

impl MockStoryTask {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id_str: id.into() }
    }
}

#[async_trait]
impl Task for MockStoryTask {
    fn id(&self) -> &str {
        &self.id_str
    }

    async fn run(&self, context: GraphContext) -> graph_flow::Result<TaskResult> {
        let joke = context.get::<String>("joke_text").unwrap_or_default();
        let story = format!("Backstory for the joke: '{}'", joke);

        let _ = context.set("story_text", story.clone());

        Ok(TaskResult::move_to_next())
    }
}

#[tokio::test]
async fn test_hitl_approval_on_first_try() {
    let builder = GraphBuilder::new("test_graph_approval");

    // The Fluent API returns its own value, so we assign it to the builder
    let builder = builder.add_task(Arc::new(MockJokeTask::new("joke")));
    let builder = builder.add_task(Arc::new(MockHumanTask::new("human", true)));
    let builder = builder.add_task(Arc::new(MockStoryTask::new("story")));

    let builder = builder.add_edge("joke", "human");
    let builder = builder.add_edge("human", "story");

    let _graph = builder.build().expect("The graph should build successfully.");

    let context = GraphContext::new();
    let _ = context.set("theme", "universe".to_string());
    let _ = context.set("iteration", 0);

    let joke_task = MockJokeTask::new("joke");
    let _ = joke_task.run(context.clone()).await;

    let human_task = MockHumanTask::new("human", true);
    let _ = human_task.run(context.clone()).await;

    let approved = context.get::<bool>("approved").unwrap_or(false);
    let iteration = context.get::<i32>("iteration").unwrap_or(0);

    assert!(approved, "The joke was supposed to be approved.");
    assert_eq!(iteration, 1, "One iteration was supposed to take place.");
}

#[tokio::test]
async fn test_hitl_cycle_and_retry_flow() {
    let builder = GraphBuilder::new("test_graph_cycle");

    let builder = builder.add_task(Arc::new(MockJokeTask::new("joke")));
    let builder = builder.add_task(Arc::new(MockHumanTask::new("human", false)));
    let builder = builder.add_task(Arc::new(MockStoryTask::new("story")));

    let builder = builder.add_edge("joke", "human");

    let _graph = builder.build().expect("The graph should build successfully.");

    let context = GraphContext::new();
    let _ = context.set("theme", "universe".to_string());
    let _ = context.set("iteration", 0);

    let joke_task = MockJokeTask::new("joke");
    let _ = joke_task.run(context.clone()).await;

    let human_task = MockHumanTask::new("human", false);
    let _ = human_task.run(context.clone()).await;

    let mut approved = context.get::<bool>("approved").unwrap_or(true);
    assert!(!approved, "The joke in the first attempt should not have been approved.");

    let _ = joke_task.run(context.clone()).await;
    let human_task_approved = MockHumanTask::new("human", true);
    let _ = human_task_approved.run(context.clone()).await;

    approved = context.get::<bool>("approved").unwrap_or(false);
    let iteration = context.get::<i32>("iteration").unwrap_or(0);

    assert!(approved, "The joke was supposed to be approved on the second attempt.");
    assert_eq!(iteration, 2, "Two iterations were supposed to take place.");
}

#[test]
fn test_hitl_state_default_deserialization() {
    let state = HitlState {
        theme: "universe".to_string(),
        joke_text: "Joke".to_string(),
        story_text: "Story".to_string(),
        approved: true,
        iteration: 1,
    };

    let json = serde_json::to_string(&state).unwrap();
    let deserialized: HitlState = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.theme, "universe");
    assert!(deserialized.approved);
    assert_eq!(deserialized.iteration, 1);
}
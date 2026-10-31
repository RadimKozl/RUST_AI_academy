use anyhow::Result;
use rig::providers::ollama;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Definition of logical conditions for the router (supports AND, OR, NOR, NOT)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicCondition {
    /// Contains the keyword in the target field
    Contains { target: String, keyword: String },
    /// Equals the exact value
    Equals { target: String, value: String },
    /// All conditions must be met (AND)
    And(Vec<LogicCondition>),
    /// At least one condition must be met (OR)
    Or(Vec<LogicCondition>),
    /// No condition may be met (NOR)
    Nor(Vec<LogicCondition>),
    /// Negation of a condition (NOT)
    Not(Box<LogicCondition>),
}

/// Recursive function for evaluating a logical tree of conditions
pub fn evaluate_condition(cond: &LogicCondition, context: &HashMap<String, String>) -> bool {
    match cond {
        LogicCondition::Contains { target, keyword } => {
            if let Some(val) = context.get(target) {
                val.to_lowercase().contains(&keyword.to_lowercase())
            } else {
                false
            }
        }
        LogicCondition::Equals { target, value } => {
            if let Some(val) = context.get(target) {
                val.eq_ignore_ascii_case(value)
            } else {
                false
            }
        }
        LogicCondition::And(conditions) => {
            conditions.iter().all(|c| evaluate_condition(c, context))
        }
        LogicCondition::Or(conditions) => {
            conditions.iter().any(|c| evaluate_condition(c, context))
        }
        LogicCondition::Nor(conditions) => {
            !conditions.iter().any(|c| evaluate_condition(c, context))
        }
        LogicCondition::Not(c) => {
            !evaluate_condition(c, context)
        }
    }
}

/// Step recording for XAI / Neo4j memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterStepRecord {
    pub step_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// Result of dynamic routing
#[derive(Debug, Serialize, Deserialize)]
pub struct RouterChainResult {
    pub matched_rule: String,
    pub final_output: String,
    pub history: Vec<RouterStepRecord>,
}

/// Rule for a router: Connects a logical condition and an action branch (agent)
pub struct RoutingRule {
    pub name: String,
    pub condition: LogicCondition,
    pub system_instruction: String,
    pub prompt_template: String,
}

/// Universal Rule-Based Router with Boolean Logic Support
pub async fn run_rule_based_router(
    model_name: &str,
    question: &str,
    classifier_system: &str,
    classifier_prompt_tmpl: &str,
    rules: &[RoutingRule],
    default_rule: &RoutingRule,
) -> Result<RouterChainResult> {
    let client = ollama::Ollama::new();

    // Step 1: Classification / Context extraction using an LLM
    let classifier_model = client.completion(model_name);
    let classifier_agent = rig::agent::AgentBuilder::new(classifier_model)
        .preamble(classifier_system)
        .build();

    let classifier_prompt = classifier_prompt_tmpl.replace("{question}", question);
    let classification_response = classifier_agent.prompt(&classifier_prompt).await?;
    let classification = classification_response.output().trim().to_string();

    // We will create a context map for evaluating conditions
    let mut context = HashMap::new();
    context.insert("question".to_string(), question.to_string());
    context.insert("topic".to_string(), classification.clone());

    let mut history = vec![RouterStepRecord {
        step_name: "classifier".to_string(),
        input_value: question.to_string(),
        generated_output: classification.clone(),
    }];

    // Step 2: Iterating through rules and evaluating the Boolean tree (AND, OR, NOR...)
    let mut selected_rule = default_rule;
    for rule in rules {
        if evaluate_condition(&rule.condition, &context) {
            selected_rule = rule;
            break;
        }
    }

    // Step 3: Executing the selected branch
    let branch_model = client.completion(model_name);
    let branch_agent = rig::agent::AgentBuilder::new(branch_model)
        .preamble(&selected_rule.system_instruction)
        .build();

    let branch_prompt = selected_rule.prompt_template.replace("{question}", question);
    let branch_response = branch_agent.prompt(&branch_prompt).await?;
    let final_output = branch_response.output().trim().to_string();

    history.push(RouterStepRecord {
        step_name: selected_rule.name.clone(),
        input_value: question.to_string(),
        generated_output: final_output.clone(),
    });

    Ok(RouterChainResult {
        matched_rule: selected_rule.name.clone(),
        final_output,
        history,
    })
}
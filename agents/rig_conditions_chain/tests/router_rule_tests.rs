use rig_conditions_chain::{evaluate_condition, run_rule_based_router, LogicCondition, RoutingRule};
use std::collections::HashMap;

#[test]
fn test_logic_condition_evaluator() {
    let mut context = HashMap::new();
    context.insert("topic".to_string(), "LangChain".to_string());
    context.insert("question".to_string(), "Give me breaking news".to_string());

    // 1. Test Contains
    let cond_contains = LogicCondition::Contains {
        target: "topic".to_string(),
        keyword: "langchain".to_string(),
    };
    assert!(evaluate_condition(&cond_contains, &context));

    // 2. Test AND (LangChain topic AND question contains "news")
    let cond_and = LogicCondition::And(vec![
        LogicCondition::Contains { target: "topic".to_string(), keyword: "LangChain".to_string() },
        LogicCondition::Contains { target: "question".to_string(), keyword: "news".to_string() },
    ]);
    assert!(evaluate_condition(&cond_and, &context));

    // 3. Test OR (topic is Anthropic OR question contains "news")
    let cond_or = LogicCondition::Or(vec![
        LogicCondition::Equals { target: "topic".to_string(), value: "Anthropic".to_string() },
        LogicCondition::Contains { target: "question".to_string(), keyword: "news".to_string() },
    ]);
    assert!(evaluate_condition(&cond_or, &context));

    // 4. Test NOR (neither topic is Anthropic nor topic is Google)
    let cond_nor = LogicCondition::Nor(vec![
        LogicCondition::Equals { target: "topic".to_string(), value: "Anthropic".to_string() },
        LogicCondition::Equals { target: "topic".to_string(), value: "Google".to_string() },
    ]);
    assert!(evaluate_condition(&cond_nor, &context));

    // 5. Test NOT
    let cond_not = LogicCondition::Not(Box::new(LogicCondition::Equals {
        target: "topic".to_string(),
        value: "Anthropic".to_string(),
    }));
    assert!(evaluate_condition(&cond_not, &context));
}

#[tokio::test]
async fn test_rule_based_router_integration() {
    let model_name = "qwen3.5:4b";
    let classifier_system = "You are a keyword extractor. Respond with a single category word: `LangChain`, `Anthropic`, or `Other`.";
    let classifier_prompt = "Classify this question into one category (`LangChain`, `Anthropic`, `Other`):\n<question>\n{question}\n</question>\nCategory:";

    let rules = vec![
        RoutingRule {
            name: "langchain_news_branch".to_string(),
            condition: LogicCondition::And(vec![
                LogicCondition::Contains { target: "topic".to_string(), keyword: "langchain".to_string() },
                LogicCondition::Contains { target: "question".to_string(), keyword: "news".to_string() },
            ]),
            system_instruction: "You are a tech journalist.".to_string(),
            prompt_template: "Answer: {question}".to_string(),
        },
    ];

    let default_rule = RoutingRule {
        name: "fallback_branch".to_string(),
        condition: LogicCondition::Equals { target: "topic".to_string(), value: "".to_string() },
        system_instruction: "You are a helpful assistant.".to_string(),
        prompt_template: "Answer: {question}".to_string(),
    };

    let question = "Give me news about LangChain.";
    let result = run_rule_based_router(
        model_name,
        question,
        classifier_system,
        classifier_prompt,
        &rules,
        &default_rule,
    ).await;

    match result {
        Ok(res) => {
            assert!(!res.matched_rule.is_empty());
            assert!(!res.final_output.is_empty());
            assert_eq!(res.history.len(), 2, "The history must include the classification and the triggered branch.");
            println!("Rule-based router test passed successfully! Matched rule: {}", res.matched_rule);
        }
        Err(e) => {
            println!("Ollama is unavailable (expected in CI/offline environments): {}", e);
        }
    }
}
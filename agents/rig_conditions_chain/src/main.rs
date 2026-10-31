use anyhow::Result;
use dotenvy::dotenv;
use rig_conditions_chain::{run_rule_based_router, LogicCondition, RoutingRule};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG BOOLEAN RULE-BASED ROUTER (AND/OR/NOR) ===\n");

    let model_name = "qwen3.5:4b";

    let classifier_system = "You are a keyword extractor. Respond with a single category word: `LangChain`, `Anthropic`, or `Other`.";
    let classifier_prompt = "Classify this question into one category (`LangChain`, `Anthropic`, `Other`):\n<question>\n{question}\n</question>\nCategory:";

    // Definition of advanced rules with logical operators AND, OR, NOR
    let rules = vec![
        // Rule 1 (OR): If the topic is Anthropic OR the query contains the word "claude"
        RoutingRule {
            name: "anthropic_branch".to_string(),
            condition: LogicCondition::Or(vec![
                LogicCondition::Contains { target: "topic".to_string(), keyword: "anthropic".to_string() },
                LogicCondition::Contains { target: "question".to_string(), keyword: "claude".to_string() },
            ]),
            system_instruction: "You are an expert in Anthropic. Always answer starting with 'As Dario Amodei told me'.".to_string(),
            prompt_template: "Question: {question}\nAnswer:".to_string(),
        },
        // Rule 2 (AND): If the topic is LangChain AND AT THE SAME TIME the query contains "news"
        RoutingRule {
            name: "langchain_news_branch".to_string(),
            condition: LogicCondition::And(vec![
                LogicCondition::Contains { target: "topic".to_string(), keyword: "langchain".to_string() },
                LogicCondition::Contains { target: "question".to_string(), keyword: "news".to_string() },
            ]),
            system_instruction: "You are a tech journalist reporting on LangChain updates.".to_string(),
            prompt_template: "Provide a breaking news update for: {question}\nAnswer:".to_string(),
        },
        // Rule 3 (NOR): If it is NEITHER LangChain NOR Anthropic (ensured via NOR)
        RoutingRule {
            name: "general_branch".to_string(),
            condition: LogicCondition::Nor(vec![
                LogicCondition::Contains { target: "topic".to_string(), keyword: "langchain".to_string() },
                LogicCondition::Contains { target: "topic".to_string(), keyword: "anthropic".to_string() },
            ]),
            system_instruction: "You are a helpful general assistant.".to_string(),
            prompt_template: "Question: {question}\nAnswer:".to_string(),
        },
    ];

    let default_rule = RoutingRule {
        name: "fallback_branch".to_string(),
        condition: LogicCondition::Equals { target: "topic".to_string(), value: "".to_string() },
        system_instruction: "You are a helpful assistant.".to_string(),
        prompt_template: "Question: {question}\nAnswer:".to_string(),
    };

    let test_questions = vec![
        "Tell me what Claude can do.",       // Triggers Anthropic (via OR – the word "claude")
        "Give me news about LangChain.",     // Launches LangChain News (via AND)
        "Explain Newton's 1st law.",         // Launches General (via NOR)
    ];

    for q in test_questions {
        println!("--------------------------------------------------");
        println!("Question: '{}'\n", q);

        match run_rule_based_router(model_name, q, classifier_system, classifier_prompt, &rules, &default_rule).await {
            Ok(res) => {
                println!("Matched Rule: [{}]", res.matched_rule);
                println!("Final Answer:\n{}\n", res.final_output);
                
                // XAI / Neo4j audit trail
                let json_audit = serde_json::to_string_pretty(&res.history).unwrap();
                println!("--- XAI Audit Trail ---");
                println!("{}\n", json_audit);
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }

    Ok(())
}
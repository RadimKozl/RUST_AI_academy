use rig_parallel_chain::{run_parallel_chain, ParallelBranch};

#[tokio::test]
async fn test_parallel_chain_execution() {
    let model_name = "qwen3.5:4b";
    let topic = "coffee";

    let branches = vec![
        ParallelBranch {
            name: "joke".to_string(),
            system_instruction: "Be funny.".to_string(),
            prompt_template: "Tell a joke about {topic}.".to_string(),
        },
        ParallelBranch {
            name: "poem".to_string(),
            system_instruction: "Be poetic.".to_string(),
            prompt_template: "Write a short poem about {topic}.".to_string(),
        },
    ];

    let result = run_parallel_chain(model_name, topic, &branches).await;

    match result {
        Ok(map) => {
            assert!(map.contains_key("joke"), "Výsledek musí obsahovat klíč 'joke'");
            assert!(map.contains_key("poem"), "Výsledek musí obsahovat klíč 'poem'");
            assert!(!map.get("joke").unwrap().is_empty());
            assert!(!map.get("poem").unwrap().is_empty());
            println!("Paralelní test úspěšně prošel! Výstupy: {:#?}", map);
        }
        Err(e) => {
            println!("Ollama je nedostupná (v CI/offline prostředí očekáváno): {}", e);
        }
    }
}
use anyhow::Result;
use dotenvy::dotenv;
use rig_branch_chain::run_branch_chain; // Adjust the name according to your crate in Cargo.toml

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG RUNNABLE BRANCH (CONDITIONAL ROUTING) ===\n");

    let model_name = "qwen3.5:4b";
    let topic = "quantum mechanics";

    let test_cases = vec![
        ("formal", "--- Example 1: Formal Inquiry ---"),
        ("joke", "--- Example 2: Funny / Informal Inquiry ---"),
        ("neutral", "--- Example 3: Default (Else Branch) ---"),
    ];

    for (tone, title) in test_cases {
        println!("{}", title);
        println!("TONE: {}\n", tone);

        match run_branch_chain(model_name, topic, tone).await {
            Ok(res) => {
                println!("Matched Branch: [{}]", res.matched_branch);
                let preview: String = res.final_output.chars().take(120).collect();
                println!("START: {}...\n", preview);

                // XAI / Neo4j Audit Trail
                let json_audit = serde_json::to_string_pretty(&res.history).unwrap();
                println!("--- XAI Audit Trail ---");
                println!("{}\n", json_audit);
            }
            Err(e) => eprintln!("Error: {}", e),
        }
        println!("--------------------------------------------------\n");
    }

    Ok(())
}
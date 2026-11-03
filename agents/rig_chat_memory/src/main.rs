use anyhow::Result;
use dotenvy::dotenv;
use rig_chat_memory::{run_hybrid_chat_turn, SqliteMemoryManager};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG HYBRID & PERSISTENT CHAT MEMORY ===\n");

    let db_path = "chat_history.db";
    let session_id = "socks_company_session";
    let base_model = "qwen3.5:4b";
    let summary_model = "qwen3.5:4b";

    let db_manager = SqliteMemoryManager::new(db_path)?;
    db_manager.clear_session(session_id)?; // Clean session start

    let queries = vec![
        "What would be a good company name for a company that makes colorful socks?",
        "Give me another 5 concise options.",
        "And add a creative motto for the first option.",
        "Okay, and what logo design would go with that motto?",
    ];

    for (idx, q) in queries.iter().enumerate() {
        println!("--------------------------------------------------");
        println!("Turn {}: User Question: '{}'\n", idx + 1, q);

        match run_hybrid_chat_turn(&db_manager, session_id, base_model, summary_model, q).await {
            Ok(res) => {
                if let Some(sum) = &res.used_summary {
                    println!("--- Active Conversation Summary ---");
                    println!("{}\n", sum);
                }

                println!("--- Assistant Answer ---");
                println!("{}\n", res.final_output);

                // XAI / Neo4j Audit Trail
                let json_audit = serde_json::to_string_pretty(&res.history).unwrap();
                println!("--- XAI Audit Trail (JSON) ---");
                println!("{}\n", json_audit);
            }
            Err(e) => eprintln!("Error in turn {}: {}", idx + 1, e),
        }
    }

    Ok(())
}
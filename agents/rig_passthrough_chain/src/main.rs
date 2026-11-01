use anyhow::Result;
use dotenvy::dotenv;
use rig_passthrough_chain::run_passthrough_rag_chain; // Adjust the name according to your crate in Cargo.toml

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("=== RUST RIG RUNNABLE PASSTHROUGH (RAG PATTERN) ===\n");

    let model_name = "qwen3.5:4b";
    let question = "Where is the company that made the Tesla Model Y headquartered?";

    let context_system = "You are a factual knowledge assistant. Provide concise, direct facts.";
    let context_prompt = "For the given question ({question}), write ONE sentence of relevant FACT (e.g. CEO, year of foundation, headquarters).";

    let final_system = "You are a precise assistant answering questions based on the provided context.";
    let final_prompt = "Answer the question based on the context provided.\nCONTEXT: {context}\nQUESTION: {question}\nResponse:";

    println!("User Question: '{}'\n", question);

    match run_passthrough_rag_chain(
        model_name,
        question,
        context_system,
        context_prompt,
        final_system,
        final_prompt,
    )
    .await
    {
        Ok(res) => {
            println!("--- Generated Context (Retriever simulation) ---");
            println!("{}\n", res.generated_context);

            println!("--- Final RAG Output ---");
            println!("{}\n", res.final_output);

            // XAI / Neo4j Audit Trail
            let json_audit = serde_json::to_string_pretty(&res.history).unwrap();
            println!("--- XAI Audit Trail (JSON) ---");
            println!("{}", json_audit);
        }
        Err(e) => eprintln!("Error during execution: {}", e),
    }

    Ok(())
}
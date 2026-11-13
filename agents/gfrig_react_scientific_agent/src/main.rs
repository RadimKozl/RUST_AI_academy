use gfrig_react_scientific_agent::ScientificGraphWorkflow;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let query = "Find recent papers on quantum error correction and evaluate polynomial x^2 + 4*x + 4";
    
    println!("Launching the RIG Graph-flow agent for the query: '{}'\n", query);

    let mut workflow = ScientificGraphWorkflow::new(query);
    workflow.run().await?;

    let final_state = workflow.get_state();
    
    println!("\n================ FINAL GRAPH OUTPUT ================");
    println!("{}", final_state.final_output);

    if let Some(report) = &final_state.xai_report {
        println!("\n================ XAI AUDIT REPORT (JSON) ================");
        println!("{}", serde_json::to_string_pretty(report)?);
    }

    Ok(())
}
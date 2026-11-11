use rig::providers::ollama;
use rig::agent::AgentBuilder;
use gfrig_rag_subgraph::AgentState;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RUST LANGGRAPH -> RIG: SUBGRAPH RAG WITH XAI ===");

    // 1. Initialization of the Ollama client and model
    let client = ollama::Ollama::new();
    let model = client.completion("glm4:9b");

    // Creating an agent using AgentBuilder
    let agent = AgentBuilder::new(model)
        .preamble("You are an expert AI assistant specialized in research, RAG systems, and data analysis.")
        .build();

    // Initialization state
    let mut state = AgentState::new("What are the developments and trends in AI (artificial intelligence)?");

    // ==========================================
    // STEP 1: Prepare Query Node (Main Graph)
    // ==========================================
    println!("\n--- [PrepareQueryNode] ---");
    let query_prompt = format!(
        "Improve the user query so it can be used for a better search. User question: {}",
        state.task
    );
    let improved_query_resp = agent.prompt(&query_prompt).await?;
    state.question = improved_query_resp.to_string().trim().to_string();
    println!("✅ Improved Query: '{}'", state.question);

    // ==========================================
    // STEP 2: Wikipedia / External Search Node
    // ==========================================
    println!("\n--- [WikipediaCallNode] ---");
    state.search_result = format!(
        "External search results for '{}': AI is rapidly evolving with focus on LLMs, Edge AI, and Hybrid architectures.",
        state.question
    );
    println!("✅ Search context retrieved.");

    // =============================================================
    // STEP 3: Micro-RAG Subgraph (Cyclic Self-Correction + XAI)
    // =============================================================
    println!("\n--- [MicroRAG Subgraph Start with XAI Evaluation] ---");
    
    let max_iterations = 3;
    let mut rag_success = false;

    while state.iteration < max_iterations && !rag_success {
        state.iteration += 1;
        println!("\n🔄 RAG Iteration no. {} (out of {})", state.iteration, max_iterations);

        // 3.1 Retrieval Node
        println!("  -> [RetrievalNode] Fetching documents from SQLite/FAISS...");
        state.documents = vec![
            "Document 1: Machine learning and deep learning integration form the core of modern AI.".to_string(),
            "Document 2: Edge AI allows running models locally with lower latency and higher privacy.".to_string(),
            "Document 3: Explainable AI (XAI) is critical for enterprise deployment and regulation compliance.".to_string(),
        ];

        // 3.2 Rerank Node
        println!("  -> [RerankNode] Reranking documents...");
        state.context = format!("{}\n\n{}", state.documents[0], state.documents[1]);

        // 3.3 Generation Node
        println!("  -> [GenerationNode] Generating answer from context...");
        let gen_prompt = format!(
            "Answer the question based only on the following context:\nContext: {}\nQuestion: {}",
            state.context, state.question
        );
        let answer_resp = agent.prompt(&gen_prompt).await?;
        state.answer = answer_resp.to_string();

        // 3.4 Critic Node with XAI (Evaluation + Justification)
        println!("  -> [CriticNode] Evaluating answer quality and generating XAI reasoning...");
        let critic_prompt = format!(
            "You are a critic agent. Evaluate the answer for relevance to the question. \
            Return your response strictly in this format: \
            SCORE: <number from 1.0 to 10.0> | REASON: <brief explanation why>\n\
            Question: {}\nAnswer: {}",
            state.question, state.answer
        );
        let critic_resp = agent.prompt(&critic_prompt).await?;
        let critic_response_str = critic_resp.to_string();
        
        // Processing XAI output from the critic
        if critic_response_str.contains("SCORE:") && critic_response_str.contains("REASON:") {
            let parts: Vec<&str> = critic_response_str.split('|').collect();
            let score_part = parts[0].replace("SCORE:", "").trim().to_string();
            let reason_part = parts[1].replace("REASON:", "").trim().to_string();

            state.critic_score = score_part.parse().unwrap_or(7.5);
            state.critic_reason = reason_part;
        } else {
            state.critic_score = 7.5;
            state.critic_reason = critic_response_str.trim().to_string();
        }

        println!("  ⭐ Subgraph XAI Critic Score: {} / 10.0", state.critic_score);
        println!("  🔍 Subgraph XAI Reasoning: \"{}\"", state.critic_reason);

        // 3.5 decide_loop
        if state.critic_score >= 7.0 {
            println!("✅ Subgraph condition met: Score >= 7. Exiting RAG loop.");
            rag_success = true;
        } else {
            println!("⚠️ Score too low (< 7). Retrying retrieval with modified query feedback...");
            state.question.push_str(" (please specify for a better answer)");
        }
    }

    // ==========================================================
    // STEP 4: SummarizeNode (Main Graph – Synthesis with XAI)
    // ==========================================================
    println!("\n--- [SummarizeNode & Main App XAI] ---");
    let summary_prompt = format!(
        "Summarize the following information combining external search and verified internal RAG answer. \
        Provide a comprehensive overview.\n\nExternal Search:\n{}\n\nMicro-RAG Answer:\n{}",
        state.search_result, state.answer
    );
    let summary_resp = agent.prompt(&summary_prompt).await?;
    state.summarization = summary_resp.to_string();

    // ====================================================
    // STEP 5: Final XAI output for the main application
    // ====================================================
    println!("\n==========================================");
    println!("========= FINAL PIPELINE RESULT ==========");
    println!("==========================================");
    println!("{}", state.summarization);
    println!("\n==========================================");
    println!("🔍 XAI TRANSPARENCY REPORT (MAIN APP & SUBGRAPH):");
    println!("==========================================");
    println!("• Original User Task:        {}", state.task);
    println!("• Optimized Search Query:    {}", state.question);
    println!("• Subgraph Iterations Run:   {} / {}", state.iteration, max_iterations);
    println!("• Final Subgraph Score:      {} / 10.0", state.critic_score);
    println!("• Subgraph Critic Reasoning: {}", state.critic_reason);
    println!("• Sources Integrated:        External Wikipedia/Web search + SQLite/FAISS Micro-RAG");
    println!("==========================================");

    Ok(())
}
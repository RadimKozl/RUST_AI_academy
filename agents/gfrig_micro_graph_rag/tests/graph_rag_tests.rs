// tests/graph_rag_tests.rs
use gfrig_micro_graph_rag::{cosine_similarity, AdvancedRagDatabase};
use tempfile::NamedTempFile;

#[test]
fn test_cosine_similarity() {
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![1.0, 0.0, 0.0];
    let v3 = vec![0.0, 1.0, 0.0];

    assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-5);
    assert!((cosine_similarity(&v1, &v3)).abs() < 1e-5);
}

#[tokio::test]
async fn test_graph_rag_database_operations() {
    let temp_db = NamedTempFile::new().unwrap();
    
    // 1. Initialization of the database in a temporary file
    let db = AdvancedRagDatabase::init(temp_db.path()).await.unwrap();

    // 2. Insertion of a document and a chunk
    let doc_id = db
        .insert_document("Agentic frameworks like AutoGen and LangGraph.", "arxiv_test")
        .await
        .unwrap();
    assert!(doc_id > 0);

    let embedding = vec![1.0, 0.0, 0.0];
    let chunk_id = db
        .insert_chunk(doc_id, "Agentic chunk text", &embedding)
        .await
        .unwrap();
    assert!(chunk_id > 0);

    // 3. Insertion of entities and relationships (G-RAG knowledge graph)
    let entities = vec![
        ("Agentic Applications", "Concept", "Autonomous systems"),
        ("AutoGen", "Framework", "Multi-agent framework"),
    ];
    let relations = vec![
        ("Agentic Applications", "UTILIZES", "AutoGen"),
    ];

    let graph_res = db.insert_graph_knowledge(chunk_id, &entities, &relations).await;
    assert!(graph_res.is_ok(), "Graph knowledge insertion must succeed");

    // 4. Test of combined G-RAG retrieval (Vector + Petgraph hydration)
    let query_emb = vec![1.0, 0.0, 0.0];
    let search_result = db.search_graph_rag(&query_emb, "Agentic Applications", 2).await;
    assert!(search_result.is_ok());

    let result_str = search_result.unwrap();
    
    // Verification that the output contains both a vector chunk and graph entities and relationships
    assert!(result_str.contains("Agentic chunk text"), "Result should contain vector chunk");
    assert!(result_str.contains("Agentic Applications"), "Result should contain entity name");
    assert!(result_str.contains("AutoGen"), "Result should contain related entity");
    assert!(result_str.contains("UTILIZES"), "Result should contain relation type");
}
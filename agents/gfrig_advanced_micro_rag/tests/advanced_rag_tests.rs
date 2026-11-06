use gfrig_advanced_micro_rag::{cosine_similarity, AdvancedRagDatabase};
use tempfile::NamedTempFile;

#[test]
fn test_cosine_similarity_advanced() {
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![1.0, 0.0, 0.0];
    let v3 = vec![0.0, 1.0, 0.0];

    assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-5);
    assert!((cosine_similarity(&v1, &v3)).abs() < 1e-5);
}

#[tokio::test]
async fn test_advanced_rag_db_operations() {
    let temp_db = NamedTempFile::new().unwrap();
    // Added .await and .unwrap() to the asynchronous init()
    let db = AdvancedRagDatabase::init(temp_db.path()).await.unwrap();

    // 1. Inserting a document (.await)
    let doc_id = db.insert_document("Advanced multi-agent framework test content.", "test_source.txt").await.unwrap();
    assert!(doc_id > 0);

    // 2. Insertion of a vector (chunk) (.await)
    let embedding = vec![0.8, 0.2, 0.1];
    let chunk_res = db.insert_chunk(doc_id, "Advanced multi-agent chunk", &embedding).await;
    assert!(chunk_res.is_ok());

    // 3. Similarity-based search (.await)
    let query_embedding = vec![0.75, 0.25, 0.1];
    let search_res = db.search(&query_embedding, 1).await.unwrap();
    
    assert_eq!(search_res.len(), 1);
    assert_eq!(search_res[0].0, "Advanced multi-agent chunk");
    assert!(search_res[0].1 > 0.9, "Similarity score should be high");
}
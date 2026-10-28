use local_micro_rag::{chunk_text, cosine_similarity, RagDatabase};
use tempfile::NamedTempFile;

#[test]
fn test_chunking_logic() {
    let text = "Hello world! This is a test string for Rust chunking verification.";
    let chunks = chunk_text(text, 20, 5);
    
    assert!(!chunks.is_empty());
    assert!(chunks[0].len() <= 20);
}

#[test]
fn test_cosine_similarity() {
    let vec_a = vec![1.0, 0.0, 0.0];
    let vec_b = vec![1.0, 0.0, 0.0];
    let vec_c = vec![0.0, 1.0, 0.0];

    // Identical vectors -> similarity 1.0
    assert!((cosine_similarity(&vec_a, &vec_b) - 1.0).abs() < 1e-5);
    // Orthogonal vectors -> similarity 0.0
    assert!((cosine_similarity(&vec_a, &vec_c)).abs() < 1e-5);
}

#[test]
fn test_sqlite_db_operations() {
    let temp_db = NamedTempFile::new().unwrap();
    let mut db = RagDatabase::init(temp_db.path()).unwrap();

    let doc_id = db.insert_document("test.txt", "Full content", "Short summary").unwrap();
    assert!(doc_id > 0);

    let emb = vec![0.1, 0.2, 0.3];
    let chunk_res = db.insert_chunk(doc_id, "Full content chunk", &emb);
    assert!(chunk_res.is_ok());

    let results = db.search(&emb, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].path, "test.txt");
}
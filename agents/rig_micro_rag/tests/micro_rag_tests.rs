use rig_micro_rag::{cosine_similarity, MicroRagDatabase};
use tempfile::NamedTempFile;

#[test]
fn test_cosine_similarity() {
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![1.0, 0.0, 0.0];
    let v3 = vec![0.0, 1.0, 0.0];

    assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-5);
    assert!((cosine_similarity(&v1, &v3)).abs() < 1e-5);
}

#[test]
fn test_micro_rag_db() {
    let temp_db = NamedTempFile::new().unwrap();
    let mut db = MicroRagDatabase::init(temp_db.path()).unwrap();

    let doc_id = db.insert_document("Test content", "test.txt", 1).unwrap();
    assert!(doc_id > 0);

    let emb = vec![0.5, 0.5, 0.0];
    let res = db.insert_chunk(doc_id, "Test content chunk", &emb);
    assert!(res.is_ok());

    let search_res = db.search(&emb, 1).unwrap();
    assert_eq!(search_res.len(), 1);
    assert_eq!(search_res[0].0, "Test content chunk");
}
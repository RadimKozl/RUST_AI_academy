use rig_chat_memory::SqliteMemoryManager;

#[test]
fn test_sqlite_memory_manager() {
    let db_path = "test_memory.db";
    let session_id = "test_session";

    let manager = SqliteMemoryManager::new(db_path).unwrap();
    manager.clear_session(session_id).unwrap();

    manager.add_message(session_id, "user", "Hello").unwrap();
    manager.add_message(session_id, "assistant", "Hi there!").unwrap();

    let history = manager.get_messages(session_id).unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].role, "user");
    assert_eq!(history[0].content, "Hello");
    assert_eq!(history[1].role, "assistant");
    assert_eq!(history[1].content, "Hi there!");

    manager.clear_session(session_id).unwrap();
    let empty_history = manager.get_messages(session_id).unwrap();
    assert_eq!(empty_history.len(), 0);

    // Cleanup after test
    let _ = std::fs::remove_file(db_path);
}
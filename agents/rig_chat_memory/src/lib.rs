use anyhow::{Context, Result};
use rig::providers::ollama;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

/// Message record storable in the DB and sent to the XAI audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageRecord {
    pub role: String,
    pub content: String,
}

/// Memory step recording for XAI / Neo4j memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStepRecord {
    pub step_name: String,
    pub input_value: String,
    pub generated_output: String,
}

/// Result of one round of conversation with memory
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryChainResult {
    pub session_id: String,
    pub used_summary: Option<String>,
    pub recent_messages_count: usize,
    pub final_output: String,
    pub history: Vec<MemoryStepRecord>,
}

/// SQLite persistence manager for Chat History
pub struct SqliteMemoryManager {
    db_path: String,
}

impl SqliteMemoryManager {
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)
            .with_context(|| format!("Failed to open SQLite DB at {}", db_path))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        Ok(Self {
            db_path: db_path.to_string(),
        })
    }

    /// Clearing history for the given session
    pub fn clear_session(&self, session_id: &str) -> Result<()> {
        let conn = Connection::open(&self.db_path)?;
        conn.execute("DELETE FROM chat_history WHERE session_id = ?1", params![session_id])?;
        Ok(())
    }

    /// Saving the message
    pub fn add_message(&self, session_id: &str, role: &str, content: &str) -> Result<()> {
        let conn = Connection::open(&self.db_path)?;
        conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?1, ?2, ?3)",
            params![session_id, role, content],
        )?;
        Ok(())
    }

    /// Loading all session messages
    pub fn get_messages(&self, session_id: &str) -> Result<Vec<ChatMessageRecord>> {
        let conn = Connection::open(&self.db_path)?;
        let mut stmt = conn.prepare(
            "SELECT role, content FROM chat_history WHERE session_id = ?1 ORDER BY id ASC",
        )?;

        let message_iter = stmt.query_map(params![session_id], |row| {
            Ok(ChatMessageRecord {
                role: row.get(0)?,
                content: row.get(1)?,
            })
        })?;

        let mut messages = Vec::new();
        for msg in message_iter {
            messages.push(msg?);
        }
        Ok(messages)
    }
}

/// Summarization of older chat history via an LLM agent
pub async fn summarize_history(
    model_name: &str,
    messages: &[ChatMessageRecord],
) -> Result<String> {
    let client = ollama::Ollama::new();
    let model = client.completion(model_name);

    let agent = rig::agent::AgentBuilder::new(model)
        .preamble("You are an expert at summarizing conversations concisely.")
        .build();

    let mut formatted_history = String::new();
    for msg in messages {
        formatted_history.push_str(&format!("{}: {}\n", msg.role.to_uppercase(), msg.content));
    }

    let prompt = format!(
        "Summarize the following conversation in 1-2 concise sentences. Maintain key factual context.\n\nCONVERSATION:\n{}\n\nSUMMARY:",
        formatted_history
    );

    let response = agent.prompt(&prompt).await?;
    Ok(response.output().trim().to_string())
}

/// Main function for initiating a conversational turn with hybrid memory (Trimming + Summarization + Persistence)
pub async fn run_hybrid_chat_turn(
    db_manager: &SqliteMemoryManager,
    session_id: &str,
    base_model_name: &str,
    summary_model_name: &str,
    user_input: &str,
) -> Result<MemoryChainResult> {
    let client = ollama::Ollama::new();
    let model = client.completion(base_model_name);

    let mut audit_history = Vec::new();

    // 1. Loading existing history from SQLite
    let existing_messages = db_manager.get_messages(session_id)?;

    // 2. Hybrid Memory Refinement (Trimming & Summarization)
    let recent_count = 4;
    let (used_summary, recent_messages) = if existing_messages.len() > recent_count {
        let older_messages = &existing_messages[..existing_messages.len() - recent_count];
        let recent = &existing_messages[existing_messages.len() - recent_count..];

        let summary = summarize_history(summary_model_name, older_messages).await?;

        audit_history.push(MemoryStepRecord {
            step_name: "history_summarization".to_string(),
            input_value: format!("Summarizing {} older messages", older_messages.len()),
            generated_output: summary.clone(),
        });

        (Some(summary), recent.to_vec())
    } else {
        (None, existing_messages.clone())
    };

    // 3. Construction of the system prompt and context
    let mut preamble = String::from("You are a helpful and intelligent assistant.");
    if let Some(ref sum) = used_summary {
        preamble.push_str(&format!("\n\nCONVERSATION SUMMARY SO FAR:\n{}", sum));
    }

    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(&preamble)
        .build();

    // Constructing the final prompt containing the truncated messages and the new question
    let mut full_prompt = String::new();
    if !recent_messages.is_empty() {
        full_prompt.push_str("Recent Conversation History:\n");
        for msg in &recent_messages {
            full_prompt.push_str(&format!("{}: {}\n", msg.role.to_uppercase(), msg.content));
        }
        full_prompt.push_str("\n");
    }
    full_prompt.push_str(&format!("USER: {}\nASSISTANT:", user_input));

    // 4. Calling the LLM to generate a response
    let response = agent.prompt(&full_prompt).await?;
    let final_output = response.output().trim().to_string();

    // 5. Saving the user's query and the agent's response to an SQLite database
    db_manager.add_message(session_id, "user", user_input)?;
    db_manager.add_message(session_id, "assistant", &final_output)?;

    audit_history.push(MemoryStepRecord {
        step_name: "llm_chat_turn".to_string(),
        input_value: user_input.to_string(),
        generated_output: final_output.clone(),
    });

    Ok(MemoryChainResult {
        session_id: session_id.to_string(),
        used_summary,
        recent_messages_count: recent_messages.len(),
        final_output,
        history: audit_history,
    })
}
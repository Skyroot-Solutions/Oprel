import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from oprel.core.config import Config

CONFIG = Config()
DB_PATH = CONFIG.cache_dir / "chat_history.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    # Create conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            model_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Create messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)
    # Create users table for profile info
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            avatar_initials TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Create user_settings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            temperature REAL,
            top_p REAL,
            top_k INTEGER,
            repeat_penalty REAL,
            max_tokens INTEGER,
            system_instruction TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Create required indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_time ON messages(created_at)")
    
    conn.commit()
    conn.close()

def create_conversation(model_id: str, title: str = "New Chat") -> str:
    conn = get_db()
    cursor = conn.cursor()
    conv_id = f"chat_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO conversations (id, title, model_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (conv_id, title, model_id, now, now)
    )
    conn.commit()
    conn.close()
    return conv_id

def delete_conversation(conversation_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()

def rename_conversation(conversation_id: str, new_title: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", 
                   (new_title, datetime.now().isoformat(), conversation_id))
    conn.commit()
    conn.close()

def reset_conversation(conversation_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    now = datetime.now().isoformat()
    cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    conn.commit()
    conn.close()

def add_message(conversation_id: str, role: str, content: Any):
    import json
    raw_content = content
    if not isinstance(content, str):
        content = json.dumps(content)
        
    conn = get_db()
    cursor = conn.cursor()
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (msg_id, conversation_id, role, content, now)
    )
    # Update timestamp and possibly title based on first user message
    cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    
    # Optional: Update title if it's the first user message
    if role == "user":
        # If content is a list (vision), use the text part if available
        title_content = raw_content
        if isinstance(raw_content, list):
            try:
                # Expecting [{type: text, text: ...}, ...]
                for item in raw_content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        title_content = item.get('text', '')
                        break
            except:
                pass

        cursor.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND role = 'user'", (conversation_id,))
        count = cursor.fetchone()[0]
        if count == 1:
            title = str(title_content)[:30] + "..." if len(str(title_content)) > 30 else str(title_content)
            cursor.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
            
    conn.commit()
    conn.close()

def get_conversation_messages(conversation_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    import json
    messages = []
    for row in rows:
        content = row["content"]
        if isinstance(content, str) and (content.startswith('[') or content.startswith('{')):
            try:
                content = json.loads(content)
            except:
                pass
        messages.append({"role": row["role"], "content": content})
    return messages

def list_conversations():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.id, c.title, c.model_id, c.created_at, c.updated_at, COUNT(m.id) as message_count
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_id
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        LIMIT 100
    """)
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "model_id": row["model_id"],
            "created_at": row["created_at"],
            "last_updated": row["updated_at"],
            "message_count": row["message_count"],
        }
        for row in rows
    ]

def get_active_conversation_count():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM conversations")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_user():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name, role, avatar_initials FROM users WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"name": row["name"], "role": row["role"], "initials": row["avatar_initials"]}
    return None

def set_user(name: str, role: str):
    conn = get_db()
    cursor = conn.cursor()
    # Get initials from name
    initials = "".join([n[0] for n in name.split()[:2]]).upper()
    
    cursor.execute("""
        INSERT OR REPLACE INTO users (id, name, role, avatar_initials, updated_at)
        VALUES (1, ?, ?, ?, ?)
    """, (name, role, initials, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return {"name": name, "role": role, "initials": initials}

def get_user_settings():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_settings WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def set_user_settings(settings: dict):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO user_settings (id, temperature, top_p, top_k, repeat_penalty, max_tokens, system_instruction, updated_at)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?)
    """, (
        settings.get("temperature"),
        settings.get("top_p"),
        settings.get("top_k"),
        settings.get("repeat_penalty"),
        settings.get("max_tokens"),
        settings.get("system_instruction"),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()
    return settings

# Initialize DB when module is loaded
init_db()

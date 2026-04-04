import sqlite3
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List
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
    # Create download_logs table for persistent download history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS download_logs (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            model_name TEXT,
            quantization TEXT,
            status TEXT NOT NULL,
            size_bytes INTEGER DEFAULT 0,
            duration_seconds REAL DEFAULT 0,
            error TEXT,
            started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_download_logs_time ON download_logs(started_at DESC)")
    
    # Create inference_logs table for analytics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inference_logs (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            latency_ms REAL DEFAULT 0,
            tps REAL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_inference_logs_time ON inference_logs(created_at DESC)")

    # Create provider_configs table — API keys & enabled models for external providers
    # api_key is stored as-is; for production deployments consider encrypting at rest.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS provider_configs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            api_key TEXT NOT NULL DEFAULT '',
            base_url TEXT DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            enabled_model_ids TEXT NOT NULL DEFAULT '[]',
            available_model_ids TEXT NOT NULL DEFAULT '[]',
            last_fetched TEXT DEFAULT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def create_conversation(model_id: str, title: str = "New Chat", conversation_id: str = None) -> str:
    """Create a new conversation with optional custom ID"""
    conn = get_db()
    cursor = conn.cursor()
    conv_id = conversation_id if conversation_id else f"chat_{uuid.uuid4().hex[:12]}"
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


def save_download_log(model_id: str, model_name: str, quantization: str, status: str,
                      size_bytes: int = 0, duration_seconds: float = 0,
                      error: str = None, started_at: str = None, completed_at: str = None):
    """Persist a download event to the logs table."""
    conn = get_db()
    cursor = conn.cursor()
    log_id = f"dl_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO download_logs
            (id, model_id, model_name, quantization, status, size_bytes,
             duration_seconds, error, started_at, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        log_id, model_id, model_name, quantization, status,
        size_bytes, duration_seconds, error,
        started_at or now, completed_at or (now if status != 'downloading' else None)
    ))
    conn.commit()
    conn.close()
    return log_id


def list_download_logs(limit: int = 100):
    """Return recent download log entries, newest first."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, model_id, model_name, quantization, status, size_bytes,
               duration_seconds, error, started_at, completed_at
        FROM download_logs
        ORDER BY started_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "model_id": r["model_id"],
            "model_name": r["model_name"],
            "quantization": r["quantization"],
            "status": r["status"],
            "size_bytes": r["size_bytes"],
            "duration_seconds": r["duration_seconds"],
            "error": r["error"],
            "started_at": r["started_at"],
            "completed_at": r["completed_at"],
        }
        for r in rows
    ]


def add_inference_log(model_id: str, prompt_tokens: int, completion_tokens: int, latency_ms: float, tps: float):
    """Log an inference event for analytics."""
    conn = get_db()
    cursor = conn.cursor()
    log_id = f"inf_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO inference_logs (id, model_id, prompt_tokens, completion_tokens, latency_ms, tps, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (log_id, model_id, prompt_tokens, completion_tokens, latency_ms, tps, now))
    conn.commit()
    conn.close()
    return log_id


def get_inference_summary(days: int = 7):
    """Retrieve summary stats for the last N days."""
    conn = get_db()
    cursor = conn.cursor()
    # Simplified time calculation for sqlite
    cursor.execute("""
        SELECT 
            COUNT(*) as total_requests,
            SUM(prompt_tokens) as total_prompt_tokens,
            SUM(completion_tokens) as total_completion_tokens,
            AVG(latency_ms) as avg_latency,
            AVG(tps) as avg_tps,
            model_id
        FROM inference_logs
        WHERE created_at >= datetime('now', '-' || ? || ' days')
        GROUP BY model_id
    """, (days,))
    rows = cursor.fetchall()
    
    # Get hourly distribution for charts
    cursor.execute("""
        SELECT 
            strftime('%Y-%m-%d %H:00:00', created_at) as hour,
            SUM(prompt_tokens + completion_tokens) as total_tokens,
            AVG(tps) as tps
        FROM inference_logs
        WHERE created_at >= datetime('now', '-' || ? || ' days')
        GROUP BY hour
        ORDER BY hour ASC
    """, (days,))
    timeline = cursor.fetchall()
    
    conn.close()
    return {
        "models": [dict(r) for r in rows],
        "timeline": [dict(t) for t in timeline]
    }


# ──────────────────────────────────────────────────────────────────────────────
# Provider config CRUD
# ──────────────────────────────────────────────────────────────────────────────

def list_providers() -> list:
    """Return all configured external AI providers."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM provider_configs ORDER BY updated_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_provider(r) for r in rows]


def get_provider(provider_id: str) -> Optional[dict]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM provider_configs WHERE id = ?", (provider_id,))
    row = cursor.fetchone()
    conn.close()
    return _row_to_provider(row) if row else None


def upsert_provider(p: dict) -> dict:
    """Insert or update a provider config. `p` must have at least 'id', 'name', 'type', 'api_key'."""
    conn = get_db()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO provider_configs
            (id, name, type, api_key, base_url, enabled, enabled_model_ids, available_model_ids, last_fetched, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            type = excluded.type,
            api_key = excluded.api_key,
            base_url = excluded.base_url,
            enabled = excluded.enabled,
            enabled_model_ids = excluded.enabled_model_ids,
            available_model_ids = excluded.available_model_ids,
            last_fetched = excluded.last_fetched,
            updated_at = excluded.updated_at
    """, (
        p["id"], p["name"], p["type"],
        p.get("api_key", ""),
        p.get("base_url", ""),
        1 if p.get("enabled", True) else 0,
        json.dumps(p.get("enabled_model_ids", [])),
        json.dumps(p.get("available_model_ids", [])),
        p.get("last_fetched"),
        now,
    ))
    conn.commit()
    conn.close()
    return get_provider(p["id"])


def delete_provider(provider_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM provider_configs WHERE id = ?", (provider_id,))
    conn.commit()
    conn.close()


def _row_to_provider(row) -> dict:
    d = dict(row)
    d["enabled"] = bool(d.get("enabled", 1))
    try:
        d["enabled_model_ids"] = json.loads(d.get("enabled_model_ids") or "[]")
    except Exception:
        d["enabled_model_ids"] = []
    try:
        d["available_model_ids"] = json.loads(d.get("available_model_ids") or "[]")
    except Exception:
        d["available_model_ids"] = []
    return d

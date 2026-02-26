"""
Persistent Storage - M2 Feature
Save execution history and data to file/database
"""

import json
import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PersistentStorage:
    """
    Persistent storage for workflow data using SQLite.
    Stores execution history, workflow definitions, and settings.
    """
    
    def __init__(self, db_path: str = "data/workflow_data.db"):
        self.db_path = db_path
        self._ensure_directory()
        self._init_database()
        
    def _ensure_directory(self):
        """Ensure the data directory exists"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
    def _init_database(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Execution history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_history (
                    execution_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    workflow_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    duration_seconds REAL,
                    output TEXT,
                    error TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Workflow definitions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    workflow_type TEXT NOT NULL,
                    workflow_json TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Webhook events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS webhook_events (
                    event_id TEXT PRIMARY KEY,
                    webhook_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    event_type TEXT,
                    payload TEXT,
                    processed INTEGER DEFAULT 0,
                    processed_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Notifications table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    notification_id TEXT PRIMARY KEY,
                    notification_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT,
                    data TEXT,
                    is_read INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_exec_workflow ON execution_history(workflow_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_exec_status ON execution_history(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_webhook_source ON webhook_events(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_notif_unread ON notifications(is_read)")
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
            
    @contextmanager
    def _get_connection(self):
        """Get database connection context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
            
    # ==================== Execution History ====================
    
    def save_execution(self, execution: Dict[str, Any]) -> bool:
        """Save an execution result"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO execution_history 
                    (execution_id, workflow_id, workflow_type, status, started_at, 
                     completed_at, duration_seconds, output, error, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    execution["execution_id"],
                    execution["workflow_id"],
                    execution["workflow_type"],
                    execution["status"],
                    execution["started_at"],
                    execution.get("completed_at"),
                    execution.get("duration_seconds"),
                    json.dumps(execution.get("output")),
                    execution.get("error"),
                    json.dumps(execution.get("metadata", {}))
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save execution: {e}")
            return False
            
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get an execution by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM execution_history WHERE execution_id = ?", (execution_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_execution(row)
        return None
        
    def get_executions_by_workflow(self, workflow_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get executions for a specific workflow"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM execution_history 
                WHERE workflow_id = ? 
                ORDER BY started_at DESC 
                LIMIT ?
            """, (workflow_id, limit))
            return [self._row_to_execution(row) for row in cursor.fetchall()]
            
    def get_recent_executions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most recent executions"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM execution_history 
                ORDER BY started_at DESC 
                LIMIT ?
            """, (limit,))
            return [self._row_to_execution(row) for row in cursor.fetchall()]
            
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total count
            cursor.execute("SELECT COUNT(*) as total FROM execution_history")
            total = cursor.fetchone()["total"]
            
            # By status
            cursor.execute("""
                SELECT status, COUNT(*) as count 
                FROM execution_history 
                GROUP BY status
            """)
            by_status = {row["status"]: row["count"] for row in cursor.fetchall()}
            
            # By type
            cursor.execute("""
                SELECT workflow_type, COUNT(*) as count 
                FROM execution_history 
                GROUP BY workflow_type
            """)
            by_type = {row["workflow_type"]: row["count"] for row in cursor.fetchall()}
            
            # Average duration
            cursor.execute("""
                SELECT AVG(duration_seconds) as avg_duration 
                FROM execution_history 
                WHERE duration_seconds IS NOT NULL
            """)
            avg_duration = cursor.fetchone()["avg_duration"] or 0
            
            return {
                "total_executions": total,
                "by_status": by_status,
                "by_type": by_type,
                "average_duration_seconds": round(avg_duration, 2),
                "success_rate": round(by_status.get("completed", 0) / total * 100, 2) if total > 0 else 0
            }
            
    def _row_to_execution(self, row) -> Dict[str, Any]:
        """Convert database row to execution dict"""
        return {
            "execution_id": row["execution_id"],
            "workflow_id": row["workflow_id"],
            "workflow_type": row["workflow_type"],
            "status": row["status"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "duration_seconds": row["duration_seconds"],
            "output": json.loads(row["output"]) if row["output"] else None,
            "error": row["error"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
        }
        
    # ==================== Workflows ====================
    
    def save_workflow(
        self,
        workflow_id: str,
        name: str,
        workflow_type: str,
        workflow_json: Dict[str, Any],
        description: str = ""
    ) -> bool:
        """Save a workflow definition"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if exists to increment version
                cursor.execute("SELECT version FROM workflows WHERE workflow_id = ?", (workflow_id,))
                existing = cursor.fetchone()
                version = (existing["version"] + 1) if existing else 1
                
                cursor.execute("""
                    INSERT OR REPLACE INTO workflows 
                    (workflow_id, name, description, workflow_type, workflow_json, version, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    workflow_id,
                    name,
                    description,
                    workflow_type,
                    json.dumps(workflow_json),
                    version,
                    datetime.now().isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save workflow: {e}")
            return False
            
    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get a workflow by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "workflow_id": row["workflow_id"],
                    "name": row["name"],
                    "description": row["description"],
                    "workflow_type": row["workflow_type"],
                    "workflow_json": json.loads(row["workflow_json"]),
                    "version": row["version"],
                    "is_active": bool(row["is_active"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
        return None
        
    def list_workflows(self, workflow_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all workflows"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if workflow_type:
                cursor.execute("""
                    SELECT workflow_id, name, description, workflow_type, version, is_active, created_at 
                    FROM workflows 
                    WHERE workflow_type = ? AND is_active = 1
                    ORDER BY updated_at DESC
                """, (workflow_type,))
            else:
                cursor.execute("""
                    SELECT workflow_id, name, description, workflow_type, version, is_active, created_at 
                    FROM workflows 
                    WHERE is_active = 1
                    ORDER BY updated_at DESC
                """)
            return [dict(row) for row in cursor.fetchall()]
            
    def delete_workflow(self, workflow_id: str) -> bool:
        """Soft delete a workflow"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE workflows SET is_active = 0 WHERE workflow_id = ?", (workflow_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete workflow: {e}")
            return False
            
    # ==================== Webhook Events ====================
    
    def save_webhook_event(
        self,
        event_id: str,
        webhook_id: str,
        source: str,
        payload: Dict[str, Any],
        event_type: str = ""
    ) -> bool:
        """Save a webhook event"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO webhook_events 
                    (event_id, webhook_id, source, event_type, payload)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    event_id,
                    webhook_id,
                    source,
                    event_type,
                    json.dumps(payload)
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save webhook event: {e}")
            return False
            
    def get_webhook_events(self, source: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get webhook events"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if source:
                cursor.execute("""
                    SELECT * FROM webhook_events 
                    WHERE source = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (source, limit))
            else:
                cursor.execute("""
                    SELECT * FROM webhook_events 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
            return [{
                "event_id": row["event_id"],
                "webhook_id": row["webhook_id"],
                "source": row["source"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload"]) if row["payload"] else {},
                "processed": bool(row["processed"]),
                "processed_at": row["processed_at"],
                "created_at": row["created_at"]
            } for row in cursor.fetchall()]
            
    def mark_webhook_processed(self, event_id: str) -> bool:
        """Mark a webhook event as processed"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE webhook_events 
                    SET processed = 1, processed_at = ? 
                    WHERE event_id = ?
                """, (datetime.now().isoformat(), event_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to mark webhook processed: {e}")
            return False
            
    # ==================== Notifications ====================
    
    def save_notification(
        self,
        notification_id: str,
        notification_type: str,
        title: str,
        message: str = "",
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save a notification"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO notifications 
                    (notification_id, notification_type, title, message, data)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    notification_id,
                    notification_type,
                    title,
                    message,
                    json.dumps(data) if data else None
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save notification: {e}")
            return False
            
    def get_notifications(self, unread_only: bool = False, limit: int = 50) -> List[Dict[str, Any]]:
        """Get notifications"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if unread_only:
                cursor.execute("""
                    SELECT * FROM notifications 
                    WHERE is_read = 0 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT * FROM notifications 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
            return [{
                "notification_id": row["notification_id"],
                "notification_type": row["notification_type"],
                "title": row["title"],
                "message": row["message"],
                "data": json.loads(row["data"]) if row["data"] else None,
                "is_read": bool(row["is_read"]),
                "created_at": row["created_at"]
            } for row in cursor.fetchall()]
            
    def mark_notification_read(self, notification_id: str) -> bool:
        """Mark a notification as read"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE notifications SET is_read = 1 WHERE notification_id = ?", (notification_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to mark notification read: {e}")
            return False
            
    def get_unread_count(self) -> int:
        """Get count of unread notifications"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM notifications WHERE is_read = 0")
            return cursor.fetchone()["count"]
            
    # ==================== Settings ====================
    
    def set_setting(self, key: str, value: Any) -> bool:
        """Set a setting value"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                """, (key, json.dumps(value), datetime.now().isoformat()))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to set setting: {e}")
            return False
            
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row["value"])
        return default
        
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM settings")
            return {row["key"]: json.loads(row["value"]) for row in cursor.fetchall()}


# Singleton instance
_storage: Optional[PersistentStorage] = None


def get_storage() -> PersistentStorage:
    """Get the global storage instance"""
    global _storage
    if _storage is None:
        _storage = PersistentStorage()
    return _storage

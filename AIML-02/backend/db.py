"""SQLite worklist: one row per study, review fields filled in by the doctor."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT, patient_id TEXT, patient_name TEXT, modality TEXT, orthanc_id TEXT,
  created_at TEXT, ai_label TEXT, ai_prob REAL,
  status TEXT DEFAULT 'pending_review',
  review_decision TEXT, final_label TEXT, review_notes TEXT, reviewer TEXT, reviewed_at TEXT
)"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute(SCHEMA)
            self.conn.commit()

    def add(self, **f) -> int:
        f["created_at"] = now()
        cols = ",".join(f)
        with self.lock:
            cur = self.conn.execute(f"INSERT INTO studies ({cols}) VALUES ({','.join('?' * len(f))})", list(f.values()))
            self.conn.commit()
            return cur.lastrowid

    def get(self, sid: int):
        with self.lock:
            r = self.conn.execute("SELECT * FROM studies WHERE id=?", (sid,)).fetchone()
        return dict(r) if r else None

    def list(self, status: str | None = None):
        q, args = "SELECT * FROM studies", ()
        if status:
            q, args = q + " WHERE status=?", (status,)
        with self.lock:
            return [dict(r) for r in self.conn.execute(q + " ORDER BY id DESC", args).fetchall()]

    def review(self, sid: int, decision: str, final_label: str, notes: str, reviewer: str) -> bool:
        with self.lock:
            cur = self.conn.execute(
                "UPDATE studies SET status='reviewed', review_decision=?, final_label=?, review_notes=?, reviewer=?, reviewed_at=? WHERE id=?",
                (decision, final_label, notes, reviewer, now(), sid))
            self.conn.commit()
            return cur.rowcount > 0

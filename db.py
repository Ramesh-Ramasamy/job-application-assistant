"""Tiny SQLite application tracker. Rows are only written after the user approves."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

STATUSES = ["Drafted", "Applied", "Referral asked", "Interviewing", "Offer", "Rejected"]


def connect(path: str = "applications.db") -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.execute(
        """CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT, company TEXT, role TEXT, score INTEGER,
            status TEXT, url TEXT, payload TEXT)"""
    )
    return con


def save(con: sqlite3.Connection, state: dict, url: str = "", status: str = "Drafted") -> int:
    req, match = state.get("requirements", {}), state.get("match", {})
    payload = {k: state.get(k) for k in ("requirements", "match", "bullets", "cover_letter", "outreach", "questions")}
    cur = con.execute(
        "INSERT INTO applications (created_at, company, role, score, status, url, payload) VALUES (?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), req.get("company", ""), req.get("role", ""),
         match.get("score", 0), status, url, json.dumps(payload)),
    )
    con.commit()
    return cur.lastrowid


def list_all(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute("SELECT id, created_at, company, role, score, status, url FROM applications ORDER BY id DESC")
    cols = ["id", "created_at", "company", "role", "score", "status", "url"]
    return [dict(zip(cols, r)) for r in rows.fetchall()]


def update_status(con: sqlite3.Connection, app_id: int, status: str) -> None:
    if status not in STATUSES:
        raise ValueError(status)
    con.execute("UPDATE applications SET status=? WHERE id=?", (status, app_id))
    con.commit()

import sqlite3
from pathlib import Path
from datetime import datetime

from typing import Optional

DB_PATH = Path("leaderboard.sqlite3")

def _conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with _conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day TEXT NOT NULL,
            words TEXT NOT NULL,
            player TEXT NOT NULL,
            score INTEGER NOT NULL,
            solution TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_scores_day ON scores(day)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_scores_words ON scores(words)")
        c.commit()

def submit_score(
        day: Optional[str],
        puzzle_id: str,
        words: str,
        difficulty: Optional[str],
        player: str,
        score: int,
        solution: str,
):
    with _conn() as c:
        c.execute(
            "INSERT INTO scores(day, words, player, score, solution, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (day, words, player, int(score), solution, datetime.utcnow().isoformat() + "Z")
        )
        c.commit()

def top_scores(
        day: Optional[str] = None,
        puzzle_id: Optional[str] = None,
        limit: int = 20,
):

    q = "SELECT day, words, player, score, solution, created_at FROM scores"
    params = []
    if day:
        q += " WHERE day=?"
        params.append(day)
    q += " ORDER BY score DESC, created_at ASC LIMIT ?"
    params.append(limit)

    with _conn() as c:
        rows = c.execute(q, params).fetchall()

    return [
        {"day": r[0], "words": r[1], "player": r[2], "score": r[3], "solution": r[4], "created_at": r[5]}
        for r in rows
    ]

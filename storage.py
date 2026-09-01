"""SQLite storage layer for the English tutor plugin.

All timestamps are stored as ``YYYY-MM-DD`` (date) / ``YYYY-MM-DD HH:MM:SS``
(created_at) local-time strings so that "what did I learn yesterday" style
queries are simple SQL.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Light spaced-repetition schedule: interval in days after each successful review.
REVIEW_INTERVALS = [1, 2, 4, 7, 15, 30]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS archive_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    umo TEXT NOT NULL,
    date TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_archive_umo_date ON archive_messages(umo, date);

CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    umo TEXT NOT NULL,
    first_date TEXT NOT NULL,
    last_date TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'open',
    original TEXT NOT NULL,
    corrected TEXT NOT NULL DEFAULT '',
    explanation TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    next_review TEXT NOT NULL DEFAULT '',
    interval_days INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_errors_umo ON errors(umo, status);

CREATE TABLE IF NOT EXISTS sentences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    umo TEXT NOT NULL,
    date TEXT NOT NULL,
    sentence TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'user',
    tags TEXT NOT NULL DEFAULT '',
    context TEXT NOT NULL DEFAULT '',
    dialog TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sentences_umo_date ON sentences(umo, date);

CREATE TABLE IF NOT EXISTS vocab (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    umo TEXT NOT NULL,
    date TEXT NOT NULL,
    word TEXT NOT NULL,
    meaning TEXT NOT NULL DEFAULT '',
    example TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'ai',
    next_review TEXT NOT NULL DEFAULT '',
    interval_days INTEGER NOT NULL DEFAULT 1,
    context TEXT NOT NULL DEFAULT '',
    dialog TEXT NOT NULL DEFAULT '',
    UNIQUE(umo, word)
);

CREATE TABLE IF NOT EXISTS daily_practice (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    umo TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT 'sentences',
    items_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'generated',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_practice_date ON daily_practice(date);
"""


def advance_interval(current_days: int, remembered: bool) -> int:
    """Return the next review interval in days.

    Args:
        current_days: Current interval of the item.
        remembered: Whether the user remembered the item.

    Returns:
        The new interval in days. Forgetting resets to the shortest interval.
    """
    if not remembered:
        return REVIEW_INTERVALS[0]
    try:
        idx = REVIEW_INTERVALS.index(current_days)
    except ValueError:
        idx = 0
    return REVIEW_INTERVALS[min(idx + 1, len(REVIEW_INTERVALS) - 1)]


class TutorStore:
    """Thin synchronous DAO over one SQLite file. Safe for cross-thread use."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after v0.1.0 to pre-existing databases."""
        for table in ("sentences", "vocab"):
            cols = [r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")]
            for col in ("context", "dialog"):
                if cols and col not in cols:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                    )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _run(
        self,
        sql: str,
        params: tuple = (),
        *,
        fetch_one: bool = False,
        fetch_all: bool = False,
    ) -> Any:
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = None
            if fetch_one:
                rows = cur.fetchone()
            elif fetch_all:
                rows = cur.fetchall()
            self._conn.commit()
            return rows if (fetch_one or fetch_all) else cur

    @staticmethod
    def _rows_to_dicts(rows: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in rows] if rows else []

    # ==================== archive ====================

    def add_archive(self, umo: str, date: str, role: str, content: str) -> None:
        self._run(
            "INSERT INTO archive_messages (umo, date, role, content, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (umo, date, role, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )

    def list_archive(
        self,
        date: str | None = None,
        umo: str | None = None,
        keyword: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM archive_messages WHERE 1=1"
        params: list[Any] = []
        if date:
            sql += " AND date = ?"
            params.append(date)
        if umo:
            sql += " AND (umo = ? OR umo = '')"
            params.append(umo)
        if keyword:
            sql += " AND content LIKE ?"
            params.append(f"%{keyword}%")
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return self._rows_to_dicts(self._run(sql, tuple(params), fetch_all=True))

    def count_archive(
        self,
        date: str | None = None,
        umo: str | None = None,
        keyword: str = "",
    ) -> int:
        sql = "SELECT COUNT(*) AS c FROM archive_messages WHERE 1=1"
        params: list[Any] = []
        if date:
            sql += " AND date = ?"
            params.append(date)
        if umo:
            sql += " AND (umo = ? OR umo = '')"
            params.append(umo)
        if keyword:
            sql += " AND content LIKE ?"
            params.append(f"%{keyword}%")
        row = self._run(sql, tuple(params), fetch_one=True)
        return int(row["c"]) if row else 0

    def recent_archive(self, umo: str, limit: int) -> list[dict[str, Any]]:
        """Return the latest ``limit`` messages of a session in chronological order."""
        rows = self._run(
            "SELECT * FROM (SELECT * FROM archive_messages WHERE umo = ?"
            " ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (umo, limit),
            fetch_all=True,
        )
        return self._rows_to_dicts(rows)

    def delete_archive(self, row_id: int) -> None:
        self._run("DELETE FROM archive_messages WHERE id = ?", (row_id,))

    def archive_day_counts(self, days: int = 7) -> list[dict[str, Any]]:
        since = (datetime.now() - timedelta(days=days - 1)).strftime("%Y-%m-%d")
        rows = self._run(
            "SELECT date, COUNT(*) AS c FROM archive_messages"
            " WHERE date >= ? GROUP BY date ORDER BY date ASC",
            (since,),
            fetch_all=True,
        )
        return self._rows_to_dicts(rows)

    def cleanup_archive(self, keep_days: int) -> int:
        if keep_days <= 0:
            return 0
        before = (datetime.now() - timedelta(days=keep_days)).strftime("%Y-%m-%d")
        cur = self._run("DELETE FROM archive_messages WHERE date < ?", (before,))
        return cur.rowcount if cur and cur.rowcount and cur.rowcount > 0 else 0

    def cleanup_non_english_archive(self, is_english) -> int:
        """Delete archived messages that fail the English check.

        One-shot migration for archives recorded by older versions, which
        stored share links and system-injected prompts as English rounds.

        Args:
            is_english: Predicate ``str -> bool`` deciding if content stays.

        Returns:
            The number of deleted rows.
        """
        rows = self._rows_to_dicts(
            self._run("SELECT id, content FROM archive_messages", fetch_all=True)
        )
        removed = 0
        for row in rows:
            if not is_english(str(row["content"] or "")):
                self._run("DELETE FROM archive_messages WHERE id = ?", (row["id"],))
                removed += 1
        return removed

    # ==================== errors ====================

    def find_open_error(self, umo: str, original: str) -> dict[str, Any] | None:
        row = self._run(
            "SELECT * FROM errors WHERE status = 'open'"
            " AND (umo = ? OR umo = '')"
            " AND LOWER(original) = LOWER(?) LIMIT 1",
            (umo, original),
            fetch_one=True,
        )
        return dict(row) if row else None

    def add_error(
        self,
        umo: str,
        date: str,
        original: str,
        corrected: str = "",
        explanation: str = "",
        category: str = "",
    ) -> int:
        self._run(
            "INSERT INTO errors (umo, first_date, last_date, original, corrected,"
            " explanation, category, next_review, interval_days)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
            (umo, date, date, original, corrected, explanation, category, date),
        )
        row = self._run("SELECT last_insert_rowid() AS id", fetch_one=True)
        return int(row["id"]) if row else 0

    def touch_error(self, error_id: int, date: str) -> None:
        self._run(
            "UPDATE errors SET last_date = ?, hit_count = hit_count + 1 WHERE id = ?",
            (date, error_id),
        )

    def update_error(self, error_id: int, fields: dict[str, Any]) -> None:
        allowed = ["original", "corrected", "explanation", "category", "status"]
        sets = [f"{k} = ?" for k in fields if k in allowed]
        if not sets:
            return
        params = [fields[k] for k in fields if k in allowed]
        self._run(
            f"UPDATE errors SET {', '.join(sets)} WHERE id = ?",
            (*params, error_id),
        )

    def delete_error(self, error_id: int) -> None:
        self._run("DELETE FROM errors WHERE id = ?", (error_id,))

    def list_errors(
        self,
        umo: str | None = None,
        status: str | None = None,
        keyword: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM errors WHERE 1=1"
        params: list[Any] = []
        if umo:
            sql += " AND (umo = ? OR umo = '')"
            params.append(umo)
        if status and status != "all":
            sql += " AND status = ?"
            params.append(status)
        if keyword:
            sql += " AND (original LIKE ? OR corrected LIKE ? OR explanation LIKE ?)"
            params.extend([f"%{keyword}%"] * 3)
        sql += " ORDER BY last_date DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return self._rows_to_dicts(self._run(sql, tuple(params), fetch_all=True))

    def count_errors(
        self,
        umo: str | None = None,
        status: str | None = None,
        keyword: str = "",
    ) -> int:
        sql = "SELECT COUNT(*) AS c FROM errors WHERE 1=1"
        params: list[Any] = []
        if umo:
            sql += " AND (umo = ? OR umo = '')"
            params.append(umo)
        if status and status != "all":
            sql += " AND status = ?"
            params.append(status)
        if keyword:
            sql += " AND (original LIKE ? OR corrected LIKE ? OR explanation LIKE ?)"
            params.extend([f"%{keyword}%"] * 3)
        row = self._run(sql, tuple(params), fetch_one=True)
        return int(row["c"]) if row else 0

    def top_open_errors(
        self, umo: str | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM errors WHERE status = 'open'"
        params: list[Any] = []
        if umo:
            sql += " AND (umo = ? OR umo = '')"
            params.append(umo)
        sql += " ORDER BY hit_count DESC, last_date DESC LIMIT ?"
        params.append(limit)
        return self._rows_to_dicts(self._run(sql, tuple(params), fetch_all=True))

    def due_errors(
        self, umo: str | None = None, today: str = ""
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM errors WHERE status = 'open' AND next_review != ''"
        params: list[Any] = []
        if umo:
            sql += " AND (umo = ? OR umo = '')"
            params.append(umo)
        sql += " AND next_review <= ? ORDER BY next_review ASC LIMIT 20"
        params.append(today)
        return self._rows_to_dicts(self._run(sql, tuple(params), fetch_all=True))

    def mark_error_review(self, error_id: int, remembered: bool, today: str) -> None:
        row = self._run(
            "SELECT interval_days FROM errors WHERE id = ?", (error_id,), fetch_one=True
        )
        if not row:
            return
        interval = advance_interval(int(row["interval_days"]), remembered)
        next_review = (datetime.now() + timedelta(days=interval)).strftime("%Y-%m-%d")
        self._run(
            "UPDATE errors SET interval_days = ?, next_review = ? WHERE id = ?",
            (interval, next_review, error_id),
        )

    # ==================== sentences ====================

    def add_sentence(
        self,
        umo: str,
        date: str,
        sentence: str,
        note: str = "",
        source: str = "user",
        tags: str = "",
        context: str = "",
        dialog: str = "",
    ) -> None:
        exists = self._run(
            "SELECT id FROM sentences WHERE (umo = ? OR umo = '')"
            " AND LOWER(sentence) = LOWER(?) LIMIT 1",
            (umo, sentence),
            fetch_one=True,
        )
        if exists:
            return
        self._run(
            "INSERT INTO sentences (umo, date, sentence, note, source, tags, context, dialog)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (umo, date, sentence, note, source, tags, context, dialog),
        )

    def list_sentences(
        self,
        umo: str | None = None,
        date: str | None = None,
        keyword: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM sentences WHERE 1=1"
        params: list[Any] = []
        if umo:
            sql += " AND (umo = ? OR umo = '')"
            params.append(umo)
        if date:
            sql += " AND date = ?"
            params.append(date)
        if keyword:
            sql += " AND (sentence LIKE ? OR note LIKE ?)"
            params.extend([f"%{keyword}%"] * 2)
        sql += " ORDER BY date DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return self._rows_to_dicts(self._run(sql, tuple(params), fetch_all=True))

    def count_sentences(
        self,
        umo: str | None = None,
        date: str | None = None,
        keyword: str = "",
    ) -> int:
        sql = "SELECT COUNT(*) AS c FROM sentences WHERE 1=1"
        params: list[Any] = []
        if umo:
            sql += " AND (umo = ? OR umo = '')"
            params.append(umo)
        if date:
            sql += " AND date = ?"
            params.append(date)
        if keyword:
            sql += " AND (sentence LIKE ? OR note LIKE ?)"
            params.extend([f"%{keyword}%"] * 2)
        row = self._run(sql, tuple(params), fetch_one=True)
        return int(row["c"]) if row else 0

    def recent_sentences(
        self, umo: str | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM sentences"
        params: list[Any] = []
        if umo:
            sql += " WHERE (umo = ? OR umo = '')"
            params.append(umo)
        sql += " ORDER BY date DESC, id DESC LIMIT ?"
        params.append(limit)
        return self._rows_to_dicts(self._run(sql, tuple(params), fetch_all=True))

    def update_sentence(self, sentence_id: int, fields: dict[str, Any]) -> None:
        allowed = ["sentence", "note", "tags", "context", "dialog"]
        sets = [f"{k} = ?" for k in fields if k in allowed]
        if not sets:
            return
        params = [fields[k] for k in fields if k in allowed]
        self._run(
            f"UPDATE sentences SET {', '.join(sets)} WHERE id = ?",
            (*params, sentence_id),
        )

    def delete_sentence(self, sentence_id: int) -> None:
        self._run("DELETE FROM sentences WHERE id = ?", (sentence_id,))

    # ==================== vocab ====================

    def add_vocab(
        self,
        umo: str,
        date: str,
        word: str,
        meaning: str = "",
        example: str = "",
        source: str = "ai",
        context: str = "",
        dialog: str = "",
    ) -> None:
        self._run(
            "INSERT INTO vocab (umo, date, word, meaning, example, source, next_review, interval_days, context, dialog)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)"
            " ON CONFLICT(umo, word) DO UPDATE SET"
            " meaning = CASE WHEN excluded.meaning != '' THEN excluded.meaning ELSE vocab.meaning END,"
            " example = CASE WHEN excluded.example != '' THEN excluded.example ELSE vocab.example END,"
            " context = CASE WHEN excluded.context != '' THEN excluded.context ELSE vocab.context END,"
            " dialog = CASE WHEN excluded.dialog != '' THEN excluded.dialog ELSE vocab.dialog END",
            (umo, date, word, meaning, example, source, date, context, dialog),
        )

    def list_vocab(
        self,
        umo: str | None = None,
        keyword: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM vocab WHERE 1=1"
        params: list[Any] = []
        if umo:
            sql += " AND (umo = ? OR umo = '')"
            params.append(umo)
        if keyword:
            sql += " AND (word LIKE ? OR meaning LIKE ?)"
            params.extend([f"%{keyword}%"] * 2)
        sql += " ORDER BY date DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return self._rows_to_dicts(self._run(sql, tuple(params), fetch_all=True))

    def count_vocab(self, umo: str | None = None, keyword: str = "") -> int:
        sql = "SELECT COUNT(*) AS c FROM vocab WHERE 1=1"
        params: list[Any] = []
        if umo:
            sql += " AND (umo = ? OR umo = '')"
            params.append(umo)
        if keyword:
            sql += " AND (word LIKE ? OR meaning LIKE ?)"
            params.extend([f"%{keyword}%"] * 2)
        row = self._run(sql, tuple(params), fetch_one=True)
        return int(row["c"]) if row else 0

    def find_vocab(self, umo: str, word: str) -> dict[str, Any] | None:
        row = self._run(
            "SELECT * FROM vocab WHERE (umo = ? OR umo = '')"
            " AND LOWER(word) = LOWER(?) LIMIT 1",
            (umo, word),
            fetch_one=True,
        )
        return dict(row) if row else None

    def due_vocab(
        self, umo: str | None = None, today: str = ""
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM vocab WHERE next_review != ''"
        params: list[Any] = []
        if umo:
            sql += " AND (umo = ? OR umo = '')"
            params.append(umo)
        sql += " AND next_review <= ? ORDER BY next_review ASC LIMIT 20"
        params.append(today)
        return self._rows_to_dicts(self._run(sql, tuple(params), fetch_all=True))

    def mark_vocab_review(self, vocab_id: int, remembered: bool, today: str) -> None:
        row = self._run(
            "SELECT interval_days FROM vocab WHERE id = ?", (vocab_id,), fetch_one=True
        )
        if not row:
            return
        interval = advance_interval(int(row["interval_days"]), remembered)
        next_review = (datetime.now() + timedelta(days=interval)).strftime("%Y-%m-%d")
        self._run(
            "UPDATE vocab SET interval_days = ?, next_review = ? WHERE id = ?",
            (interval, next_review, vocab_id),
        )

    def update_vocab(self, vocab_id: int, fields: dict[str, Any]) -> None:
        allowed = ["word", "meaning", "example", "context", "dialog"]
        sets = [f"{k} = ?" for k in fields if k in allowed]
        if not sets:
            return
        params = [fields[k] for k in fields if k in allowed]
        self._run(
            f"UPDATE vocab SET {', '.join(sets)} WHERE id = ?",
            (*params, vocab_id),
        )

    def delete_vocab(self, vocab_id: int) -> None:
        self._run("DELETE FROM vocab WHERE id = ?", (vocab_id,))

    # ==================== daily practice ====================

    def save_daily(
        self, date: str, umo: str, ptype: str, items: list[dict[str, Any]]
    ) -> None:
        self._run("DELETE FROM daily_practice WHERE date = ?", (date,))
        self._run(
            "INSERT INTO daily_practice (date, umo, type, items_json, status, created_at)"
            " VALUES (?, ?, ?, ?, 'generated', ?)",
            (
                date,
                umo,
                ptype,
                json.dumps(items, ensure_ascii=False),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )

    def get_daily(self, date: str) -> dict[str, Any] | None:
        row = self._run(
            "SELECT * FROM daily_practice WHERE date = ? LIMIT 1",
            (date,),
            fetch_one=True,
        )
        if not row:
            return None
        data = dict(row)
        try:
            data["items"] = json.loads(data.pop("items_json"))
        except (TypeError, ValueError):
            data["items"] = []
        return data

    def set_daily_status(self, daily_id: int, status: str) -> None:
        self._run(
            "UPDATE daily_practice SET status = ? WHERE id = ?", (status, daily_id)
        )

    def delete_daily(self, daily_id: int) -> None:
        self._run("DELETE FROM daily_practice WHERE id = ?", (daily_id,))

    def list_daily(self, limit: int = 30, offset: int = 0) -> list[dict[str, Any]]:
        rows = self._run(
            "SELECT * FROM daily_practice ORDER BY date DESC LIMIT ? OFFSET ?",
            (limit, offset),
            fetch_all=True,
        )
        result = self._rows_to_dicts(rows)
        for item in result:
            try:
                item["items"] = json.loads(item.pop("items_json"))
            except (TypeError, ValueError):
                item["items"] = []
        return result

    def count_daily(self) -> int:
        row = self._run("SELECT COUNT(*) AS c FROM daily_practice", fetch_one=True)
        return int(row["c"]) if row else 0

    # ==================== overview ====================

    def overview(self) -> dict[str, Any]:
        def _count(table: str) -> int:
            row = self._run(f"SELECT COUNT(*) AS c FROM {table}", fetch_one=True)
            return int(row["c"]) if row else 0

        return {
            "sentences": _count("sentences"),
            "errors": _count("errors"),
            "open_errors": self.count_errors(status="open"),
            "vocab": _count("vocab"),
            "archive": _count("archive_messages"),
            "daily": _count("daily_practice"),
        }

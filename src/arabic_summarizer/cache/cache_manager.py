"""
مدیریت کش با SQLite - ساده، بدون وابستگی بیرونی، مناسب برای اجرای آفلاین.
"""

import sqlite3
from pathlib import Path
from typing import Optional


class CacheManager:
    def __init__(self, db_path: str = "cache_store/summary_cache.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_table()

    def _init_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                summary TEXT
            )
            """
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[str]:
        cur = self.conn.execute("SELECT summary FROM cache WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set(self, key: str, summary: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, summary) VALUES (?, ?)",
            (key, summary),
        )
        self.conn.commit()

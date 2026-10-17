from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import tempfile
import threading
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def now():
    return datetime.now(UTC).isoformat()


def stable_hash(value):
    return hashlib.sha256(value if isinstance(value, bytes) else str(value).encode()).hexdigest()


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str))


def write_text(path, content):
    """Publish a whole text file atomically, with a unique sibling for each concurrent writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 BOM for Chinese/Spanish text in Excel; authoritative data remain in SQLite.
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, dtype=str, encoding="utf-8-sig")


def load_sample(project):
    path = project.root / "sample.csv"
    design = json.loads((project.root / "sample_design.json").read_text())
    if design["sample_hash"] != stable_hash(path.read_bytes()):
        raise ValueError("冻结的视频 sample.csv 已被修改；请恢复原文件或另建研究快照")
    return read_csv(path)


class Project:
    def __init__(self, config="config.toml"):
        self.config_path = Path(config).resolve()
        self.base = self.config_path.parent
        self.cfg = tomllib.loads(self.config_path.read_text(encoding="utf-8"))
        load_dotenv(self.base / ".env", override=False)
        s = self.cfg["study"]
        if not s["run_id"].replace("_", "").replace("-", "").isalnum():
            raise ValueError("run_id 只能包含字母、数字、下划线、连字符")
        if s["comment_window"] not in {"all", "year", "first30"}:
            raise ValueError("comment_window 必须是 all/year/first30")
        self.root = self.base / s["data_dir"] / s["run_id"]
        self.reports = self.base / s["reports_dir"] / s["run_id"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        self.ledger = self.base / s["data_dir"] / "quota.sqlite"
        self.store = Store(self.root / "research.sqlite")
        identity = {k: s[k] for k in ["year", "timezone", "channel_id", "run_id"]}
        prior = self.store.meta("identity")
        if prior and prior != identity:
            raise ValueError("快照的频道/年份/时区不可修改；请使用新的 run_id")
        self.store.set_meta("identity", identity)
        self.store.set_meta("created_at", self.store.meta("created_at") or now())
        self.salt_path = self.base / s["data_dir"] / ".author_salt"
        if not self.salt_path.exists():
            try:
                fd = os.open(self.salt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as f:
                    f.write(secrets.token_hex(32))
            except FileExistsError:
                pass
        self.salt = self.salt_path.read_text().strip().encode()

    def author_hash(self, author):
        return hmac.new(self.salt, author.encode(), hashlib.sha256).hexdigest() if author else None

    @contextlib.contextmanager
    def lock(self):
        with (self.root / ".workflow.lock").open("w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("同一快照已有命令运行；请等待完成") from None
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS raw_pages
          (cache_key TEXT PRIMARY KEY, endpoint TEXT, params TEXT, fetched_at TEXT, payload BLOB);
        CREATE TABLE IF NOT EXISTS raw_page_history
          (history_id INTEGER PRIMARY KEY AUTOINCREMENT, cache_key TEXT, endpoint TEXT, params TEXT,
           fetched_at TEXT, payload BLOB, archived_at TEXT, reason TEXT);
        CREATE TABLE IF NOT EXISTS inventory
          (video_id TEXT PRIMARY KEY, published_hint TEXT, title TEXT);
        CREATE TABLE IF NOT EXISTS videos
          (video_id TEXT PRIMARY KEY, fetched_at TEXT, available INTEGER, payload TEXT);
        CREATE TABLE IF NOT EXISTS comments
          (comment_id TEXT PRIMARY KEY, video_id TEXT, root_id TEXT, is_reply INTEGER,
           fetched_at TEXT, payload TEXT);
        CREATE INDEX IF NOT EXISTS comments_video ON comments(video_id);
        CREATE TABLE IF NOT EXISTS collection
          (video_id TEXT PRIMARY KEY, status TEXT, reason TEXT, updated_at TEXT);
        """)
        self.db.commit()

    def execute(self, sql, args=()):
        with self.lock, self.db:
            self.db.execute(sql, args)

    def many(self, sql, rows):
        with self.lock, self.db:
            self.db.executemany(sql, rows)

    def rows(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def meta(self, key):
        r = self.rows("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(r[0]["value"]) if r else None

    def set_meta(self, key, value):
        self.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))

    def frame(self, table):
        if not table.replace("_", "").isalnum():
            raise ValueError("Invalid table")
        with self.lock:
            return pd.read_sql_query(f'SELECT * FROM "{table}"', self.db)

    def save_frame(self, table, frame):
        with self.lock:
            frame.to_sql(table, self.db, if_exists="replace", index=False)

    def archive_pages(self, scopes, reason):
        """Invalidate only expired pagination chains, retaining every prior raw response.

        scopes: (endpoint, parameter name, set of parameter values) tuples.
        """
        stamp = now()
        moved = 0
        with self.lock, self.db:
            for endpoint, field, values in scopes:
                pages = self.db.execute("SELECT * FROM raw_pages WHERE endpoint=?", (endpoint,)).fetchall()
                for row in pages:
                    if json.loads(row["params"]).get(field) not in values:
                        continue
                    self.db.execute("""INSERT INTO raw_page_history
                        (cache_key,endpoint,params,fetched_at,payload,archived_at,reason)
                        VALUES (?,?,?,?,?,?,?)""", (row["cache_key"], row["endpoint"], row["params"],
                            row["fetched_at"], row["payload"], stamp, reason))
                    self.db.execute("DELETE FROM raw_pages WHERE cache_key=?", (row["cache_key"],))
                    moved += 1
        return moved

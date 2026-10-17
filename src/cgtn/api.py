from __future__ import annotations

import gzip
import json
import math
import os
import sqlite3
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from .core import now, stable_hash


class StopCollection(Exception):
    """Resumable time/quota stop; not a complete dataset."""


class APIError(Exception):
    def __init__(self, reason, status=0):
        self.reason, self.status = reason, status
        super().__init__(f"YouTube API: {reason} (HTTP {status})")


def scrub_authors(obj, project):
    """Keep text verbatim, but pseudonymize API account identifiers before raw storage."""
    if isinstance(obj, list):
        return [scrub_authors(x, project) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        if k in {"authorDisplayName", "authorProfileImageUrl", "authorChannelUrl"}:
            continue
        if k == "authorChannelId":
            out["author_hash"] = project.author_hash(v.get("value", ""))
        else:
            out[k] = scrub_authors(v, project)
    return out


class API:
    def __init__(self, project, hours=0):
        self.p = project
        self.cfg = project.cfg["api"]
        self.key = os.environ.get("YOUTUBE_API_KEY", "").strip()
        if not self.key:
            raise ValueError("请先在 .env 中填写 YOUTUBE_API_KEY；见 README.md")
        if not math.isfinite(hours) or hours < 0:
            raise ValueError("hours 必须为有限非负数；0 表示不限运行时长")
        if not math.isfinite(self.cfg["requests_per_second"]) or self.cfg["requests_per_second"] <= 0:
            raise ValueError("requests_per_second 必须为有限正数")
        self.deadline = time.monotonic() + hours * 3600 if hours else None
        self.stop = threading.Event()
        self.rate_lock = threading.Lock()
        self.last_request = 0.0
        self.client = httpx.Client(timeout=self.cfg["timeout_seconds"], trust_env=True)
        with sqlite3.connect(self.p.ledger) as db:
            db.execute("CREATE TABLE IF NOT EXISTS usage (day TEXT PRIMARY KEY, units INTEGER)")

    def check(self):
        if self.stop.is_set() or (self.deadline is not None and time.monotonic() >= self.deadline):
            self.stop.set()
            raise StopCollection("已到采集时限/停止点，进度已保存；相同命令可续传")

    def pause(self, seconds):
        """Backoff/rate limiting remains interruptible even without a wall-clock limit."""
        if self.deadline is not None:
            seconds = min(seconds, max(0, self.deadline - time.monotonic()))
        self.stop.wait(seconds)
        self.check()

    def reserve(self):
        self.check()
        day = datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
        with sqlite3.connect(self.p.ledger, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            used = db.execute("SELECT units FROM usage WHERE day=?", (day,)).fetchone()
            if used and used[0] >= self.cfg["daily_budget"]:
                self.stop.set()
                raise StopCollection("已到本地每日 API 配额预算；配额重置后重跑同一命令续传")
            db.execute("INSERT INTO usage VALUES (?,1) ON CONFLICT(day) DO UPDATE SET units=units+1", (day,))

    def get(self, endpoint, **params):
        # Only these list endpoints are used; each is currently 1 quota unit.
        if endpoint not in {"channels", "playlistItems", "videos", "commentThreads", "comments"}:
            raise ValueError("Unsupported endpoint")
        params = {k: v for k, v in params.items() if v is not None}
        canonical = json.dumps(params, sort_keys=True)
        key = stable_hash(endpoint + canonical)
        cached = self.p.store.rows("SELECT * FROM raw_pages WHERE cache_key=?", (key,))
        if cached:
            r = cached[0]
            return json.loads(gzip.decompress(r["payload"])), r["fetched_at"]
        for attempt in range(self.cfg["retries"] + 1):
            self.check()
            with self.rate_lock:
                delay = 1 / self.cfg["requests_per_second"] - (time.monotonic() - self.last_request)
                if delay > 0:
                    self.pause(delay)
                self.reserve()
                self.last_request = time.monotonic()
            try:
                # The key never enters request URL, cache, console output or saved exceptions.
                r = self.client.get(f"https://www.googleapis.com/youtube/v3/{endpoint}",
                                    params=params, headers={"X-Goog-Api-Key": self.key})
                body = r.json()
            except (httpx.HTTPError, ValueError):
                reason, status = "network_or_invalid_json", 0
            else:
                if r.status_code == 200:
                    stamp = now()
                    body = scrub_authors(body, self.p)
                    self.p.store.execute("INSERT OR REPLACE INTO raw_pages VALUES (?,?,?,?,?)",
                                         (key, endpoint, canonical, stamp,
                                          gzip.compress(json.dumps(body, ensure_ascii=False).encode())))
                    return body, stamp
                err = body.get("error", {})
                reason = (err.get("errors") or [{}])[0].get("reason", "unknown_api_error")
                status = r.status_code
                if reason in {"quotaExceeded", "dailyLimitExceeded"}:
                    self.stop.set()
                    raise StopCollection("Google 项目配额已耗尽；请在 Cloud Console 核对")
                # Comment processingFailure may be transient despite its HTTP 400 status.
                # Retry only this documented comment error, with the same bounded budget.
                comment_processing_failure = (
                    endpoint in {"commentThreads", "comments"}
                    and status == 400 and reason == "processingFailure"
                )
                if not comment_processing_failure and status not in {429, 500, 502, 503, 504} and reason not in {
                    "rateLimitExceeded", "userRateLimitExceeded", "backendError"
                }:
                    raise APIError(reason, status)
            if attempt == self.cfg["retries"]:
                raise APIError(reason, status)
            self.pause(min(2 ** attempt, 30))
        raise AssertionError("unreachable")

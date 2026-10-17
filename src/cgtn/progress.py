"""Plain terminal messages triggered by work events, never by a timer."""
from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

_OUTPUT_LOCK = threading.Lock()
ROLES = {"coder": "初编 1/3", "reviewer": "盲复核 2/3", "adjudicator": "裁决 3/3", "checker": "检查 2/2"}


def label(role):
    parts = role.split("/")
    return "/".join(parts[:-1] + [ROLES.get(parts[-1], parts[-1])])


def event(stage, message):
    with _OUTPUT_LOCK:
        print(f"[{datetime.now(UTC).astimezone():%H:%M:%S}] [{stage}] {message}", flush=True)


def duration(seconds):
    minutes, seconds = divmod(int(max(0, seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class BatchProgress:
    def __init__(self, role, total, cached, every=10, maximum=0, cached_unavailable=0):
        self.stage, self.total, self.cached = label(role), total, cached
        self.every, self.maximum = every, maximum
        self.success, self.failed, self.handled, self.recoverable = 0, 0, 0, 0
        self.unavailable, self.cached_unavailable = 0, cached_unavailable
        self.started = time.monotonic()

    def start(self, requests, workers):
        event(self.stage, f"开始：共 {self.total:,} 批；复用 {self.cached:,} 批；"
              f"待处理 {self.total - self.cached - self.cached_unavailable:,} 批；不可编码复用 {self.cached_unavailable:,} 批；并发上限 {workers}。")
        self.show("进度", self.cached, requests, 0)

    def update(self, success, requests, pending, recent, recoverable=False, unavailable=False):
        self.handled += 1
        self.success += int(success)
        self.failed += int(not success and not recoverable and not unavailable)
        self.unavailable += int(unavailable)
        self.recoverable += int(not success and recoverable)
        if self.handled == 1 or self.handled % self.every == 0 or not success:
            self.show("进度", self.cached + self.success, requests, pending, recent)

    def show(self, state, done, requests, pending, recent="", saved_errors=None, saved_recoverable=None, saved_unavailable=None):
        fraction = f"{100 * done / self.total:.1f}%" if self.total else "无任务"
        failures = f"本次失败 {self.failed:,}" if saved_errors is None else f"当前失败记录 {saved_errors:,}"
        recoveries = self.recoverable if saved_recoverable is None else saved_recoverable
        missing = self.cached_unavailable + self.unavailable if saved_unavailable is None else saved_unavailable
        event(self.stage, f"{state}：成功 {done:,}/{self.total:,} ({fraction})；未完成 {self.total - done - missing:,}；不可编码 {missing:,}；"
              f"本次成功 {done - self.cached:,}；{failures}；待恢复 {recoveries:,}；在途 {pending}；"
              f"本次总请求 {requests:,}/{self.maximum or '不限'}；阶段用时 {duration(time.monotonic() - self.started)}"
              + (f"；最近 {recent}" if recent else ""))

    def finish(self, saved_complete, requests, saved_errors=None, saved_recoverable=None, saved_unavailable=0):
        self.show("阶段结束" if saved_complete + saved_unavailable == self.total else "已暂停，可续传", saved_complete, requests, 0,
                  saved_errors=saved_errors, saved_recoverable=saved_recoverable, saved_unavailable=saved_unavailable)

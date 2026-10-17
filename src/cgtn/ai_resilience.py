"""Bounded automatic recovery between drained request queues; never changes model measurements."""
from __future__ import annotations

import math
import threading
import time

from .api import StopCollection
from .progress import event

SERVICE_PAUSES = {"network_retries_exhausted", "transient_retries_exhausted", "retry_after_exceeds_wait_limit"}


def wait_for_recovery(seconds):
    # Called only on the controlling thread after all request workers have drained.
    # Ctrl+C interrupts this wait; there are no periodic progress messages.
    threading.Event().wait(seconds)


class AutomaticRecovery:
    def __init__(self, cfg):
        self.rounds = cfg.get("auto_recovery_rounds", 6)
        self.delay = cfg.get("auto_recovery_delay_seconds", 30)
        self.wait_budget = cfg.get("auto_recovery_wait_seconds", 1800)
        self.deferred_rounds = cfg.get("deferred_retry_rounds", 1)
        if (type(self.rounds) is not int or not 0 <= self.rounds <= 20 or
                type(self.deferred_rounds) is not int or not 0 <= self.deferred_rounds <= 3 or
                any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                    for x in [self.delay, self.wait_budget]) or
                not self.delay <= self.wait_budget <= 86400):
            raise ValueError("自动恢复轮数须为 0..20，延后重试轮数须为 0..3；等待参数须满足 0 < delay <= wait <= 86400")
        self.used, self.waited = 0, 0.0

    def settings(self):
        return {"auto_recovery_rounds": self.rounds, "auto_recovery_delay_seconds": self.delay,
                "auto_recovery_wait_seconds": self.wait_budget, "deferred_retry_rounds": self.deferred_rounds}

    @staticmethod
    def check_budget(runner):
        if runner.maximum and runner.count >= runner.maximum:
            runner.stop_reason = "request_budget_exhausted"
            runner.stop.set()
            raise StopCollection("本次 AI 总请求预算已到，已保存进度；自动恢复不会重置请求预算")
        if (getattr(runner, "output_limit", 0) and
                runner.output_committed + runner.output_reserved + runner.config["max_completion_tokens"] > runner.output_limit):
            runner.stop_reason = "output_token_budget_exhausted"
            runner.stop.set()
            raise StopCollection("本次输出 token 派发预算不足，已保存进度；自动恢复不会重置 token 预算")

    def resume(self, runner):
        """Only resume known transient service failures, never config/integrity/billing/user stops."""
        reason = runner.stop_reason
        if reason not in SERVICE_PAUSES:
            return False
        self.check_budget(runner)
        delay = max(min(self.delay * 2 ** self.used, 300), runner.resume_not_before - time.monotonic(), 0)
        if self.used >= self.rounds or delay > self.wait_budget - self.waited:
            runner.execution_event("recovery_limit_reached", reason, self.used, 0)
            event("自动恢复", "已到自动恢复轮数或等待上限；进度已保存，保留未完成状态供下次续传。")
            return False
        self.used += 1
        self.waited += delay
        runner.execution_event("recovery_scheduled", reason, self.used, delay)
        event("自动恢复", f"{reason}；第 {self.used}/{self.rounds} 轮，等待 {delay:.1f} 秒后续传；"
              f"本次累计请求仍为 {runner.count}/{runner.maximum or '不限'}。Ctrl+C 可停止。")
        try:
            wait_for_recovery(delay)
        except KeyboardInterrupt:
            runner.stop_reason = "interrupted"
            runner.execution_event("recovery_interrupted", "interrupted", self.used, delay)
            raise
        self.check_budget(runner)
        runner.renew_client()
        runner.stop_reason = ""
        runner.stop.clear()
        runner.execution_event("recovery_resumed", reason, self.used, delay)
        event("自动恢复", "等待结束，连接池已重建；复用成功结果，只继续未完成任务。")
        return True

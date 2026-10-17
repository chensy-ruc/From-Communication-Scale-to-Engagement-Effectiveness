"""Explain known provider-account blocks without guessing balances or retrying billing failures."""
from __future__ import annotations

import json
import re
from urllib.parse import urlsplit

from .api import StopCollection

BIGMODEL_ERRORS_URL = "https://docs.bigmodel.cn/cn/api/api-code"
DEEPSEEK_ERRORS_URL = "https://api-docs.deepseek.com/quick_start/error_codes/"


def deepseek_content_refusal(base_url, status, error):
    """Match the observed provider refusal envelope, never arbitrary HTTP 400 prose."""
    return (urlsplit(base_url).hostname == "api.deepseek.com" and status == 400
            and isinstance(error, dict) and error.get("code") == "invalid_request_error"
            and isinstance(error.get("message"), str)
            and re.fullmatch(r"Content Exists Risk(?:\s*\(request_id:\s*[A-Za-z0-9-]{1,128}\))?",
                             error["message"].strip()) is not None)


def provider_action(error):
    if error == "deepseek_insufficient_balance":
        category, reason, action = ("account_balance", "DeepSeek 服务报告账户余额不足（HTTP 402）",
                                    "请在 DeepSeek 开放平台核对账户余额，恢复可用后重跑原命令续传")
    elif error == "bigmodel_1113_insufficient_quota":
        category, reason, action = ("account_balance", "智谱服务报告账户已欠费（业务码 1113）",
                                    "请在智谱开放平台核对账户并充值，确认可用余额恢复后重跑原命令续传")
    elif error in {"bigmodel_1308_period_quota_exhausted", "bigmodel_1310_period_quota_exhausted"}:
        category, reason, action = ("period_quota", "智谱服务报告周期使用额度已用尽",
                                    "请在智谱平台核对额度重置时间，额度恢复后重跑原命令；本机不猜测重置时间")
    elif error in {f"bigmodel_{code}_period_quota_and_balance_exhausted" for code in [1316, 1317]}:
        category, reason, action = ("period_quota_and_balance", "智谱服务报告周期额度用尽且账户余额不足",
                                    "请在智谱平台核对周期额度与余额，服务恢复可用后重跑原命令")
    elif error in {f"bigmodel_{code}_{kind}_spend_limit" for code, kind in
                   [(1318, "account"), (1319, "account"), (1320, "enterprise"), (1321, "enterprise")]}:
        category, reason, action = ("spend_limit", "智谱服务报告周期额度用尽且已达账户或企业消费上限",
                                    "请由账户管理员核对消费上限与周期额度，服务恢复可用后重跑原命令")
    elif error in {"insufficient_quota", "billing_hard_limit_reached"}:
        category, reason, action = ("provider_quota_or_billing", "模型服务报告额度不足或达到计费上限",
                                    "请在对应模型平台核对余额、额度与消费上限，恢复可用后重跑原命令")
    else:
        return None
    return {"error_code": error, "category": category, "reason": reason, "action": action,
            "automatic_retry": False, "live_account_checked": False,
            "documentation_url": (BIGMODEL_ERRORS_URL if error.startswith("bigmodel_") else
                                  DEEPSEEK_ERRORS_URL if error.startswith("deepseek_") else None)}


class ProviderActionRequired(StopCollection):
    """A resumable external-account block, distinct from a code/configuration fault."""
    def __init__(self, error):
        self.action = provider_action(error)
        if self.action is None:
            raise ValueError("未知账户暂停错误码")
        super().__init__(f"[{error}] {self.action['reason']}。{self.action['action']}。"
                         "已保存成功结果和失败位置；此类账户阻断不自动重试。")


def latest_provider_block(executions):
    if not executions:
        return None
    latest = max(executions, key=lambda r: (r["started_at"], r["execution_id"]))
    if latest["status"] not in {"error", "paused", "awaiting_provider_action"}:
        return None
    action = provider_action(latest["stop_reason"])
    return ({**action, "execution_id": latest["execution_id"], "recorded_at": latest["ended_at"],
             "execution_status_as_recorded": latest["status"]} if action else None)


def latest_stop(executions, attempts):
    """Describe the most recent execution, never an obsolete billing/phase stop."""
    if not executions:
        return None
    latest = max(executions, key=lambda r: (r["started_at"], r["execution_id"]))
    if latest["status"] not in {"paused", "error", "interrupted", "awaiting_provider_action"}:
        return None
    relevant = [(row, json.loads(row["usage"])) for row in sorted(attempts, key=lambda r: r["id"], reverse=True)]
    relevant = [(r, u) for r, u in relevant if u.get("_client_execution_id") == latest["execution_id"]]
    error = next((r["status"] for r, _ in relevant if r["status"] not in {"complete", "started"}), None)
    reason = latest["stop_reason"]
    messages = {
        "consecutive_annotation_failures": "连续编码校验失败触发保护；有效结果保留，未进入下一阶段",
        "unfinished_annotation_tasks": "当前阶段还有未完成/失败任务，尚不能进入下一阶段",
        "evidence_quarantine_limit_reached": "少量单条引文失败的隔离上限已用完；请检查系统性编码问题，失败结果和断点已保留",
        "request_budget_exhausted": "命令的累计请求上限已到，重跑续传或使用 --max-requests 0",
        "output_token_budget_exhausted": "命令的累计输出预算已到；这不是单次输出上限",
        "network_retries_exhausted": "网络重试及自动恢复未能完成，请检查网络后续传",
        "transient_retries_exhausted": "服务临时错误重试及自动恢复未能完成，请稍后续传",
        "retry_after_exceeds_wait_limit": "服务要求的等待超过本机恢复上限，请稍后续传",
        "interrupted": "运行被中断，已落盘结果可以续传",
    }
    action = provider_action(reason)
    description = action["reason"] + "；" + action["action"] if action else messages.get(reason, "详情见请求与执行记录")
    return {"execution_id": latest["execution_id"], "stop_reason": reason,
            "phase": relevant[0][1].get("_client_phase") if relevant else None,
            "last_request_error": error, "description": description, "recorded_at": latest["ended_at"]}

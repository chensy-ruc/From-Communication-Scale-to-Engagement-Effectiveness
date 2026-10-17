"""Descriptive audit figures with explicit denominators, missingness and source hashes."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

from . import ai_figures as chart
from .coding import COMMENT_LABELS
from .core import now, stable_hash, write_json

ROLE = {"coder": ("初编", "Coder"), "reviewer": ("盲复核", "Reviewer"),
        "adjudicator": ("裁决", "Adjudicator"), "final": ("最终", "Final"), "checker": ("检查子集", "Check subset")}
STATUS = {
    "complete": ("校验通过", "Validated", chart.TEAL),
    "unavailable": ("不可编码", "Unavailable", "#b6a2c8"),
    "failed": ("当前失败", "Failed", chart.RED),
    "recovery_pending": ("待恢复", "Recovery pending", chart.ORANGE),
    "started_unconfirmed": ("在途／未确认", "Unconfirmed", "#9276ac"),
    "not_started": ("未开始", "Not started", "#dce3e7"),
    "not_ready": ("等待全流程", "Awaiting workflow", chart.GRAY),
    "conflicting_results": ("缓存冲突", "Conflicting cache", "#6f2e2e"),
}


def role_label(role):
    if role.startswith("historical_"):
        return chart.text("历史：", "Historical: ") + role_label(role.removeprefix("historical_"))
    return chart.text(*ROLE.get(role, (role, role)))


def progress_chart(frame, dest, note, unit, screening=False):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, kind in zip(axes, ["videos", "comments"], strict=True):
        part = frame.loc[frame.kind.eq(kind)].copy()
        if part.empty:
            chart.empty(ax)
            continue
        y, left = np.arange(len(part)), np.zeros(len(part))
        for state, (zh, en, color) in STATUS.items():
            values = part[state].to_numpy() / part.total.to_numpy()
            if values.sum():
                ax.barh(y, values, left=left, height=.57, color=color, label=chart.text(zh, en))
            left += values
        names = [(role_label(row.role) + (chart.text("（当前计划）", " (planned so far)") if row.role == "adjudicator" else "")
                  + f"  {row.complete:,}/{row.total:,}") for row in part.itertuples()]
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.01)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(chart.text("视频" if kind == "videos" else "评论", kind.title()))
        ax.set_xlabel(chart.text("各状态占该角色计划总数；左侧为通过数／总数", "Share of planned units; labels = validated / planned"))
    handles, labels = [], []
    for ax in axes:
        hh, ll = ax.get_legend_handles_labels()
        for h, lab in zip(hh, ll, strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .04), ncol=min(4, len(labels)), frameon=False, fontsize=8)
    title = chart.text("去重分析单元的处理状态" if unit else "原始任务批次的处理状态",
                       "Deduplicated coding units" if unit else "Original task batches")
    return chart.save(fig, dest, "unit_completion" if unit else "batch_completion", title,
        ("每角色内独立计数，不重复计算父批与子批。不可编码含服务拒绝与有限证据失败隔离，原因及分母另表保存。检查分母仅含抽中条目；未冻结清单时不虚构检查总数，最终标签等待两个阶段完成。" if screening else
         "每角色内独立计数；不把拆批与父批重复相加。裁决分母只含当前已产生的计划，final 须等待全流程完成。"),
        note, n=int(frame.total.sum()), bottom=.18)


def requests_chart(frame, dest, note):
    fig, ax = plt.subplots(figsize=(10, 5))
    outcomes = [("fully_valid", "整批通过", "Fully valid", chart.TEAL),
                ("partially_valid", "部分项通过", "Partially valid", chart.ORANGE),
                ("invalid", "未采信输出", "Not accepted", chart.RED),
                ("started_unconfirmed", "在途／未确认", "Unconfirmed", chart.GRAY)]
    roles = [r for r in ROLE if r != "final" and r in set(frame.role)]
    roles += sorted(set(frame.role) - set(roles) - {"final"})
    bottom = np.zeros(len(roles))
    if not roles:
        chart.empty(ax, chart.text("尚无模型请求", "No model requests"))
    else:
        for key, zh, en, color in outcomes:
            values = [int((frame.role.eq(role) & frame.outcome.eq(key)).sum()) for role in roles]
            bars = ax.bar(np.arange(len(roles)), values, bottom=bottom, color=color, label=chart.text(zh, en), width=.6)
            ax.bar_label(bars, labels=[str(n) if n else "" for n in values], label_type="center", fontsize=10)
            bottom += values
        ax.set_xticks(np.arange(len(roles)), [role_label(r) for r in roles])
        ax.set_ylabel(chart.text("请求次数（含重试）", "Requests, including retries"))
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.13), ncol=2, fontsize=9)
        ax.set_ylim(0, max(bottom) * 1.2)
    return chart.save(fig, dest, "request_outcomes", chart.text("每次模型请求的输出状态", "Outcomes of model requests"),
        "部分项通过的请求仍可能触发补编；历史错误不等于最终失败单元。请求结果不是语义准确率。", note, n=len(frame))


def errors_chart(frame, dest, note):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    values = frame.groupby("status").requests.sum().sort_values() if len(frame) else pd.Series(dtype=float)
    if values.empty:
        chart.empty(ax, chart.text("尚无记录的请求错误", "No recorded request errors"))
    else:
        bars = ax.barh(values.index, values, color=chart.RED, height=.55)
        ax.bar_label(bars, padding=4)
        ax.set_xlim(0, values.max() * 1.18)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(chart.text("历史请求数，含后来已恢复的请求", "Historical requests, including subsequently recovered tasks"))
    return chart.save(fig, dest, "request_errors", chart.text("请求校验与接口错误分布", "Request validation and API errors"),
        "按请求错误代码计数；部分合格响应也列入相应错误。不能把本图相加当作失败评论总数。", note, n=int(values.sum()))


def selected_labels(distribution, role):
    selected = distribution.loc[distribution.role.eq(role) & distribution.scope.eq("audience")].copy()
    part = selected.loc[(selected.label.eq("substantive_response") & selected.position.eq("reply")) |
                        (~selected.label.eq("substantive_response") & selected.position.eq("all"))]
    return part.set_index("label").reindex(COMMENT_LABELS).reset_index()


def labels_chart(distribution, dest, note, missingness=False, screening=False):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11.2))
    eligible_roles, missing_estimates = 0, False
    roles = ["coder", "checker", "final"] if screening or "checker" in distribution.role.values else ["coder", "reviewer", "adjudicator", "final"]
    for ax in list(axes.flat)[len(roles):]:
        ax.set_visible(False)
    for ax, role in zip(axes.flat, roles):
        part = selected_labels(distribution, role)
        ax.set_title(role_label(role))
        if not part.successful_comments.sum():
            chart.empty(ax, chart.text("此阶段尚无可展示的完成标签", "No completed labels for this stage"))
            continue
        eligible_roles += int(missingness or part.resolved.sum() > 0)
        missing_estimates |= bool(not missingness and part.resolved.eq(0).any())
        y = np.arange(len(part))
        names = [chart.label(row.label) + (" *" if row.label == "substantive_response" else "") for row in part.itertuples()]
        if missingness:
            left = np.zeros(len(part))
            for key, zh, en, color in [("positive", "是（1）", "Positive", chart.TEAL),
                  ("negative", "否（0）", "Negative", "#bdcfdb"), ("abstained", "未决（null）", "Abstained", chart.ORANGE)]:
                ratios = (part[key] / part.successful_comments.replace(0, np.nan)).fillna(0).to_numpy()
                ax.barh(y, ratios, left=left, color=color, height=.65, label=chart.text(zh, en))
                left += ratios
            for i, row in enumerate(part.itertuples()):
                ax.text(1.02, i, f"null={row.abstained}/{row.successful_comments}", va="center", fontsize=8)
            ax.set_xlim(0, 1.4)
            ax.set_xticks([0, .5, 1])
        else:
            values = pd.to_numeric(part.positive_over_resolved, errors="coerce")
            good = values.notna()
            ax.barh(y[good], values[good], color=chart.TEAL, height=.6)
            for i, row in enumerate(part.itertuples()):
                text = f"{row.positive}/{row.resolved}" if row.resolved else chart.text("无已判断项", "No resolved labels")
                ax.text(1.02, i, text, va="center", fontsize=8)
            ax.set_xlim(0, 1.4)
            ax.set_xticks([0, .5, 1])
        ax.set_yticks(y, names, fontsize=9)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="x", alpha=.12)
        ax.set_xlabel(chart.text("各状态 / 完成评论" if missingness else "阳性 / 该标签已判断项",
                                 "State / completed comments" if missingness else "Positive / resolved labels"), fontsize=9)
    if missingness and eligible_roles:
        handles, names = next(ax.get_legend_handles_labels() for ax in axes.flat if ax.get_legend_handles_labels()[0])
        fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .035), ncol=3, frameon=False)
    title = chart.text("成功评论中各标签的确定与弃权" if missingness else "成功评论中各类参与的阳性占比",
                       "Label resolution among completed comments" if missingness else "Positive labels among resolved comments")
    caption = ("受众评论，保留垃圾标记；每行以该标签已判断数为分母。右侧是阳性数／已判断数。各类可重叠。"
               if not missingness else "受众评论，保留垃圾标记；每行以该阶段完成评论数为分母，null 单列。右侧是未决数／完成数。")
    caption += " * 实质回复只用回复作为分母。排除明确垃圾的子集、根/回复细分见标签源表。"
    note += chart.text("  受众评论；* 实质回复仅以回复为分母。", "  Audience comments; * denominator includes replies only.")
    return chart.save(fig, dest, "label_missingness" if missingness else "comment_label_prevalence", title, caption,
                      note, "available" if eligible_roles == len(roles) and not missing_estimates else "partial" if eligible_roles else "unavailable",
                      eligible_roles, bottom=.1)


def strata_chart(frame, dest, note, screening=False):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    for ax, dimension, order in [(axes[0], "length_band", ["empty", "1-20", "21-80", "81-240", "241+"]),
                                 (axes[1], "language_group", ["es", "und", "other"])]:
        second = "checker" if screening or "checker" in frame.role.values else "reviewer"
        for offset, role, color in [(-.2, "coder", chart.BLUE), (.2, second, chart.TEAL)]:
            part = frame.loc[frame.dimension.eq(dimension) & frame.role.eq(role)].set_index("stratum").reindex(order)
            valid = part.total.notna()
            y = np.arange(len(order)) + offset
            ax.barh(y[valid], part.loc[valid, "completion_rate"], height=.36, color=color, label=role_label(role))
            for pos, row in zip(y[valid], part.loc[valid].itertuples(), strict=True):
                ax.text(1.01, pos, f"{int(row.complete)}/{int(row.total)}", va="center", fontsize=8,
                        transform=ax.get_yaxis_transform(), clip_on=False)
        ax.set_yticks(np.arange(len(order)), order)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        max_rate = frame.loc[frame.dimension.eq(dimension), "completion_rate"].max()
        ax.set_xlim(0, min(1.01, max(.01, max_rate * 1.2)))
        ax.set_xlabel(chart.text("处理覆盖率；右侧为通过数／目标数", "Completion; right labels = validated / targets"))
        ax.set_title(chart.text("按评论字数" if dimension == "length_band" else "按自动语言标记",
                               "Comment length" if dimension == "length_band" else "Detected language"))
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .045), ncol=2, frameon=False)
    return chart.save(fig, dest, "completion_by_stratum", chart.text("不同评论群体的处理覆盖率", "Completion coverage across comment groups"),
        "柱上为通过数／该群体全部目标数；不按请求数加权。早期差异可能来自任务排序，不能认定随机缺失；und 是语言未定。",
        note, n=int(frame.loc[frame.dimension.eq("length_band") & frame.role.eq("coder"), "total"].sum()), bottom=.16)


def cost_chart(frame, dest, note):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    part = frame.sort_values("attempt_id")
    if part.empty:
        for ax in axes:
            chart.empty(ax)
    else:
        x = np.arange(1, len(part) + 1)
        known = pd.to_numeric(part.reported_tokens, errors="coerce")
        axes[0].plot(x, known.fillna(0).cumsum() / 1000, color=chart.BLUE)
        axes[0].set(xlabel=chart.text("累计请求序号", "Cumulative request index"),
                    ylabel=chart.text("已知累计 tokens（千）", "Known cumulative tokens (thousands)"))
        axes[0].set_title(chart.text(f"服务用量未知：{known.isna().sum()} 次", f"Unknown token usage: {known.isna().sum()} requests"))
        seconds = pd.to_numeric(part.elapsed_seconds, errors="coerce")
        good = seconds.notna()
        axes[1].scatter(x[good], seconds[good], color=chart.TEAL, s=22)
        axes[1].set(xlabel=chart.text("累计请求序号", "Cumulative request index"),
                    ylabel=chart.text("响应耗时（秒）", "Response duration (seconds)"))
    return chart.save(fig, dest, "request_costs", chart.text("模型请求用量与响应耗时", "Reported usage and response duration"),
        "按请求发出序号排列，包含失败和重试。累计量只计服务已报告 tokens，未知用量不等于免费；并行响应时长之和不是墙钟耗时，也不是金额。",
        note, n=len(part))


def generate(dest, tables, final_ready, screening=False):
    folder = dest / "figures"
    note = chart.text("全量模型编码；统计描述不代表语义准确率。", "Complete AI coding; description does not establish accuracy.") if final_ready else chart.text(
        "阶段性审计：实验未完成，类别分布不能推广至全体评论。", "Interim audit: incomplete coding; distributions are not population results.")
    if screening:
        note += chart.text("  初编为全量；检查仅含重点和随机抽中条目，未抽中不等于通过检查。",
                           "  Coder covers all units; checker covers only selected units, not population validation.")
    with chart.style():
        figures = [requests_chart(tables["request_attempts.csv"], folder, note),
            errors_chart(tables["request_errors.csv"], folder, note),
            progress_chart(tables["batch_summary.csv"], folder, note, False, screening),
            progress_chart(tables["unit_summary.csv"], folder, note, True, screening),
            labels_chart(tables["comment_label_distribution.csv"], folder, note, screening=screening),
            labels_chart(tables["comment_label_distribution.csv"], folder, note, True, screening),
            strata_chart(tables["completion_by_stratum.csv"], folder, note, screening),
            cost_chart(tables["request_attempts.csv"], folder, note)]
    manifest = {"created_at": now(), "figure_version": 2, "workflow": "screening" if screening else "legacy_three_pass", "figures": figures,
        "font": chart.FONT or "DejaVu Sans", "source_note": note,
        "source_hashes": {name: stable_hash((dest / name).read_bytes()) for name in tables}}
    write_json(folder / "manifest.json", manifest)
    return manifest

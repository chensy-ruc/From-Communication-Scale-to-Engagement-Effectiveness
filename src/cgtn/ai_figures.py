"""Publication figures from saved statistics; no model calls or new inferential tests."""
from __future__ import annotations

import inspect
import textwrap
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

from .core import now, stable_hash, write_json
from .progress import event

BLUE, TEAL, ORANGE, RED, GRAY = "#336b87", "#278c80", "#d68a3c", "#be5b53", "#b7c2c8"
_FONTS = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Arial Unicode MS", "Noto Sans CJK SC", "Microsoft YaHei", "Heiti SC"] if f in _FONTS), None)
LABELS = {
    "deep_rate": ("认知参与率", "Cognitive participation"),
    "dialogue_conversion_rate": ("可观测对话比例", "Observed dialogue"),
    "learning_expression_rate": ("学习表达率", "Learning expression"),
    "sharing_intent_rate": ("分享意向率", "Sharing intent"),
    "topic": ("视频主题", "Video topic"), "form": ("视频形式", "Video form"),
    "information_request": ("信息询问", "Information request"), "fact_check": ("事实核验行为", "Fact-checking behavior"),
    "opinion": ("观点表达", "Opinion"), "reasoned_opinion": ("有论据的观点", "Reasoned opinion"),
    "cross_cultural": ("跨文化比较", "Cross-cultural comparison"), "emotion": ("情感表达", "Emotion"),
    "substantive_response": ("实质回复", "Substantive reply"),
    "learning_or_reconsideration": ("学习／重新考虑", "Learning / reconsideration"),
    "sharing_intent": ("分享意向", "Sharing intent"), "spam": ("垃圾信息", "Spam"),
    "china_governance": ("中国治理", "China governance"), "economy_technology": ("经济科技", "Economy / technology"),
    "international_relations": ("国际关系", "International relations"), "culture_society": ("文化社会", "Culture / society"),
    "people_exchange": ("人文交流", "People-to-people exchange"), "other": ("其他", "Other"),
    "log_views": ("log(1 + 观看量)", "log(1 + views)"), "log_duration": ("log(1 + 时长)", "log(1 + duration)"),
    "log_age": ("log(1 + 视频年龄)", "log(1 + age)"), "root_cognitive_rate": ("根评论认知比例", "Root cognitive rate"),
}
METRICS = ["deep_rate", "dialogue_conversion_rate", "learning_expression_rate", "sharing_intent_rate"]


def text(zh, en):
    return zh if FONT else en


def label(key):
    return text(*LABELS[key]) if key in LABELS else str(key).replace("_", " ")


@contextmanager
def style():
    with plt.rc_context({"font.family": FONT or "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "savefig.dpi": 220, "figure.facecolor": "white"}):
        yield


def empty(ax, message=None):
    ax.set_axis_off()
    ax.text(.5, .5, message or text("没有可绘制的有效数据", "No eligible observations"),
            ha="center", va="center", transform=ax.transAxes, color="#65747e", wrap=True)


def numeric(frame, columns):
    return frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def save(fig, dest, slug, title, caption, note, status="available", n=0, bottom=.075):
    dest.mkdir(parents=True, exist_ok=True)
    try:
        fig.suptitle(title, fontsize=15, y=.99)
        fig.text(.02, .013, note, color="#65747e", fontsize=9)
        fig.tight_layout(rect=(0, bottom, 1, .94), h_pad=2, w_pad=2.8)
        for suffix in ["png", "pdf"]:
            fig.savefig(dest / f"{slug}.{suffix}", facecolor="white")
    finally:
        plt.close(fig)
    event("制图", f"已输出 {slug}.png / .pdf；状态 {status}。")
    return {"name": slug, "title": title, "caption": caption, "png": f"{slug}.png", "pdf": f"{slug}.pdf",
            "status": status, "n": int(n)}


def overview(videos, dest, slug="sample_overview", note=None):
    """Metadata-only overview, also usable before any semantic coding."""
    with style():
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
        monthly = videos.groupby("month", sort=True).size()
        if len(monthly):
            bars = axes[0].bar(np.arange(len(monthly)), monthly, color=BLUE, width=.7)
            axes[0].set_xticks(np.arange(len(monthly)), [str(x)[-2:] for x in monthly.index])
            axes[0].bar_label(bars, padding=3, fontsize=8)
            axes[0].set_ylim(0, monthly.max() * 1.18)
            axes[0].set(xlabel=text("发布月份（UTC）", "Publication month (UTC)"), ylabel=text("视频数", "Videos"))
        else:
            empty(axes[0])
        views = numeric(videos, ["view_count"]).view_count.dropna()
        views = views[views.ge(0)]
        if len(views):
            axes[1].hist(np.log10(1 + views), bins=35, color=TEAL, edgecolor="white", linewidth=.5)
            axes[1].set(xlabel=text("log10(1 + 采集时累计观看量)", "log10(1 + cumulative views at collection)"),
                        ylabel=text("视频数", "Videos"))
        else:
            empty(axes[1])
        axes[0].set_title(text("月度视频数量", "Monthly video counts"))
        axes[1].set_title(text("观看量分布", "Views distribution"))
        return save(fig, dest, slug, text(f"公开视频样本概览 · N={len(videos):,}", f"Public video frame · N={len(videos):,}"),
                    "月份按 UTC 发布时间统计；观看量为采集时累计值，横轴作 log10(1+x) 变换。", note or
                    text("元数据图；不使用 AI 标签。累计观看量不代表发布当年末的历史计数。",
                         "Metadata only. Views are measured at collection, not at the end of the publication year."),
                    "available" if len(views) else "unavailable", len(videos))


def agreement_chart(frame, dest, note):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(5, len(frame) * .48 + 2)), sharey=True)
    if frame.empty:
        for ax in axes:
            empty(ax)
    else:
        y = np.arange(len(frame))
        names = [f"{label(r.label)}  (n={int(r.n_units):,})" for r in frame.itertuples()]
        for offset, col, color, title in [(-.24, "both_coverage", GRAY, text("两轮均已判断", "Both resolved")),
              (0, "agreement_on_resolved", BLUE, text("已判断项一致率", "Agreement among resolved")),
              (.24, "final_coverage", TEAL, text("裁决后覆盖率", "Final coverage"))]:
            values = numeric(frame, [col])[col]
            good = values.notna()
            axes[0].barh(y[good] + offset, values[good], height=.22, color=color, label=title)
        axes[0].set_yticks(y, names)
        axes[0].invert_yaxis()
        axes[0].set_xlim(0, 1.04)
        axes[0].xaxis.set_major_formatter(PercentFormatter(1))
        axes[0].set_title(text("标签覆盖与一致比例", "Coverage and agreement"))
        values = numeric(frame, ["kappa_ai_vs_ai"]).kappa_ai_vs_ai
        good = values.notna()
        axes[1].scatter(values[good], y[good], color=BLUE, s=36, zorder=3)
        for position in y[~good]:
            axes[1].text(.04, position, text("不可估计", "Not estimable"), color="#65747e",
                         transform=axes[1].get_yaxis_transform(), va="center")
        axes[1].set(xlim=(-1.05, 1.05), xlabel="Cohen κ", title=text("裁决前 AI 一致性", "Pre-adjudication AI consistency"))
        axes[1].axvline(0, color=GRAY, linewidth=1)
        for ax in axes:
            ax.grid(axis="x", alpha=.15)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .032), ncol=3, frameon=False, fontsize=9)
    status = "unavailable" if frame.empty else "partial" if frame.kappa_ai_vs_ai.isna().any() else "available"
    return save(fig, dest, "ai_agreement", text("AI 编码一致性与覆盖率", "AI coding consistency and coverage"),
                "n 为各标签的编码单元数。一致率以两轮均非缺失项为分母；常量／无配对标签的 κ 不画为零。AI 一致性不是准确率。",
                note, status, len(frame))


def coverage_chart(frame, dest, note):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    statuses = {
        "ai_only_exploratory": ("可计算", "Observed", TEAL), "measured": ("可计算", "Observed", TEAL),
        "no_valid_denominator_or_model_abstention": ("分母无效／标签未决", "Invalid denominator / abstention", ORANGE),
        "no_audience_thread": ("无受众线程", "No audience thread", GRAY),
        "collection_unavailable": ("采集不可用", "Collection unavailable", RED),
        "validation_failed": ("验证未通过", "Validation failed", "#8d79a6"),
    }
    table = frame.pivot_table(index="metric", columns="status", values="videos", aggfunc="sum", fill_value=0).reindex(METRICS).fillna(0)
    totals = table.sum(axis=1)
    if not totals.gt(0).any():
        empty(ax)
    else:
        left = np.zeros(len(table))
        for key in list(statuses) + [x for x in table.columns if x not in statuses]:
            if key not in table or not table[key].sum():
                continue
            zh, en, color = statuses.get(key, (key, key, "#777777"))
            ax.barh(np.arange(len(table)), table[key], left=left, color=color, height=.6, label=text(zh, en))
            left += table[key].to_numpy()
        for i, metric in enumerate(table.index):
            observed = sum(table.loc[metric].get(k, 0) for k in ["ai_only_exploratory", "measured"])
            ax.text(totals.iloc[i] + max(totals) * .015, i, f"n={int(observed):,}/{int(totals.iloc[i]):,}", va="center", fontsize=9)
        ax.set_yticks(np.arange(len(table)), [label(x) for x in table.index])
        ax.invert_yaxis()
        ax.set_xlim(0, totals.max() * 1.26)
        ax.set_xlabel(text("视频数量；右侧 n 为可计算视频数／全部视频数", "Videos; right-hand n = observed / all videos"))
        ax.grid(axis="x", alpha=.15)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .032), ncol=2, frameon=False, fontsize=8)
    return save(fig, dest, "measurement_coverage", text("参与指标的可测量范围", "Measurement coverage"),
                "缺失原因来自 measurement_coverage.csv。空值、不可获取和无有效分母均未计为低质量零值。", note,
                "available" if totals.gt(0).any() else "unavailable", int(totals.max()) if len(totals) else 0, bottom=.15)


def scatter_chart(frame, correlations, dest, note):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    available = 0
    for ax, metric in zip(axes, METRICS[:2]):
        d = numeric(frame, ["view_count", metric]).dropna()
        d = d[d.view_count.ge(0)]
        ax.set_title(label(metric))
        if d.empty:
            empty(ax)
            continue
        available += 1
        ax.scatter(np.log10(1 + d.view_count), d[metric], color=BLUE, alpha=.3, s=16, linewidths=0, rasterized=True)
        ax.set(xlabel=text("log10(1 + 累计观看量)", "log10(1 + cumulative views)"), ylim=(-.035, 1.06))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.15)
        match = correlations[correlations.y.eq(metric)]
        rho = match.iloc[0].rho if len(match) else np.nan
        value = f"ρ={rho:.3f}" if np.isfinite(rho) else text("ρ 不可估计", "ρ not estimable")
        ax.text(.04, .96, f"n={len(d):,}\n{value}", transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    return save(fig, dest, "scale_participation", text("传播规模与参与指标", "Reach and participation"),
                "每点代表一个视频；观看量作 log10(1+x) 变换。n 和 Spearman ρ 使用各指标有效视频；缺失值未填零。",
                note, "available" if available == 2 else "partial" if available else "unavailable", len(frame))


def correlation_chart(frame, dest, note, variants=None):
    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    available = 0
    for variant, offset, color, marker, title in variants or [
        ("adjudicated", -.15, BLUE, "o", text("裁决后标签", "Adjudicated labels")),
        ("blind_agreement_only", .15, ORANGE, "D", text("仅保留盲编码一致标签", "Blind agreement only"))]:
        ax.plot([], [], marker=marker, color=color, linestyle="none", label=title)
        for i, metric in enumerate(METRICS):
            rows = frame[frame.variant.eq(variant) & frame.metric.eq(metric)]
            row = rows.iloc[0] if len(rows) else None
            y = i + offset
            if row is None or not np.isfinite(row.rho):
                ax.text(.04, y, text("不可估计", "Not estimable"), transform=ax.get_yaxis_transform(), color="#65747e", va="center", fontsize=8)
            else:
                available += 1
                ax.plot(row.rho, y, marker=marker, color=color, markersize=5)
                if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                    ax.hlines(y, row.ci_low, row.ci_high, color=color, linewidth=1.6)
                else:
                    ax.annotate(text("区间缺失", "CI unavailable"), (row.rho, y), xytext=(7, 3), textcoords="offset points", fontsize=7)
            n = int(row.n) if row is not None and pd.notna(row.n) else 0
            ax.text(1.025, y, f"n={n:,}", color=color, va="center", fontsize=9, transform=ax.get_yaxis_transform())
    ax.set_yticks(np.arange(len(METRICS)), [label(x) for x in METRICS])
    ax.set(xlim=(-1.04, 1.04), ylim=(len(METRICS) - .5, -.5), xlabel=text("Spearman ρ 与 95% bootstrap 区间", "Spearman ρ and 95% bootstrap interval"))
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.grid(axis="x", alpha=.15)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .032), ncol=2, frameon=False, fontsize=9)
    return save(fig, dest, "correlation_sensitivity", text("规模关联及编码方案敏感性", "Reach correlations and coding sensitivity"),
                "读取既有相关与分层 bootstrap 结果，不新增检验。两种编码方案的有效样本可能不同；区间不包含 AI 误分类误差。",
                note, "available" if available == 8 else "partial" if available else "unavailable", available)


def groups_chart(frame, dest, note, min_group_n):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    known = frame[frame.topic.notna() & ~frame.topic.isin(["", "unknown", "not_measured"])]
    available = 0
    for ax, metric in zip(axes, ["view_count"] + METRICS[:2]):
        ax.set_title(text("观看量", "Views") if metric == "view_count" else label(metric))
        groups = []
        for topic, g in known.groupby("topic", sort=True):
            values = numeric(g, [metric])[metric].dropna()
            if metric == "view_count":
                values = np.log10(1 + values[values.ge(0)])
            if len(values) >= min_group_n:
                groups.append((topic, values))
        if len(groups) < 2:
            empty(ax, text(f"有效主题组不足 2 组\n每组至少 {min_group_n} 个有效视频", f"Fewer than two eligible topics\nMinimum {min_group_n} videos per group"))
            continue
        available += 1
        orientation = {"orientation": "horizontal"} if "orientation" in inspect.signature(ax.boxplot).parameters else {"vert": False}
        bp = ax.boxplot([v for _, v in groups], patch_artist=True, showfliers=False, **orientation)
        for patch in bp["boxes"]:
            patch.set(facecolor=BLUE, alpha=.65)
        for line in bp["medians"]:
            line.set(color="#24333e", linewidth=1.5)
        names = [f"{textwrap.fill(label(k), 18)}\n(n={len(v):,})" for k, v in groups]
        ax.set_yticks(np.arange(1, len(groups) + 1), names, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
        if metric == "view_count":
            ax.set_xlabel("log10(1 + views)")
        else:
            ax.set_xlim(-.03, 1.03)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
    return save(fig, dest, "topic_groups", text("主题组之间的分布差异", "Distributions by topic"),
                f"箱体为四分位区间、横线为中位数；须线采用 1.5×IQR 规则，图中隐藏离群点。每组至少 {min_group_n} 个有效视频；统计检验仍见 groups_*.csv。",
                note, "available" if available == 3 else "partial" if available else "unavailable", len(known))


def regression_chart(coefficients, statuses, dest, note):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    terms = ["log_views", "log_duration", "log_age", "root_cognitive_rate"]
    available = 0
    models = [("machine_quality_binomial", text("认知参与模型", "Cognitive participation model")),
              ("machine_conversion_binomial", text("可观测对话模型", "Observed dialogue model"))]
    for ax, (model, title) in zip(axes, models):
        ax.set_title(title)
        records = statuses[statuses.model.eq(model)]
        record = records.iloc[0] if len(records) else None
        d = coefficients[coefficients.model.eq(model) & coefficients.term.isin(terms)].copy()
        if record is None or record.status != "ok" or d.empty:
            reason = str(record.get("reason", "no_estimate")) if record is not None else "no_model_record"
            empty(ax, text("模型未估计\n", "Model not estimated\n") + textwrap.fill(reason, 32))
            continue
        d = d.set_index("term").reindex([x for x in terms if x in set(d.term)])
        d = d[np.isfinite(d.coef)]
        if d.empty:
            empty(ax)
            continue
        available += 1
        for i, row in enumerate(d.itertuples()):
            ax.plot(row.coef, i, "o", color=BLUE)
            if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                ax.hlines(i, row.ci_low, row.ci_high, color=BLUE, linewidth=1.7)
        ax.set_yticks(np.arange(len(d)), [label(x) for x in d.index], fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color=GRAY, linewidth=1)
        ax.grid(axis="x", alpha=.15)
        ax.set_xlabel(text("log OR 与 95% 稳健置信区间", "log OR and robust 95% confidence interval"))
        ax.set_title(f"{title} · n={int(record.n):,}")
    return save(fig, dest, "regression_coefficients", text("主要回归系数", "Main regression coefficients"),
                "展示分组二项模型的连续变量系数（log OR），保留原变换尺度。主题／形式等分类控制变量和截距见完整系数表；区间不包含编码误差，不作因果解释。",
                note, "available" if available == 2 else "partial" if available else "unavailable", available)


def generate(agent_dest, min_group_n=5, note=None):
    """Seven figures, backed by the saved source tables and an auditable file manifest."""
    source_names = ["ai_agreement.csv", "analysis/analysis_videos.csv", "analysis/measurement_coverage.csv",
                    "analysis/correlations.csv", "analysis/sensitivity_correlations.csv",
                    "analysis/models_coefficients.csv", "analysis/models_status.csv"]
    sources = {name: pd.read_csv(agent_dest / name) for name in source_names}
    frame = sources["analysis/analysis_videos.csv"]
    dest = agent_dest / "analysis/figures"
    note = note or text("AI 模型标签的探索性分析；统计区间不包含编码误差。",
                        "Exploratory AI-coded analysis; intervals exclude coding error.")
    with style():
        plots = [overview(frame, dest, note=note),
                 agreement_chart(sources["ai_agreement.csv"], dest, note),
                 coverage_chart(sources["analysis/measurement_coverage.csv"], dest, note),
                 scatter_chart(frame, sources["analysis/correlations.csv"], dest, note),
                 correlation_chart(sources["analysis/sensitivity_correlations.csv"], dest, note),
                 groups_chart(frame, dest, note, min_group_n),
                 regression_chart(sources["analysis/models_coefficients.csv"], sources["analysis/models_status.csv"], dest, note)]
    manifest = {"created_at": now(), "figure_version": 1, "source_note": note,
                "font": FONT or "DejaVu Sans", "language": "zh" if FONT else "en",
                "source_hashes": {name: stable_hash((agent_dest / name).read_bytes()) for name in source_names}, "figures": plots}
    write_json(dest / "manifest.json", manifest)
    return manifest


def markdown(manifest):
    lines = ["", "## 图表", "", "PNG 供直接阅读，PDF 可用于论文排版；源表哈希和缺失状态见 [图表清单](analysis/figures/manifest.json)。"]
    for i, figure in enumerate(manifest["figures"], 1):
        lines += ["", f"### 图 {i}：{figure['title']}", "",
                  f"![{figure['title']}](analysis/figures/{figure['png']})", "",
                  figure["caption"] + f" [下载 PDF](analysis/figures/{figure['pdf']})"]
    return lines

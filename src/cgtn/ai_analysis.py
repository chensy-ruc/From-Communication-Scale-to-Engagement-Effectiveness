"""Machine-coded outcomes with explicit validation and missingness gates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ai_client as client
from . import ai_data as data
from . import ai_eval
from .analysis import boot_correlation, comparisons, correct_p, descriptive, figures
from .coding import COGNITIVE
from .core import write_csv, write_json
from .progress import event
from .structure import fit_models


def participation(comments, gate):
    rows = []
    requirements = {"deep_rate": ["spam"] + COGNITIVE,
                    "dialogue_conversion_rate": ["spam", "substantive_response"],
                    "learning_expression_rate": ["spam", "learning_or_reconsideration"],
                    "sharing_intent_rate": ["spam", "sharing_intent"]}
    comments = comments.copy()
    # OR is identifiable if one label is 1, or if every component is a known 0.
    comments["_cognitive"] = np.where(comments[COGNITIVE].eq(1).any(axis=1), 1.0,
                                       np.where(comments[COGNITIVE].notna().all(axis=1), 0.0, np.nan))
    comments["_time"] = pd.to_datetime(comments.published_at, utc=True, format="ISO8601")
    audience_all = comments[comments.is_channel_author.eq(0)]
    known_roots = audience_all[audience_all.is_reply.eq(0) & audience_all.author_hash.notna()
                              & audience_all.author_hash.ne("") & audience_all.spam.eq(0)]
    root_context = known_roots[["video_id", "root_id", "author_hash", "_time"]].rename(
        columns={"author_hash": "_root_author", "_time": "_root_time"})
    replies = audience_all[audience_all.is_reply.eq(1) & audience_all.author_hash.notna()
                           & audience_all.author_hash.ne("")].merge(root_context,
                            on=["video_id", "root_id"], how="inner", validate="many_to_one")
    replies = replies[replies.author_hash.ne(replies._root_author) & replies._time.ge(replies._root_time)].copy()
    replies["_positive"] = replies.spam.eq(0) & replies.substantive_response.eq(1)
    replies["_possible"] = ~replies.spam.eq(1) & ~replies.substantive_response.eq(0)
    thread = replies.groupby(["video_id", "root_id"])[["_positive", "_possible"]].any()
    thread["_dialogue"] = np.where(thread._positive, 1.0, np.where(thread._possible, np.nan, 0.0))
    # No eligible later reply establishes zero for this observable thread proxy.
    root_states = root_context.merge(thread[["_dialogue"]].reset_index(), on=["video_id", "root_id"],
                                    how="left", indicator=True, validate="one_to_one")
    root_states.loc[root_states._merge.eq("left_only"), "_dialogue"] = 0.0
    dialogue_by_video = {v: g._dialogue for v, g in root_states.groupby("video_id", sort=False)}
    spam_validated = gate.get("comments.spam", False)
    cognitive_validated = all(gate.get("comments." + c, False) for c in requirements["deep_rate"])
    dialogue_validated = all(gate.get("comments." + c, False) for c in requirements["dialogue_conversion_rate"])
    for vid, group in comments.groupby("video_id", sort=True):
        audience = group[group.is_channel_author.eq(0)]
        spam_ready = spam_validated and audience.spam.notna().all()
        users = audience[audience.spam.eq(0)]
        row = {"video_id": vid, "machine_comment_rows": len(group),
               "n_unavailable_comments": int(audience.coding_status.eq("unavailable").sum()) if "coding_status" in audience else 0,
               "n_audience_comments": len(users) if spam_ready else np.nan,
               "n_cognitive_comments": np.nan, "deep_rate": np.nan, "n_known_roots": np.nan,
               "n_dialogue_roots": np.nan, "dialogue_conversion_rate": np.nan, "root_cognitive_rate": np.nan,
               "learning_expression_rate": np.nan, "sharing_intent_rate": np.nan,
               "n_unresolved_spam_comments": int(audience.spam.isna().sum()),
               "n_unresolved_cognitive_comments": int(users._cognitive.isna().sum()),
               "n_unresolved_dialogue_roots": np.nan,
               "deep_rate_lower": np.nan, "deep_rate_upper": np.nan,
               "dialogue_conversion_rate_lower": np.nan, "dialogue_conversion_rate_upper": np.nan}
        if spam_ready and cognitive_validated and len(users):
            positive, unknown = int(users._cognitive.eq(1).sum()), int(users._cognitive.isna().sum())
            row["deep_rate_lower"], row["deep_rate_upper"] = positive / len(users), (positive + unknown) / len(users)
            if not unknown:
                row["n_cognitive_comments"], row["deep_rate"] = positive, positive / len(users)
        candidates = audience[audience.is_reply.eq(0) & audience.author_hash.notna() & audience.author_hash.ne("")]
        roots = candidates[candidates.spam.eq(0)]
        roots_ready = spam_validated and candidates.spam.notna().all()
        if roots_ready:
            row["n_known_roots"] = len(roots)
        if roots_ready and cognitive_validated and len(roots) and roots._cognitive.notna().all():
            row["root_cognitive_rate"] = roots._cognitive.mean()
        if roots_ready and dialogue_validated and len(roots):
            values = dialogue_by_video[vid]
            positive, unknown = int(values.eq(1).sum()), int(values.isna().sum())
            row["n_unresolved_dialogue_roots"] = unknown
            row["dialogue_conversion_rate_lower"] = positive / len(roots)
            row["dialogue_conversion_rate_upper"] = (positive + unknown) / len(roots)
            if not unknown:
                row["n_dialogue_roots"], row["dialogue_conversion_rate"] = positive, positive / len(roots)
        for metric, label in [("learning_expression_rate", "learning_or_reconsideration"), ("sharing_intent_rate", "sharing_intent")]:
            if spam_ready and gate.get("comments." + label, False) and len(users) and users[label].notna().all():
                row[metric] = users[label].eq(1).mean()
        rows.append(row)
    columns = ["video_id", "machine_comment_rows", "n_unavailable_comments", "n_audience_comments", "n_cognitive_comments", "deep_rate", "n_known_roots",
               "n_dialogue_roots", "dialogue_conversion_rate", "root_cognitive_rate", "learning_expression_rate", "sharing_intent_rate",
               "n_unresolved_spam_comments", "n_unresolved_cognitive_comments", "n_unresolved_dialogue_roots",
               "deep_rate_lower", "deep_rate_upper", "dialogue_conversion_rate_lower", "dialogue_conversion_rate_upper"]
    return pd.DataFrame(rows, columns=columns), requirements


def analyze(p, experiment):
    root = data.folder(p, experiment)
    report = ai_eval.checked(p, experiment)
    prediction = client.predictions(root, report["version"], "all")
    dest = p.reports / "ai" / experiment / "analysis"
    analyze_predictions(p, root, prediction, report, dest)


def analysis_frame(root, prediction, report):
    automated = report["reference_type"] == "none_ai_only"
    videos = pd.read_csv(root / "videos.csv", keep_default_na=False)
    comments = pd.read_csv(root / "comments.csv", keep_default_na=False)
    gate = report["gate"]
    for kind, frame, key in [("comments", comments, "comment_id"), ("videos", videos, "video_id")]:
        values = prediction[kind]
        if not values.id.is_unique or set(values.id) != set(frame[key]):
            raise ValueError(f"{kind} 预测 ID 未完整覆盖冻结输入；拒绝缩小分析分母")
    comments = comments.merge(prediction["comments"].drop(columns="split").rename(columns={"id": "comment_id"}),
                              on="comment_id", validate="one_to_one")
    metrics, requirements = participation(comments, gate)
    d = videos.merge(metrics, on="video_id", how="left", validate="one_to_one")
    coverage = pd.read_csv(root / "coverage.csv", keep_default_na=False).rename(columns={"status": "collection_status"})
    d = d.merge(coverage, on="video_id", how="left", validate="one_to_one")
    empty_complete = d.machine_comment_rows.isna() & d.collection_status.eq("complete")
    d.loc[empty_complete, "machine_comment_rows"] = 0
    d.loc[empty_complete, "n_unavailable_comments"] = 0
    if gate.get("comments.spam"):
        d.loc[empty_complete, ["n_audience_comments", "n_known_roots"]] = 0
    for metric, dependencies in requirements.items():
        status_column = metric + "_status"
        d[status_column] = np.where(d[metric].notna(), "ai_only_exploratory" if automated else "measured",
                                    "no_valid_denominator_or_model_abstention")
        d.loc[d[metric].isna() & d.n_unavailable_comments.gt(0), status_column] = "provider_unavailable_or_abstention"
        if not all(gate.get("comments." + c, False) for c in dependencies):
            d[status_column] = "validation_failed"
        d.loc[empty_complete, status_column] = "no_audience_thread"
        d.loc[~d.collection_status.eq("complete"), status_column] = "collection_unavailable"
    video_labels = prediction["videos"].set_index("id")
    for label in data.VIDEO_FIELDS:
        d[label] = d.video_id.map(video_labels[label]) if gate.get("videos." + label) else "not_measured"
    d["platform_format"] = "not_measured"
    d["coding_source"] = "ai_agents_unvalidated" if automated else "model"
    d["model_version_hash"] = report["version"]
    d["stratum"] = d.month + "_" + d.duration_band
    for c in ["view_count", "like_count", "comment_count", "duration_seconds", "age_days", "likes_per_1000_views"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for target, source in [("log_views", "view_count"), ("log_duration", "duration_seconds"), ("log_age", "age_days")]:
        d[target] = np.log1p(d[source])
    return d, requirements


def regression_inputs(p, frame, gate):
    """Apply the configured rare-category policy to regression controls only."""
    threshold = p.cfg["analysis"].get("rare_category_n", 10)
    if type(threshold) is not int or threshold < 1:
        raise ValueError("analysis.rare_category_n 须为正整数")
    model = frame.copy()
    controls = "log_views + log_duration + log_age"
    mapping = {"minimum_category_videos": threshold,
               "scope": "known video labels in the frozen analysis frame; descriptive groups unchanged",
               "labels": {}}
    for label in data.VIDEO_FIELDS:
        if not gate.get("videos." + label):
            continue
        counts = model[label].value_counts()
        rare = counts[counts < threshold].index.tolist()
        model[label] = model[label].where(~model[label].isin(rare), "pooled_rare")
        included = model[label].nunique() > 1
        mapping["labels"][label] = {"counts_before_pooling": counts.to_dict(), "pooled_categories": rare,
                                    "missing_videos": int(model[label].isna().sum()), "included_control": included}
        if included:
            controls += f" + C({label})"
    return model, controls, mapping


def analyze_predictions(p, root, prediction, report, dest):
    study = data.manifest(root)
    automated = report["reference_type"] == "none_ai_only"
    gate = report["gate"]
    dest.mkdir(parents=True, exist_ok=True)
    for kind, values in prediction.items():
        write_csv(values, dest / f"predictions_{kind}.csv")
    event("统计", "开始汇总参与指标，检查完整分母和标签缺失。")
    d, requirements = analysis_frame(root, prediction, report)
    write_csv(d, dest / "analysis_videos.csv")
    coverage_rows = []
    for metric in requirements:
        for status, group in d.groupby(metric + "_status", dropna=False):
            coverage_rows.append({"metric": metric, "status": status, "videos": len(group),
                                  "median_views": group.view_count.median(),
                                  "median_machine_comments": group.machine_comment_rows.median()})
    write_csv(pd.DataFrame(coverage_rows), dest / "measurement_coverage.csv")
    write_csv(descriptive(d, list(requirements)), dest / "descriptive.csv")
    event("统计", f"已汇总 {len(d):,} 个视频；覆盖率与描述统计已保存。开始相关与 bootstrap。")
    correlations = []
    for i, y in enumerate(requirements, 1):
        result = boot_correlation(d, "view_count", y, p.cfg["analysis"]["bootstrap_reps"], p.cfg["study"]["seed"])
        correlations.append({"x": "view_count", "y": y, **result})
        event("统计", f"相关分析 {i}/{len(requirements)}：{y}；有效视频 {result['n']:,}。")
    write_csv(correct_p(pd.DataFrame(correlations), "fdr_bh"), dest / "correlations.csv")
    proxy = type("AnalysisContext", (), {"cfg": p.cfg, "reports": dest})()
    event("统计", "开始组间比较与回归；无法估计的项目会记录跳过原因。")
    comparisons(proxy, d)
    model_frame, controls, pooling = regression_inputs(p, d, gate)
    write_json(dest / "model_category_pooling.json", pooling)
    fit_models(p, model_frame, [("machine_quality_binomial", "n_cognitive_comments", controls, "binomial", "n_audience_comments"),
                     ("machine_conversion_binomial", "n_dialogue_roots", controls + " + root_cognitive_rate", "binomial", "n_known_roots")], dest)
    if not automated:
        figures(proxy, d, d if d.deep_rate.notna().any() else None)
    availability = [{"metric": name, "dependencies": labels,
                     "failed_validation_labels": None if automated else [c for c in labels if not gate.get("comments." + c, False)],
                     "observed_videos": int(d[name].notna().sum())} for name, labels in requirements.items()]
    notes = (["所有语义标签来自两个盲编码 AI 与分歧裁决 AI，无人工参考，不报告准确率/召回率或人工信度。",
              "AI 一致性使用裁决前结果；相同模型的不同提示词不构成独立测量有效性的证据。",
              "只有该指标可由相关受众标签确定时计算；null 不当作 0，不靠删除不确定评论改变分母。",
              "分析为模型标签的探索性关联；置信区间不包含模型误分类与裁决误差。"] if automated else
             ["单名人工参考不等于双人编码信度，标签筛选通过也不是效度保证。",
              "只有相关标签通过预设验证且指标可由已知标签确定时计算；确定的垃圾评论不影响质量指标。",
              "区间未传播机器误分类和人工参考误差，系数只反映条件关联，不识别因果机制。"])
    notes += ["原人工 coding/ 与 codes_* 表未改动，机器标签没有 reviewed=1 的人工声明。",
              "关闭评论、无受众评论、未通过验证及模型弃权不赋值为低质量零分。",
              "主题来自标题/说明，形式仅使用提供的字幕；不能宣称模型观看了视频。",
              "学习/重新考虑表达与分享意向只是文本指标，不等于实际认知改变、分享行为和传播扩散。"]
    notes += ["认知参与是 OR：任一认知标签为 1 即可确定为认知参与；没有正标签但有未知项则仍为未知。",
              "根评论认知比例仅依赖有效根评论；对话仅依赖同线程较晚的已知异账户受众回复。",
              "lower/upper 为未知标签取可能值所得的逻辑上下界，不是置信区间；分母不确定时仍缺失。",
              "measurement_coverage.csv 按缺失原因报告有效视频数、观看与评论规模，帮助识别选择性缺失。",
              "回归按 rare_category_n 合并稀有视频类别；映射见 model_category_pooling.json，描述与组间比较保留原类别。",
              "分组二项模型以成功数/分母判断是否恒定；成功条数相同但分母不同仍可估计。"]
    notes += ["服务拒绝为不可编码，单列数量和来源；任一盲编码或裁决不可用时最终单元保留缺失，不充当模型弃权、成功或负标签。"]
    if report.get("workflow") == "screening":
        notes[:2] = ["语义标签来自单轮初编及重点复查＋非重点部分分层随机抽查；没有全量盲复核或裁决。",
                     "抽查模型看不到初编标签，但同模型仍可能共同出错；只报告分域检查一致性，不报告准确率或人工信度。"]
        notes[-1] = "初编或已选检查被服务拒绝时保留不可编码缺失；分歧或弃权字段保留 null，未抽中仍是未经独立检查的初编标签。"
    write_json(dest / "report.json", {"model_version": report["version"], "analysis_logic_version": 4,
        "coding_source": "ai_screening_unvalidated" if report.get("workflow") == "screening" else "ai_agents_unvalidated" if automated else "model",
        "comment_window": study["window"], "comment_cutoff": study["comment_cutoff"],
        "videos": len(d), "comments": study["n_comments"], "metrics": availability,
        "threads_per_video": study["threads_per_video"], "analysis_configuration": p.cfg["analysis"],
        "analysis_seed": p.cfg["study"]["seed"],
        "validation_reference": report["reference_type"], "validation_gate": None if automated else gate,
        "measurement_policy": "resolved_ai_only_exploratory" if automated else "human_reference_gated",
        "scope": "conditional_associations_in_observed_machine_coded_sample",
        "notes": notes})
    (dest / "ANALYSIS_NOTES.md").write_text(
        "# AI 研究分析\n\n" + "\n\n".join(notes) +
        "\n\n请查看本目录 report.json、models_status.csv；常量/缺失/样本不足会自动跳过模型。\n",
        encoding="utf-8")
    event("统计", f"主分析已保存：{dest}；包含测量口径、缺失与模型状态。")

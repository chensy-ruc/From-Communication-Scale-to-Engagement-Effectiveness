from __future__ import annotations

import itertools
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .core import load_sample, read_csv, stable_hash, write_csv, write_json


def safe_rate(n, d):
    return n / d if d else np.nan


def participation(p, first30=False):
    comments = p.store.frame("clean_comments")
    codes = p.store.frame("codes_comments")
    units = read_csv(p.root / "coding/comments/units.csv")
    frame = comments.merge(codes, on="comment_id", validate="one_to_one")
    frame = frame.merge(units[["comment_id", "thread_probability"]], on="comment_id", validate="one_to_one")
    if first30:
        frame = frame[frame.within_first30.eq(1)].copy()
    threads = []
    for (vid, root), g in frame.groupby(["video_id", "root_id"], sort=True):
        rr = g[g.comment_id.eq(root)]
        if rr.empty:
            continue
        root_row = rr.iloc[0]
        users = g[g.spam.eq(0) & g.is_channel_author.eq(0)].copy()
        if root_row.spam == 1 or root_row.is_channel_author == 1:
            continue
        later = users[users.is_reply.eq(1) & (pd.to_datetime(users.published_at, utc=True) >= pd.Timestamp(root_row.published_at))]
        root_author = root_row.author_hash
        different = later[later.author_hash.notna() & later.author_hash.ne(root_author)] if pd.notna(root_author) else later.iloc[:0]
        substantive = different[different.substantive_response.eq(1)]
        author_counts = users.author_hash.dropna().value_counts()
        sustained = (len(users) >= 3 and len(author_counts) >= 2 and author_counts.max() >= 2
                     and ((users.cognitive.eq(1) | users.substantive_response.eq(1)).sum() >= 2)
                     and len(substantive) >= 1)
        threads.append({"video_id": vid, "root_id": root, "root_cognitive": int(root_row.cognitive),
                        "cross_user_reply": int(len(different) > 0), "dialogue_conversion": int(len(substantive) > 0),
                        "sustained_dialogue": int(sustained), "root_author_missing": int(pd.isna(root_author)),
                        "thread_probability": float(root_row.thread_probability)})
    thread_df = pd.DataFrame(threads, columns=["video_id", "root_id", "root_cognitive", "cross_user_reply", "dialogue_conversion",
                                               "sustained_dialogue", "root_author_missing", "thread_probability"])
    rows = []
    for vid, g in frame.groupby("video_id", sort=True):
        audience = g[g.spam.eq(0) & g.is_channel_author.eq(0)]
        t = thread_df[thread_df.video_id.eq(vid)]
        # Missing author IDs cannot establish cross-user conversion: remove from that denominator.
        valid = t[t.root_author_missing.eq(0)]
        rows.append({"video_id": vid, "coded_comments": len(g), "n_audience_comments": len(audience),
                     "n_cognitive_comments": int(audience.cognitive.sum()),
                     "deep_rate": safe_rate(audience.cognitive.sum(), len(audience)),
                     "mean_quality": audience.quality_level.mean(),
                     "reply_rate": audience.is_reply.mean(), "unique_known_authors": audience.author_hash.nunique(),
                     "missing_author_comments": int(audience.author_hash.isna().sum()),
                     "n_roots": len(t), "n_valid_roots": len(valid),
                     "n_dialogue_roots": int(valid.dialogue_conversion.sum()),
                     "n_root_cognitive": int(valid.root_cognitive.sum()),
                     "root_cognitive_rate": safe_rate(valid.root_cognitive.sum(), len(valid)),
                     "dialogue_conversion_rate": safe_rate(valid.dialogue_conversion.sum(), len(valid)),
                     "cross_user_reply_rate": safe_rate(valid.cross_user_reply.sum(), len(valid)),
                     "sustained_dialogue_rate": safe_rate(valid.sustained_dialogue.sum(), len(valid)),
                     "learning_expression_rate": audience.learning_or_reconsideration.mean(),
                     "sharing_intent_rate": audience.sharing_intent.mean(),
                     "n_spam": int(g.spam.sum())})
    return pd.DataFrame(rows), thread_df, frame


def descriptive(df, columns):
    out = []
    for c in columns:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        out.append({"variable": c, "n": len(x), "missing": len(df) - len(x), "mean": x.mean(),
                    "sd": x.std(), "median": x.median(), "q25": x.quantile(.25), "q75": x.quantile(.75),
                    "min": x.min(), "max": x.max(), "zeros": int(x.eq(0).sum())})
    return pd.DataFrame(out)


def comment_descriptives(p, census):
    """Describe observed text and reply structure without inventing semantic codes."""
    comments = p.store.frame("clean_comments")
    states = p.store.frame("collection")[["video_id", "status"]]
    base = census[["video_id", "month"]].merge(states, on="video_id", how="left", validate="one_to_one")
    base["status"] = base.status.fillna("not_started")
    summaries, distributions, languages, structures = [], [], [], []
    scopes = [("collected_all", comments), ("analysis_window", comments[comments.eligible_window.eq(1)])]
    for scope, frame in scopes:
        audience = frame[frame.is_channel_author.eq(0)]
        summaries.append({
            "scope": scope, "comments": len(frame), "videos_with_observed_comments": frame.video_id.nunique(),
            "top_level_comments": int(frame.is_reply.eq(0).sum()), "replies": int(frame.is_reply.eq(1).sum()),
            "reply_share": safe_rate(frame.is_reply.eq(1).sum(), len(frame)),
            "channel_comments": int(frame.is_channel_author.eq(1).sum()), "audience_comments": len(audience),
            "unique_known_audience_accounts": audience.author_hash.nunique(),
            "missing_author_comments": int(frame.author_hash.isna().sum()),
            "empty_text_comments": int(frame.empty_text.eq(1).sum()),
            "url_flagged_comments": int(frame.has_url.eq(1).sum()),
            "same_author_repeat_flagged_comments": int(frame.repeat_by_same_author.eq(1).sum()),
            "within_2025_comments": int(frame.within_year.eq(1).sum()),
            "within_first30_comments": int(frame.within_first30.eq(1).sum()),
        })
        description = descriptive(frame, ["like_count", "char_count"])
        description.insert(0, "scope", scope)
        distributions.append(description)
        for lang, n in frame.language.fillna("und").value_counts().items():
            languages.append({"scope": scope, "language": lang, "comments": int(n), "share": n / len(frame)})
        counts = frame.groupby("video_id").agg(
            observed_comments=("comment_id", "size"), observed_replies=("is_reply", "sum"))
        structure = base.merge(counts, on="video_id", how="left", validate="one_to_one")
        for column in ["observed_comments", "observed_replies"]:
            structure[column] = pd.to_numeric(structure[column], errors="raise").fillna(0).astype("int64")
        structure["observed_top_level"] = structure.observed_comments - structure.observed_replies
        complete = structure.status.eq("complete")
        # Preserve observed counts even for partial data, but never treat missing coverage as a genuine zero.
        structure["complete_comments"] = structure.observed_comments.where(complete)
        structure["complete_replies"] = structure.observed_replies.where(complete)
        denominator = structure.complete_comments.where(structure.complete_comments.gt(0))
        structure["reply_share"] = structure.complete_replies / denominator
        structure.insert(0, "scope", scope)
        structures.append(structure)
    write_csv(pd.DataFrame(summaries), p.reports / "comment_summary.csv")
    write_csv(pd.concat(distributions, ignore_index=True), p.reports / "descriptive_comments.csv")
    write_csv(pd.DataFrame(languages, columns=["scope", "language", "comments", "share"]),
              p.reports / "comment_language_summary.csv")
    write_csv(pd.concat(structures, ignore_index=True), p.reports / "video_comment_structure.csv")
    (p.reports / "COMMENT_DESCRIPTIVE_NOTES.md").write_text(
        "# 评论描述统计口径\n\n"
        "collected_all 包括所有清洗后采集记录；analysis_window 仅含 eligible_window=1 的记录。\n"
        "comment_summary.csv 的回复占比按评论条数计算；video_comment_structure.csv 按视频列出。\n"
        "observed_* 是实际取得的记录数。complete_* 仅在采集状态 complete 时提供；关闭评论或采集不完整为缺失。\n"
        "完整采集但没有评论的视频计数为 0，回复占比仍为缺失。\n"
        "unique_known_audience_accounts 是已知非频道账户数，不是经验证的独立人数。\n"
        "语言为自动识别结果，und 含过短或识别不确定的文本；不能推断国籍或居住地。\n"
        "重复、链接和空文本仅为清洗标记，不自动等同垃圾信息。\n"
        "这些表尚未排除人工判定的垃圾评论，回复结构不代表实质对话、参与质量或传播效能。\n",
        encoding="utf-8",
    )


def boot_correlation(d, x, y, reps, seed):
    d = d.dropna(subset=[x, y]).reset_index(drop=True)
    if len(d) < 5 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return {"n": len(d), "rho": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan, "bootstrap_valid": 0}
    rho, pv = stats.spearmanr(d[x], d[y])
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in d.groupby("stratum", sort=True)] if "stratum" in d else [d.index.to_numpy()]
    boot = []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        xx, yy = d.loc[idx, x], d.loc[idx, y]
        if xx.nunique() >= 2 and yy.nunique() >= 2:
            boot.append(stats.spearmanr(xx, yy).statistic)
    ci = np.quantile(boot, [.025, .975]) if len(boot) >= max(20, reps / 2) else [np.nan, np.nan]
    return {"n": len(d), "rho": rho, "p": pv, "ci_low": ci[0], "ci_high": ci[1], "bootstrap_valid": len(boot)}


def correct_p(df, method):
    df = df.copy()
    df["p_adjusted"] = np.nan
    if len(df):
        mask = pd.to_numeric(df.p, errors="coerce").notna()
        if mask.any():
            df.loc[mask, "p_adjusted"] = multipletests(df.loc[mask, "p"].astype(float), method=method)[1]
    df["adjustment"] = method
    return df


def comparisons(p, df):
    omnibus, pairwise, descriptives = [], [], []
    rng = np.random.default_rng(p.cfg["study"]["seed"])
    outcomes = ["view_count", "likes_per_1000_views", "deep_rate", "dialogue_conversion_rate"]
    for factor in ["topic", "form", "platform_format"]:
        for outcome in outcomes:
            groups = []
            for name, g in df.groupby(factor, sort=True):
                x = g[outcome].dropna().to_numpy(dtype=float)
                descriptives.append({"factor": factor, "group": name, "outcome": outcome, "n": len(x),
                                     "median": np.median(x) if len(x) else np.nan,
                                     "mean": np.mean(x) if len(x) else np.nan})
                if len(x) >= p.cfg["analysis"]["min_group_n"]:
                    groups.append((name, x))
            n, k = sum(len(x) for _, x in groups), len(groups)
            if k < 2:
                omnibus.append({"factor": factor, "outcome": outcome, "n": n, "k": k,
                                 "H": np.nan, "p": np.nan, "epsilon_squared": np.nan, "status": "insufficient_groups"})
                continue
            combined = np.concatenate([x for _, x in groups])
            if np.unique(combined).size < 2:
                h, pv = 0.0, 1.0
            else:
                h, pv = stats.kruskal(*[x for _, x in groups])
            omnibus.append({"factor": factor, "outcome": outcome, "n": n, "k": k,
                             "H": h, "p": pv, "epsilon_squared": max(0, (h - k + 1) / (n - k)), "status": "ok"})
            for (a, x), (b, y) in itertools.combinations(groups, 2):
                u, pv = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
                med = [np.median(rng.choice(x, len(x), replace=True)) - np.median(rng.choice(y, len(y), replace=True))
                       for _ in range(p.cfg["analysis"]["bootstrap_reps"])]
                lo, hi = np.quantile(med, [.025, .975])
                pairwise.append({"factor": factor, "outcome": outcome, "a": a, "b": b,
                                 "n_a": len(x), "n_b": len(y), "U": u, "p": pv,
                                 "rank_biserial_a_minus_b": 2 * u / (len(x) * len(y)) - 1,
                                 "median_difference": np.median(x) - np.median(y), "ci_low": lo, "ci_high": hi})
    write_csv(correct_p(pd.DataFrame(omnibus), "fdr_bh"), p.reports / "groups_kruskal.csv")
    write_csv(correct_p(pd.DataFrame(pairwise, columns=["factor", "outcome", "a", "b", "n_a", "n_b", "U", "p",
                                                       "rank_biserial_a_minus_b", "median_difference", "ci_low", "ci_high"]), "holm"),
              p.reports / "groups_pairwise.csv")
    write_csv(pd.DataFrame(descriptives), p.reports / "groups_descriptive.csv")


def regressions(p, frame):
    df = frame.copy()
    df["log_views"] = np.log1p(df.view_count)
    df["log_duration"] = np.log1p(df.duration_seconds)
    df["log_age"] = np.log1p(df.age_days)
    mappings = {}
    for col in ["topic", "form", "platform_format"]:
        counts = df[col].value_counts()
        rare = counts[counts < p.cfg["analysis"]["rare_category_n"]].index.tolist()
        df[col] = df[col].where(~df[col].isin(rare), "pooled_rare")
        mappings[col] = rare
    write_json(p.reports / "model_category_pooling.json", mappings)
    categorical = [f"C({c})" for c in ["topic", "form", "platform_format"] if df[c].nunique() > 1]
    controls = " + ".join(["log_duration", "log_age"] + categorical)
    specifications = [
        ("scale_ols", "log_views", controls, "ols"),
        ("comment_count_nb2", "comment_count", "log_views + " + controls, "nb2"),
        ("quality_binomial", "n_cognitive_comments", "log_views + " + controls, "quality"),
        ("conversion_binomial", "n_dialogue_roots", "root_cognitive_rate + log_views + " + controls, "conversion"),
    ]
    statuses, coefficients = [], []
    for name, outcome, rhs, kind in specifications:
        status = {"model": name, "formula": outcome + " ~ " + rhs, "status": "not_fit",
                  "inference_scope": "conditional_on_sample_and_observed_comments; no causal identification"}
        try:
            need = [outcome, "log_views", "log_duration", "log_age", "topic", "form", "platform_format"]
            if kind in {"quality", "conversion"}:
                denominator = "n_audience_comments" if kind == "quality" else "n_valid_roots"
                need += [denominator]
                if kind == "conversion":
                    need += ["root_cognitive_rate"]
            d = df.dropna(subset=need).copy()
            if kind in {"quality", "conversion"}:
                d = d[d[denominator] > 0]
            X = patsy.dmatrix(rhs, d, return_type="dataframe")
            n, k = X.shape
            status.update(n=n, parameters=k, dropped=len(df) - n)
            if n < max(p.cfg["analysis"]["min_regression_n"], 5 * k):
                raise ValueError(f"insufficient_n: n={n}, parameters={k}")
            if np.linalg.matrix_rank(X) < k:
                raise ValueError("rank_deficient_design")
            if d[outcome].nunique() < 2:
                raise ValueError("constant_outcome")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if kind == "ols":
                    fit = sm.OLS(d[outcome].astype(float), X).fit(cov_type="HC3")
                    status["r_squared"] = fit.rsquared
                elif kind == "nb2":
                    fit = sm.NegativeBinomial(d[outcome].astype(float), X).fit(disp=False, maxiter=300, cov_type="HC0")
                    status["converged"] = bool(fit.mle_retvals.get("converged", False))
                    if not status["converged"]:
                        raise ValueError("nb2_not_converged")
                else:
                    success = d[outcome].astype(float)
                    failure = d[denominator].astype(float) - success
                    if min(success.sum(), failure.sum()) < 5 * k:
                        raise ValueError("too_few_events_or_non_events")
                    fit = sm.GLM(np.column_stack([success, failure]), X, family=sm.families.Binomial()).fit(cov_type="HC1")
                    status["converged"] = bool(fit.converged)
                    status["pearson_dispersion"] = fit.pearson_chi2 / fit.df_resid
                    if not fit.converged:
                        raise ValueError("binomial_not_converged")
                status["warnings"] = " | ".join(sorted({str(w.message) for w in caught}))
                if any("separation" in str(w.message).lower() for w in caught):
                    raise ValueError("perfect_separation_detected")
            if not np.isfinite(fit.bse).all():
                raise ValueError("nonfinite_standard_errors")
            conf = fit.conf_int()
            for term in fit.params.index:
                coefficient = fit.params[term]
                lo, hi = conf.loc[term]
                exponentiate = kind != "ols" and term != "alpha"
                coefficients.append({"model": name, "term": term, "coef": coefficient, "se": fit.bse[term],
                                     "p": fit.pvalues[term], "ci_low": lo, "ci_high": hi,
                                     "exp_coef": np.exp(coefficient) if exponentiate and abs(coefficient) < 700 else np.nan,
                                     "exp_ci_low": np.exp(lo) if exponentiate and abs(lo) < 700 else np.nan,
                                     "exp_ci_high": np.exp(hi) if exponentiate and abs(hi) < 700 else np.nan,
                                     "effect_type": "IRR" if kind == "nb2" else "OR" if exponentiate else "coefficient"})
            (p.reports / f"{name}.txt").write_text(str(fit.summary()), encoding="utf-8")
            status["status"] = "ok"
            status["aic"] = fit.aic
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as e:
            status["status"] = "skipped"
            status["reason"] = str(e)
        statuses.append(status)
    write_csv(pd.DataFrame(statuses), p.reports / "models_status.csv")
    write_csv(pd.DataFrame(coefficients, columns=["model", "term", "coef", "se", "p", "ci_low", "ci_high", "exp_coef",
                                                 "exp_ci_low", "exp_ci_high", "effect_type"]), p.reports / "models_coefficients.csv")


def figures(p, census, sample=None):
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log1p(census.view_count.dropna()), bins=35, color="#285c73", edgecolor="white")
    ax.set(xlabel="log(1 + cumulative views at collection)", ylabel="Videos", title="2025 public video frame")
    fig.tight_layout()
    fig.savefig(p.reports / "views_distribution.png")
    fig.savefig(p.reports / "views_distribution.pdf")
    plt.close(fig)
    if sample is not None and "deep_rate" in sample:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, y, title in zip(axes, ["deep_rate", "dialogue_conversion_rate"], ["Cognitive participation", "Observed dialogue conversion"]):
            d = sample.dropna(subset=["view_count", y])
            ax.scatter(np.log1p(d.view_count), d[y], s=18, alpha=.55, color="#285c73")
            ax.set(xlabel="log(1 + views)", ylabel=y, title=title, ylim=(-.03, 1.03))
        fig.tight_layout()
        fig.savefig(p.reports / "scale_participation.png")
        fig.savefig(p.reports / "scale_participation.pdf")
        plt.close(fig)


def verify_final(p, kind):
    info = p.store.meta(f"codes_{kind}_finalized")
    if not info:
        raise ValueError(f"请先完成 {kind} 的独立编码、reliability 和 finalize-coding")
    if info["window"] != p.cfg["study"]["comment_window"]:
        raise ValueError("编码窗口与当前分析配置不同")
    path = p.root / f"coding/{kind}/adjudicated.csv"
    if stable_hash(path.read_bytes()) != info["adjudicated_hash"]:
        raise ValueError(f"{kind} 裁决表已修改，请重新 finalize-coding")


def analyze(p, descriptive_only=False):
    census = p.store.frame("clean_videos")
    basic_cols = ["view_count", "like_count", "comment_count", "duration_seconds", "age_days",
                  "likes_per_1000_views", "comments_per_1000_views"]
    write_csv(descriptive(census, basic_cols), p.reports / "descriptive_video_census.csv")
    write_csv(census.groupby("month").agg(n_videos=("video_id", "size"), total_views=("view_count", "sum"),
                                          median_views=("view_count", "median")).reset_index(), p.reports / "monthly_summary.csv")
    comment_descriptives(p, census)
    if descriptive_only:
        figures(p, census)
        print("已生成视频与评论的描述统计；尚未进行人工编码相关的推断分析。")
        return
    for kind in ["videos", "comments"]:
        verify_final(p, kind)
    design = load_sample(p)
    df = census.merge(design, on="video_id", validate="one_to_one").merge(p.store.frame("codes_videos"), on="video_id", validate="one_to_one")
    part, threads, comments = participation(p)
    if part.empty:
        raise ValueError("没有有效受众编码，无法计算参与质量")
    df = df.merge(part, on="video_id", how="left", validate="one_to_one")
    df = df.merge(p.store.frame("collection")[["video_id", "status"]], on="video_id", how="left", validate="one_to_one")
    df["video_weight"] = pd.to_numeric(df.video_weight)
    p.store.save_frame("analysis_videos", df)
    p.store.save_frame("analysis_threads", threads)
    write_csv(df, p.reports / "analysis_videos.csv")
    write_csv(threads, p.reports / "analysis_threads.csv")
    extra = ["deep_rate", "mean_quality", "dialogue_conversion_rate", "sustained_dialogue_rate", "root_cognitive_rate"]
    write_csv(descriptive(df, basic_cols + extra), p.reports / "descriptive_analysis_sample.csv")
    weighted = []
    for col in basic_cols + extra:
        d = df.dropna(subset=[col, "video_weight"])
        weighted.append({"variable": col, "n_observed_videos": len(d),
                         "weighted_mean_video_level": np.average(d[col], weights=d.video_weight) if len(d) else np.nan,
                         "scope": "design-weighted mean among observed videos; missingness not corrected"})
    write_csv(pd.DataFrame(weighted), p.reports / "weighted_video_descriptive.csv")
    correlations = []
    for x in ["view_count", "likes_per_1000_views", "comments_per_1000_views"]:
        for y in ["deep_rate", "dialogue_conversion_rate", "sustained_dialogue_rate"]:
            correlations.append({"x": x, "y": y, **boot_correlation(df, x, y, p.cfg["analysis"]["bootstrap_reps"], p.cfg["study"]["seed"])})
    write_csv(correct_p(pd.DataFrame(correlations), "fdr_bh"), p.reports / "correlations_spearman.csv")
    comparisons(p, df)
    regressions(p, df)
    figures(p, census, df)
    sensitivity, sensitive_threads, _ = participation(p, first30=True)
    write_csv(sensitivity, p.reports / "sensitivity_first30_video_metrics.csv")
    write_csv(sensitive_threads, p.reports / "sensitivity_first30_threads.csv")
    sensitivity_correlations = []
    sets = [("at_least_10_audience_comments", df[df.n_audience_comments.ge(10)]),
            ("without_shorts_or_unknown", df[~df.platform_format.isin(["shorts", "unknown"])])]
    if not sensitivity.empty:
        sets.append(("first30_comments_current_views", df.drop(columns=part.columns.difference(["video_id"])).merge(sensitivity, on="video_id", how="inner")))
    for name, d in sets:
        for y in ["deep_rate", "dialogue_conversion_rate"]:
            sensitivity_correlations.append({"sensitivity": name, "x": "view_count", "y": y,
                **boot_correlation(d, "view_count", y, p.cfg["analysis"]["bootstrap_reps"], p.cfg["study"]["seed"])})
    write_csv(correct_p(pd.DataFrame(sensitivity_correlations), "fdr_bh"), p.reports / "sensitivity_correlations.csv")
    reliability = []
    for kind in ["videos", "comments"]:
        r = pd.read_csv(p.root / f"coding/{kind}/reliability.csv")
        reliability.extend({"kind": kind, "variable": row.variable, "alpha_nominal": row.alpha_nominal,
                            "n_pairs": row.n_pairs} for row in r.itertuples())
    preliminary = any(pd.isna(r["alpha_nominal"]) or r["alpha_nominal"] < .8 or r["n_pairs"] < 50 for r in reliability)
    report = {"video_frame": len(census), "selected_videos": len(design), "coded_videos": len(df),
              "videos_with_participation": int(df.deep_rate.notna().sum()), "coded_comment_rows": len(comments),
              "valid_threads": len(threads), "quality_status": "requires_reliability_review" if preliminary else "reliability_screen_passed",
              "reliability": reliability, "collection_status": df.status.value_counts(dropna=False).to_dict(),
              "scope": "描述统计提供设计权重；相关、组间检验和回归解释实际分析样本中的条件关联，不作全年总体设计推断。",
              "limits": ["横截面观察研究不能识别因果机制或中介因果效应。", "Views/Likes/Comments 为当前累计值，非 2025 年末值。",
                         "对话转化是线程内可观察代理；分享意愿、学习表述不等于实际扩散或认知改变。",
                         "不同视频的评论观察时长不同；控制视频年龄并报告 30 天评论窗口敏感性。",
                         "30 天评论窗口也只含当前仍可见评论，Views 仍是当前值；不构成历史 30 天曝光测量。",
                         "无评论/关闭评论/采集失败不赋值为参与质量零；匿名缺失不推断用户身份。",
                         "不依据摘要预先指定显著方向；预定示例须以实际 ID 和采集快照核验。"]}
    write_json(p.reports / "analysis_report.json", report)
    notes = ["# 分析输出说明", "", f"视频框 {len(census)}；抽样 {len(design)}；有参与指标的视频 {df.deep_rate.notna().sum()}。",
             "", f"信度状态：{report['quality_status']}。低信度、稀有类别与常量变量均须逐项审查，不能用裁决后一致率代替独立信度。",
             "", report["scope"], "", *[f"- {x}" for x in report["limits"]], "",
             "先检查 collection_coverage.csv、models_status.csv、两个 reliability.csv，再解释系数与显著性。",
             "模型系数的区间为逐项区间；回归 p 值未作跨模型多重校正，应按预注册主要假设解释。"]
    (p.reports / "ANALYSIS_NOTES.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"统计结果已输出至 {p.reports}；请先看 ANALYSIS_NOTES.md 与 models_status.csv。")

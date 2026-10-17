"""All-video metadata and observable reply structure, independent of semantic coding."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats

from .analysis import boot_correlation, correct_p, descriptive
from .core import write_csv, write_json


def fit_models(p, frame, specs, dest):
    statuses, coefficients = [], []
    for name, outcome, rhs, family, denominator in specs:
        record = {"model": name, "outcome": outcome, "rhs": rhs, "family": family, "status": "skipped"}
        try:
            need = [outcome] + ([denominator] if denominator else [])
            d = frame.dropna(subset=need).copy()
            if denominator:
                d = d[d[denominator].gt(0)]
            X = patsy.dmatrix(rhs, d, return_type="dataframe")
            d = d.loc[X.index]
            n, k = X.shape
            record.update(n=n, parameters=k, dropped=len(frame) - n)
            if n < max(p.cfg["analysis"]["min_regression_n"], 5 * k):
                raise ValueError("insufficient_complete_videos")
            if np.linalg.matrix_rank(X) != k or not np.isfinite(X).all().all():
                raise ValueError("rank_deficient_or_nonfinite_design")
            # Grouped binomial variation is in success / trials, not the success count alone.
            # One positive in 5 comments differs from one positive in 100 comments.
            if family != "binomial" and d[outcome].nunique() < 2:
                raise ValueError("constant_outcome")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if family == "ols":
                    fit = sm.OLS(d[outcome].astype(float), X).fit(cov_type="HC3")
                elif family == "nb2":
                    fit = sm.NegativeBinomial(d[outcome].astype(float), X).fit(disp=False, maxiter=300, cov_type="HC0")
                    if not fit.mle_retvals.get("converged", False):
                        raise ValueError("not_converged")
                else:
                    y = d[outcome].astype(float)
                    trials = d[denominator].astype(float)
                    failure = trials - y
                    if (not np.isfinite(y).all() or not np.isfinite(trials).all() or (y < 0).any()
                            or (failure < 0).any() or y.mod(1).ne(0).any() or trials.mod(1).ne(0).any()):
                        raise ValueError("invalid_binomial_counts")
                    if (y / trials).nunique() < 2:
                        raise ValueError("constant_binomial_proportion")
                    if min(y.sum(), failure.sum()) < 5 * k:
                        raise ValueError("insufficient_or_invalid_events")
                    fit = sm.GLM(np.column_stack([y, failure]), X, family=sm.families.Binomial()).fit(cov_type="HC1")
                    if not fit.converged:
                        raise ValueError("not_converged")
                    record["pearson_dispersion"] = fit.pearson_chi2 / fit.df_resid
                if any("separation" in str(w.message).lower() for w in caught):
                    raise ValueError("perfect_separation")
                record["warnings"] = " | ".join(sorted({str(w.message) for w in caught}))
            if not np.isfinite(fit.bse).all() or not np.isfinite(fit.params).all():
                raise ValueError("nonfinite_estimate")
            for term in fit.params.index:
                lo, hi = fit.conf_int().loc[term]
                coefficients.append({"model": name, "term": term, "coef": fit.params[term], "se": fit.bse[term],
                                     "p": fit.pvalues[term], "ci_low": lo, "ci_high": hi,
                                     "effect_type": "dispersion_alpha" if family == "nb2" and term == "alpha" else
                                                    "log_IRR" if family == "nb2" else "log_OR" if family == "binomial" else "coefficient"})
            record.update(status="ok", aic=fit.aic)
            (dest / f"{name}.txt").write_text(str(fit.summary()), encoding="utf-8")
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError, patsy.PatsyError) as exc:
            record["reason"] = str(exc)
            (dest / f"{name}.txt").write_text(
                f"本次模型未估计：{name}\n原因：{exc}\n请查看 models_status.csv。\n", encoding="utf-8")
        statuses.append(record)
    write_csv(pd.DataFrame(statuses), dest / "models_status.csv")
    write_csv(pd.DataFrame(coefficients, columns=["model", "term", "coef", "se", "p", "ci_low", "ci_high", "effect_type"]),
              dest / "models_coefficients.csv")


def thread_structure(comments):
    records = []
    for (vid, rid), g in comments.groupby(["video_id", "root_id"], sort=True):
        root = g[g.comment_id.eq(rid)]
        if root.empty or root.iloc[0].is_channel_author == 1:
            continue
        root = root.iloc[0]
        audience = g[g.is_channel_author.eq(0)]
        replies = audience[audience.is_reply.eq(1) & (pd.to_datetime(audience.published_at, utc=True, format="ISO8601") >= pd.Timestamp(root.published_at))]
        root_known = pd.notna(root.author_hash) and bool(root.author_hash)
        known = replies[replies.author_hash.notna() & replies.author_hash.ne("")]
        different = known[known.author_hash.ne(root.author_hash)] if root_known else known.iloc[:0]
        authors = audience.loc[audience.author_hash.notna() & audience.author_hash.ne(""), "author_hash"].value_counts()
        lag = ((pd.to_datetime(replies.published_at, utc=True, format="ISO8601").min() - pd.Timestamp(root.published_at)).total_seconds() / 3600
               if len(replies) else np.nan)
        records.append({"video_id": vid, "root_id": rid, "root_author_known": int(root_known),
                        "audience_comments": len(audience), "audience_replies": len(replies),
                        "has_audience_reply": int(len(replies) > 0), "cross_account_reply": int(len(different) > 0) if root_known else np.nan,
                        "repeat_account_participation": int(len(audience) >= 3 and len(authors) >= 2 and authors.max() >= 2),
                        "first_audience_reply_hours": lag})
    return pd.DataFrame(records, columns=["video_id", "root_id", "root_author_known", "audience_comments", "audience_replies",
                                         "has_audience_reply", "cross_account_reply", "repeat_account_participation", "first_audience_reply_hours"])


def frame(p):
    videos = p.store.frame("clean_videos")
    comments = p.store.frame("clean_comments")
    comments = comments[comments.eligible_window.eq(1)]
    threads = thread_structure(comments)
    grouped = threads.groupby("video_id").agg(n_roots=("root_id", "size"), n_known_roots=("root_author_known", "sum"),
        n_replied_roots=("has_audience_reply", "sum"), n_cross_account_roots=("cross_account_reply", "sum"),
        n_repeat_account_roots=("repeat_account_participation", "sum"), median_first_reply_hours=("first_audience_reply_hours", "median"))
    d = videos.merge(p.store.frame("collection")[["video_id", "status"]], on="video_id", how="left")
    d = d.merge(grouped, on="video_id", how="left")
    d["observed_comments"] = d.video_id.map(comments.groupby("video_id").size()).fillna(0)
    cols = ["n_roots", "n_known_roots", "n_replied_roots", "n_cross_account_roots", "n_repeat_account_roots", "observed_comments"]
    for col in cols:
        d[col] = pd.to_numeric(d[col], errors="raise").fillna(0).where(d.status.eq("complete"))
    for name, numerator, denominator in [("reply_thread_rate", "n_replied_roots", "n_roots"),
        ("cross_account_reply_rate", "n_cross_account_roots", "n_known_roots"),
        ("repeat_account_rate", "n_repeat_account_roots", "n_roots")]:
        d[name] = d[numerator] / d[denominator].where(d[denominator].gt(0))
    d["median_first_reply_hours"] = d.median_first_reply_hours.where(d.status.eq("complete"))
    d["stratum"] = d.month + "_" + d.duration_band
    for target, source in [("log_views", "view_count"), ("log_duration", "duration_seconds"), ("log_age", "age_days")]:
        d[target] = np.log1p(d[source])
    return d, threads


def analyze(p):
    dest = p.reports / "structure"
    dest.mkdir(parents=True, exist_ok=True)
    d, threads = frame(p)
    write_csv(d, dest / "videos.csv")
    write_csv(threads, dest / "threads.csv")
    write_csv(descriptive(d, ["observed_comments", "reply_thread_rate", "cross_account_reply_rate", "repeat_account_rate",
                              "median_first_reply_hours"]), dest / "descriptive.csv")
    correlations = []
    for x, y in [("view_count", "like_count"), ("view_count", "observed_comments"),
                 ("view_count", "reply_thread_rate"), ("view_count", "cross_account_reply_rate"),
                 ("view_count", "repeat_account_rate")]:
        correlations.append({"x": x, "y": y, **boot_correlation(d, x, y, p.cfg["analysis"]["bootstrap_reps"], p.cfg["study"]["seed"])})
    write_csv(correct_p(pd.DataFrame(correlations), "fdr_bh"), dest / "correlations.csv")
    comparisons = []
    for factor in ["month", "duration_band"]:
        for outcome in ["view_count", "observed_comments", "cross_account_reply_rate"]:
            groups = [g[outcome].dropna().to_numpy() for _, g in d.groupby(factor)]
            groups = [g for g in groups if len(g) >= p.cfg["analysis"]["min_group_n"]]
            n, k = sum(map(len, groups)), len(groups)
            h, pv = (np.nan, np.nan)
            if k >= 2:
                h, pv = stats.kruskal(*groups) if np.unique(np.concatenate(groups)).size > 1 else (0, 1)
            comparisons.append({"factor": factor, "outcome": outcome, "n": n, "k": k, "H": h, "p": pv,
                                "epsilon_squared": max(0, (h - k + 1) / (n - k)) if n > k > 1 else np.nan})
    write_csv(correct_p(pd.DataFrame(comparisons), "fdr_bh"), dest / "groups.csv")
    controls = "log_views + log_duration + log_age"
    fit_models(p, d, [("comment_count_nb2", "observed_comments", controls, "nb2", None),
                     ("cross_account_binomial", "n_cross_account_roots", controls, "binomial", "n_known_roots")], dest)
    write_json(dest / "report.json", {"videos": len(d), "threads": len(threads), "semantic_coding": False,
        "collection_status": d.status.value_counts(dropna=False).to_dict(), "scope": "current_observed_video_frame_associations",
        "limits": ["同线程异账户回复不是精确的用户对用户回复边，也不是实质对话或跨文化理解。",
                   "重复发言是结构指标，不是持续认知参与。未剔除语义判定的垃圾评论。",
                   "无评论或根作者缺失时相关比例为空；不可获取视频计数为空。",
                   "时长分组不是 Shorts；月份分组不是内容主题。",
                   "这些是观察性条件关联；bootstrap 与回归区间不覆盖平台不可见数据误差。"]})
    print(f"无需语义编码的全量结构分析已输出：{dest}")

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .core import load_sample, now, read_csv, stable_hash, write_csv, write_json

VIDEO_LABELS = {
    "topic": ["china_governance", "economy_technology", "international_relations", "culture_society", "people_exchange", "other"],
    "form": ["news_report", "interview_dialogue", "documentary", "experience_vlog", "speech_event", "other"],
    "platform_format": ["shorts", "standard", "live_replay", "unknown"],
}
COMMENT_LABELS = ["information_request", "fact_check", "opinion", "reasoned_opinion", "cross_cultural",
                  "emotion", "substantive_response", "learning_or_reconsideration", "sharing_intent", "spam"]
COGNITIVE = ["information_request", "fact_check", "reasoned_opinion", "cross_cultural"]


def labels_for(kind):
    return list(VIDEO_LABELS) if kind == "videos" else COMMENT_LABELS


def id_for(kind):
    return "video_id" if kind == "videos" else "comment_id"


def quality(df):
    cognitive = df[COGNITIVE].astype(int).max(axis=1).eq(1)
    substantive = df.substantive_response.astype(int).eq(1)
    return pd.Series(np.select([cognitive & substantive, cognitive, df.opinion.astype(int).eq(1)],
                              [3, 2, 1], default=0), index=df.index)


def prepare(p, kind, threads_per_video, reliability_fraction, allow_incomplete=False):
    if not 0 < reliability_fraction <= 1:
        raise ValueError("reliability_fraction 必须在 (0,1] 内")
    if threads_per_video < 0:
        raise ValueError("threads_per_video=0 表示全部线程，其余为正数")
    folder = p.root / "coding" / kind
    if (folder / "manifest.json").exists():
        raise ValueError("编码样本已冻结，不能覆盖已有人工劳动。新设计/窗口请新建快照")
    sample = load_sample(p)
    videos = p.store.frame("clean_videos").merge(sample, on="video_id", validate="one_to_one")
    rng = np.random.default_rng(p.cfg["study"]["seed"] + (1 if kind == "videos" else 2))
    folder.mkdir(parents=True, exist_ok=True)
    if kind == "videos":
        units = videos[["video_id", "stratum", "video_probability", "video_weight"]].copy()
        blind = videos[["video_id", "title", "description", "url", "duration_seconds"]].copy()
        b_ids = []
        for _, g in units.groupby("stratum", sort=True):
            k = min(len(g), max(1, math.ceil(len(g) * reliability_fraction)))
            b_ids.extend(rng.choice(g.video_id, k, replace=False))
    else:
        states = sample.merge(p.store.frame("collection"), on="video_id", how="left")
        unresolved = states[~states.status.isin(["complete", "commentsDisabled", "videoNotFound", "forbidden"])]
        if len(unresolved) and not allow_incomplete:
            raise ValueError(f"{len(unresolved)} 个选中视频未完整采集（含 reply_gap）。请续传；探索性分析可显式 --allow-incomplete")
        complete_ids = states.loc[states.status.eq("complete"), "video_id"]
        all_comments = p.store.frame("clean_comments")
        comments = all_comments[all_comments.video_id.isin(complete_ids) & all_comments.eligible_window.eq(1)].copy()
        # Channel-authored roots are not audience participation units. Keep channel replies as context.
        roots = comments[comments.is_reply.eq(0) & comments.is_channel_author.eq(0)]
        selected = []
        for vid, g in roots.groupby("video_id", sort=True):
            g = g.sort_values("comment_id")
            k = len(g) if threads_per_video == 0 else min(threads_per_video, len(g))
            ids = rng.choice(g.comment_id, k, replace=False)
            for root_id in ids:
                selected.append({"video_id": vid, "root_id": root_id, "thread_probability": k / len(g),
                                 "threads_population": len(g), "threads_sampled": k})
        if not selected:
            raise ValueError("当前窗口没有可编码的受众顶层评论线程；无评论不应被编码成低质量")
        thread_sample = pd.DataFrame(selected).merge(sample, on="video_id", validate="many_to_one")
        chosen = comments.merge(thread_sample, on=["video_id", "root_id"], validate="many_to_one")
        chosen["comment_probability"] = pd.to_numeric(chosen.video_probability) * chosen.thread_probability
        chosen["comment_weight"] = 1 / chosen.comment_probability
        units = chosen[["comment_id", "video_id", "root_id", "stratum", "video_probability", "video_weight",
                        "thread_probability", "comment_probability", "comment_weight", "threads_population", "threads_sampled"]]
        write_csv(thread_sample, folder / "thread_sample.csv")
        blind = chosen[["comment_id", "video_id", "root_id", "is_reply", "is_channel_author", "text_clean", "published_at"]].copy()
        title_map = videos.set_index("video_id").title
        blind["video_title"] = blind.video_id.map(title_map)
        # One context file per selected thread; no like/view counts or automatic code suggestions.
        (folder / "context").mkdir(exist_ok=True)
        paths = {}
        for root_id, g in chosen.groupby("root_id", sort=True):
            rel = "context/" + stable_hash(root_id)[:24] + ".txt"
            lines = ["仅供语境阅读；编码仍须逐条独立判断。", str(title_map[g.video_id.iloc[0]])]
            for r in g.sort_values(["is_reply", "published_at", "comment_id"]).itertuples():
                author = str(r.author_hash)[:12] if pd.notna(r.author_hash) else "unknown"
                lines.append(f"\nID={r.comment_id} | author={author} | reply={r.is_reply} | channel={r.is_channel_author} | {r.published_at}\n{r.text_clean}")
            (folder / rel).write_text("\n".join(lines), encoding="utf-8")
            paths[root_id] = rel
        blind["context_file"] = blind.root_id.map(paths)
        b_roots = []
        for _, g in thread_sample.groupby("stratum", sort=True):
            k = min(len(g), max(1, math.ceil(len(g) * reliability_fraction)))
            b_roots.extend(rng.choice(g.root_id, k, replace=False))
        b_ids = units.loc[units.root_id.isin(b_roots), "comment_id"].tolist()
    identifier = id_for(kind)
    for col in labels_for(kind) + ["uncertain", "reviewed", "coder_id", "coding_note"]:
        blind[col] = ""
    write_csv(units, folder / "units.csv")
    a = blind.sample(frac=1, random_state=p.cfg["study"]["seed"] + 10)
    b = blind[blind[identifier].isin(b_ids)].sample(frac=1, random_state=p.cfg["study"]["seed"] + 20)
    write_csv(a, folder / "coder_A.csv")
    write_csv(b, folder / "coder_B.csv")
    manifest = {"kind": kind, "created_at": now(), "a_ids": sorted(a[identifier].tolist()),
                "b_ids": sorted(b[identifier].tolist()), "reliability_fraction": reliability_fraction,
                "window": p.cfg["study"]["comment_window"], "allow_incomplete": allow_incomplete,
                "threads_per_video": threads_per_video, "units_hash": stable_hash((folder / "units.csv").read_bytes())}
    write_json(folder / "manifest.json", manifest)
    print(f"{kind}：A {len(a)} 行，B 独立复编码 {len(b)} 行。编码前请阅读 data/inputs/codebook.txt。")


def validate_frame(df, kind, expected_ids, units=None, final=False):
    key = id_for(kind)
    required = [key] + labels_for(kind) + ["uncertain", "reviewed", "coder_id", "coding_note"]
    missing = set(required) - set(df)
    if missing:
        raise ValueError(f"缺少编码列：{sorted(missing)}")
    if df[key].duplicated().any() or set(df[key]) != set(expected_ids):
        raise ValueError("编码 ID 缺失、重复或超出冻结样本；不要删行或改 ID")
    for col in labels_for(kind) + ["uncertain", "reviewed"]:
        allowed = VIDEO_LABELS.get(col, ["0", "1"])
        if (~df[col].isin(allowed)).any():
            bad = df.loc[~df[col].isin(allowed), key].head(5).tolist()
            raise ValueError(f"{col} 含空值/非法编码，示例 ID {bad}；允许 {allowed}")
    if (df.reviewed != "1").any() or df.coder_id.str.strip().eq("").any():
        raise ValueError("每行须 reviewed=1，并填写真实 coder_id")
    if ((df.uncertain == "1") & df.coding_note.str.strip().eq("")).any():
        raise ValueError("uncertain=1 时必须填写 coding_note")
    if final and (df.uncertain != "0").any():
        raise ValueError("最终裁决必须解决 uncertain；无法判断的内容请在方法中另行说明，不能默认为 0")
    if kind == "comments":
        if ((df.reasoned_opinion == "1") & (df.opinion != "1")).any():
            raise ValueError("reasoned_opinion=1 时 opinion 必须为 1")
        if units is not None:
            lookup = units.set_index("comment_id")
            is_root = df[key].map(lookup.root_id).eq(df[key])
            if (is_root & df.substantive_response.eq("1")).any():
                raise ValueError("substantive_response 只用于回复评论，顶层评论须为 0")
    return df


def agreement_metrics(x, y, ordinal=False):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    if n < 2:
        return {"agreement": None, "kappa": None, "alpha_nominal": None}
    levels = sorted(set(x) | set(y))
    a = np.array([(x == z).sum() for z in levels], dtype=float)
    b = np.array([(y == z).sum() for z in levels], dtype=float)
    agree = float(np.mean(x == y))
    pe = float((a * b).sum() / n ** 2)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else None
    counts = a + b
    de = 1 - float((counts * (counts - 1)).sum() / (2 * n * (2 * n - 1)))
    alpha = 1 - (1 - agree) / de if de > 0 else None
    out = {"agreement": agree, "kappa": kappa, "alpha_nominal": alpha}
    if ordinal:
        # Fixed, equally spaced 0..3 ordinal levels; linear weighted Cohen's kappa.
        distance_observed = np.abs(x.astype(float) - y.astype(float)).mean()
        distance_expected = sum(a[i] * b[j] * abs(float(u) - float(v)) for i, u in enumerate(levels)
                                for j, v in enumerate(levels)) / n ** 2
        out["kappa_linear"] = 1 - distance_observed / distance_expected if distance_expected > 0 else None
    return out


def reliability(p, kind, bootstrap_reps=None):
    folder = p.root / "coding" / kind
    manifest = json.loads((folder / "manifest.json").read_text())
    units = read_csv(folder / "units.csv")
    key = id_for(kind)
    a = validate_frame(read_csv(folder / "coder_A.csv"), kind, manifest["a_ids"], units)
    b = validate_frame(read_csv(folder / "coder_B.csv"), kind, manifest["b_ids"], units)
    if set(a.coder_id) & set(b.coder_id):
        raise ValueError("A、B 的 coder_id 不得重叠；同一人/复制标签不能作为双人信度")
    matched = a.merge(b, on=key, suffixes=("_a", "_b"), validate="one_to_one")
    if len(matched) < 2:
        raise ValueError("至少需要 2 个复编码单元；正式研究建议远多于此")
    labels = labels_for(kind)
    if kind == "comments":
        for suffix in ["a", "b"]:
            view = matched.rename(columns={f"{c}_{suffix}": c for c in COMMENT_LABELS})
            matched[f"quality_level_{suffix}"] = quality(view).astype(str)
        labels = labels + ["quality_level"]
    clusters = matched[key] if kind == "videos" else matched[key].map(units.set_index(key).video_id)
    cluster_indices = [np.flatnonzero(clusters.to_numpy() == c) for c in sorted(clusters.unique())]
    reps = bootstrap_reps or p.cfg["analysis"]["bootstrap_reps"]
    results = []
    for label in labels:
        x, y = matched[f"{label}_a"].to_numpy(), matched[f"{label}_b"].to_numpy()
        metrics = agreement_metrics(x, y, ordinal=label == "quality_level")
        boot = []
        rng = np.random.default_rng(p.cfg["study"]["seed"])
        if len(cluster_indices) >= 2:
            for _ in range(reps):
                idx = np.concatenate([cluster_indices[i] for i in rng.integers(len(cluster_indices), size=len(cluster_indices))])
                value = agreement_metrics(x[idx], y[idx])["kappa"]
                if value is not None:
                    boot.append(value)
        ci = np.quantile(boot, [.025, .975]) if len(boot) >= max(20, reps * .5) else [None, None]
        results.append({"variable": label, "n_pairs": len(x), "n_clusters": len(cluster_indices), **metrics,
                        "kappa_ci_low": ci[0], "kappa_ci_high": ci[1], "bootstrap_valid": len(boot),
                        "coder_a_distribution": json.dumps(pd.Series(x).value_counts().to_dict()),
                        "coder_b_distribution": json.dumps(pd.Series(y).value_counts().to_dict()),
                        "note": "常量编码使机会校正信度不可识别" if metrics["kappa"] is None else ""})
    out = pd.DataFrame(results)
    write_csv(out, folder / "reliability.csv")
    disagreements = []
    for label in labels_for(kind) + ["uncertain"]:
        for _, row in matched[matched[f"{label}_a"] != matched[f"{label}_b"]].iterrows():
            disagreements.append({key: row[key], "variable": label, "coder_a": row[f"{label}_a"], "coder_b": row[f"{label}_b"]})
    write_csv(pd.DataFrame(disagreements, columns=[key, "variable", "coder_a", "coder_b"]), folder / "disagreements.csv")
    final_path = folder / "adjudicated.csv"
    if final_path.exists():
        raise ValueError("信度报告已更新，但不会覆盖已有 adjudicated.csv。需保留旧裁决并人工同步修改")
    final = a.copy()
    bm = b.set_index(key)
    for i, row in final.iterrows():
        if row[key] in bm.index:
            for label in labels_for(kind) + ["uncertain"]:
                if row[label] != bm.loc[row[key], label]:
                    final.loc[i, label] = ""
    write_csv(final, final_path)
    write_json(folder / "reliability_run.json", {"created_at": now(),
               "a_hash": stable_hash((folder / "coder_A.csv").read_bytes()),
               "b_hash": stable_hash((folder / "coder_B.csv").read_bytes()), "pairs": len(matched)})
    print(f"信度报告与裁决表：{folder}；{len(disagreements)} 个分歧，须人工裁决。")


def finalize(p, kind):
    folder = p.root / "coding" / kind
    manifest = json.loads((folder / "manifest.json").read_text())
    run = json.loads((folder / "reliability_run.json").read_text())
    for coder in ["a", "b"]:
        if run[f"{coder}_hash"] != stable_hash((folder / f"coder_{coder.upper()}.csv").read_bytes()):
            raise ValueError("信度检验后原始编码表已改动；请归档旧裁决表后重跑 reliability")
    units = read_csv(folder / "units.csv")
    if manifest["units_hash"] != stable_hash((folder / "units.csv").read_bytes()):
        raise ValueError("冻结的编码单元表被改动")
    if manifest["window"] != p.cfg["study"]["comment_window"]:
        raise ValueError("当前评论窗口与编码窗口不一致，请恢复配置或新建快照")
    final = validate_frame(read_csv(folder / "adjudicated.csv"), kind, manifest["a_ids"], units, final=True)
    key = id_for(kind)
    disputed = read_csv(folder / "disagreements.csv")
    if not disputed.empty and final.loc[final[key].isin(disputed[key]), "coding_note"].str.strip().eq("").any():
        raise ValueError("有分歧的单元须在裁决表 coding_note 中记录裁决依据和裁决者")
    save = final[[key] + labels_for(kind) + ["coder_id", "coding_note"]].copy()
    if kind == "comments":
        save[COMMENT_LABELS] = save[COMMENT_LABELS].astype(int)
        save["quality_level"] = quality(save)
        save["cognitive"] = save[COGNITIVE].max(axis=1)
    p.store.save_frame(f"codes_{kind}", save)
    p.store.set_meta(f"codes_{kind}_finalized", {"at": now(), "n": len(save), "window": manifest["window"],
                                                "adjudicated_hash": stable_hash((folder / "adjudicated.csv").read_bytes())})
    print(f"已导入 {len(save)} 个 {kind} 最终编码。")

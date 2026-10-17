"""Held-out model-versus-human evaluation; never described as two human coders."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import ai_client as client
from . import ai_data as data
from .coding import COMMENT_LABELS, VIDEO_LABELS, agreement_metrics
from .core import now, stable_hash, write_csv, write_json


def fingerprint(prediction):
    return stable_hash(json.dumps({k: v.sort_values("id").to_json(orient="records", force_ascii=False)
                                  for k, v in prediction.items()}, sort_keys=True))


def binary_metrics(truth, predicted, positive):
    positive_true, positive_predicted = truth == positive, predicted == positive
    tp = int((positive_true & positive_predicted).sum())
    fp = int((~positive_true & positive_predicted).sum())
    fn = int((positive_true & ~positive_predicted).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    return {"tp": tp, "fp": fp, "fn": fn, "reference_positives": int(positive_true.sum()),
            "reference_negatives": int((~positive_true).sum()), "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan}


def evaluate(p, experiment, split):
    root = data.folder(p, experiment)
    config = client.settings(p, root)
    ver = client.version(config)
    th = client.frozen(root, config)["thresholds"] if split == "validation" else client.thresholds(p)
    prediction = client.predictions(root, ver, split)
    dest = p.reports / "ai" / experiment / f"evaluation_{split}"
    dest.mkdir(parents=True, exist_ok=True)
    metrics, confusion, discrepancies, gate = [], [], [], {}
    reference_counts = {}
    for kind, labels in [("videos", data.VIDEO_FIELDS), ("comments", COMMENT_LABELS)]:
        ref = data.reference(root, split, kind)
        key = "video_id" if kind == "videos" else "comment_id"
        pred = prediction[kind].rename(columns={"id": key})
        paired = ref.merge(pred, on=key, suffixes=("_human", "_model"), validate="one_to_one")
        if len(paired) != len(ref):
            raise ValueError("AI 结果未覆盖全部人工验证 ID")
        reference_counts[kind] = {"rows": len(ref), "coders": sorted(ref.coder_id.unique().tolist()),
                                  "uncertain_rows": int(ref.uncertain.eq("1").sum())}
        for label in labels:
            valid = paired.uncertain.eq("0") & ~paired[f"{label}_human"].isin(["", "unknown"])
            d = paired[valid].copy()
            truth = d[f"{label}_human"].astype(str).to_numpy()
            raw = d[f"{label}_model"]
            if kind == "comments":
                predicted = raw.map(lambda x: "__abstain__" if pd.isna(x) else str(int(x))).to_numpy()
            else:
                predicted = raw.fillna("__abstain__").astype(str).to_numpy()
            groups = [np.flatnonzero(d.video_id.to_numpy() == v) for v in sorted(d.video_id.unique())]
            coverage = float(np.mean(predicted != "__abstain__")) if len(d) else 0.0
            agree = agreement_metrics(truth, predicted)
            classes = VIDEO_LABELS[label] if kind == "videos" else ["1"]
            label_passes = []
            for positive in classes:
                row = {"kind": kind, "label": label, "positive_class": positive, "n_reference": len(d),
                       "n_video_clusters": len(groups), "excluded_uncertain_reference": len(paired) - len(d),
                       "prediction_coverage": coverage, "accuracy_including_abstentions": float(np.mean(truth == predicted)) if len(d) else np.nan,
                       "kappa_model_vs_human": agree["kappa"], **binary_metrics(truth, predicted, positive)}
                boot = []
                rng = np.random.default_rng(p.cfg["study"]["seed"])
                if len(groups) >= 2:
                    for _ in range(p.cfg["analysis"]["bootstrap_reps"]):
                        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
                        b = binary_metrics(truth[idx], predicted[idx], positive)
                        boot.append([b[c] for c in ["precision", "recall", "f1"]])
                for j, name in enumerate(["precision", "recall", "f1"]):
                    values = np.array([b[j] for b in boot if np.isfinite(b[j])])
                    lo, hi = np.quantile(values, [.025, .975]) if len(values) >= max(20, len(boot) / 2) else [np.nan, np.nan]
                    row[f"{name}_ci_low"], row[f"{name}_ci_high"] = lo, hi
                reasons = []
                for name, threshold in [("reference_positives", th["min_reference_positives"]),
                                        ("reference_negatives", th["min_reference_negatives"]),
                                        ("precision", th["min_precision"]), ("recall", th["min_recall"]),
                                        ("prediction_coverage", th["min_prediction_coverage"])]:
                    if not np.isfinite(row[name]) or row[name] < threshold:
                        reasons.append(name)
                row["screen_passed"] = not reasons
                row["screen_reason"] = ",".join(reasons)
                label_passes.append(not reasons)
                metrics.append(row)
            gate[f"{kind}.{label}"] = all(label_passes)
            pairs = pd.DataFrame({"human": truth, "model": predicted})
            for (human, model), n in pairs.groupby(["human", "model"]).size().items():
                confusion.append({"kind": kind, "label": label, "human": human, "model": model, "n": int(n)})
            for idx, human, model in zip(d[key], truth, predicted):
                if human != model:
                    discrepancies.append({"kind": kind, "id": idx, "label": label, "human": human, "model": model})
    write_csv(pd.DataFrame(metrics), dest / "metrics.csv")
    write_csv(pd.DataFrame(confusion, columns=["kind", "label", "human", "model", "n"]), dest / "confusion.csv")
    write_csv(pd.DataFrame(discrepancies, columns=["kind", "id", "label", "human", "model"]), dest / "disagreements.csv")
    report = {"created_at": now(), "version": ver, "split": split, "gate": gate, "thresholds": th,
              "reference_type": "single_human_blind_reference; not_two_human_intercoder_reliability",
              "reference_counts": reference_counts, "reference_hashes": data.reference_hashes(root, split),
              "prediction_hash": fingerprint(prediction), "scope": "observed_stratified_validation_sample",
              "notes": ["筛选标准是本项目预先固定的规则，不是测量有效性的通用证明。",
                        "null 为弃权：计入覆盖率和召回率损失，不当作 0；不确定人工参考单列排除。",
                        "区间按视频簇重抽；未纳入平台不可见数据及人工参考判断误差。",
                        "每类单独检验；视频主题/形式须全部预定义类别通过才作为主要分组变量。",
                        "验证集看过结果后不得继续调参并宣称同一验证集独立；应另建实验使用新验证视频。"]}
    write_json(root / f"evaluation_{split}.json", report)
    write_json(dest / "report.json", report)
    print(f"{split} 模型对人工参考评价完成：{sum(gate.values())}/{len(gate)} 个标签通过预设筛选；见 {dest}。")


def checked(p, experiment):
    root = data.folder(p, experiment)
    cfg = client.settings(p, root)
    client.frozen(root, cfg)
    path = root / "evaluation_validation.json"
    if not path.exists():
        raise ValueError("先完成独立验证集的 ai-evaluate")
    report = json.loads(path.read_text())
    pred = client.predictions(root, client.version(cfg), "validation")
    if report["version"] != client.version(cfg) or report["reference_hashes"] != data.reference_hashes(root, "validation"):
        raise ValueError("验证报告与当前模型/人工参考不一致")
    if report["prediction_hash"] != fingerprint(pred):
        raise ValueError("验证结果已改变，请核查后重新评价")
    return report

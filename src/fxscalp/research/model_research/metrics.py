"""Classification / probabilistic metrics and the paired block bootstrap (predictive performance only - no PnL).

Classes: 0 SHORT, 1 NO_TRADE, 2 LONG. Confusion matrices have TRUE classes on rows and PREDICTED on columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np

K = 3
EPS = 1e-15


def confusion(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    y, yhat = np.asarray(y, dtype="int64"), np.asarray(yhat, dtype="int64")
    return np.bincount(y * K + yhat, minlength=K * K).reshape(K, K)


def from_confusion(cm: np.ndarray) -> dict[str, Any]:
    cm = np.asarray(cm, dtype="float64")
    tp = np.diag(cm)
    pred, true = cm.sum(axis=0), cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred > 0, tp / pred, 0.0)
        rec = np.where(true > 0, tp / true, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    n = cm.sum()
    return {"n": int(n), "accuracy": float(tp.sum() / n) if n else float("nan"), "balanced_accuracy": float(rec.mean()),
            "macro_f1": float(f1.mean()), "precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(),
            "pred_freq": (pred / n).tolist() if n else [float("nan")] * K, "class_dist": (true / n).tolist() if n else [float("nan")] * K,
            "confusion": cm.astype(int).tolist()}


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), np.asarray(y, dtype="int64")])))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    oh = np.eye(K)[np.asarray(y, dtype="int64")]
    return float(np.mean(np.sum((np.asarray(p, dtype="float64") - oh) ** 2, axis=1)))


def ece_top(y: np.ndarray, p: np.ndarray, bins: int = 15) -> tuple[float, list[dict[str, float]]]:
    """Top-label expected calibration error with equal-mass bins, plus the reliability table."""
    p = np.asarray(p, dtype="float64")
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    correct = (pred == np.asarray(y)).astype("float64")
    order = np.argsort(conf, kind="stable")
    parts = np.array_split(order, bins)
    ece, rows = 0.0, []
    for idx in parts:
        if len(idx) == 0:
            continue
        c, a = float(conf[idx].mean()), float(correct[idx].mean())
        ece += len(idx) / len(p) * abs(c - a)
        rows.append({"n": int(len(idx)), "mean_confidence": round(c, 4), "accuracy": round(a, 4)})
    return float(ece), rows


def class_reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, list[dict[str, float]]]:
    """One-vs-rest reliability per class: mean predicted probability vs observed frequency (equal-mass bins)."""
    out = {}
    for k, name in enumerate(("SHORT", "NO_TRADE", "LONG")):
        pk, yk = np.asarray(p)[:, k], (np.asarray(y) == k).astype("float64")
        rows = []
        for idx in np.array_split(np.argsort(pk, kind="stable"), bins):
            if len(idx):
                rows.append({"n": int(len(idx)), "mean_pred": round(float(pk[idx].mean()), 4), "observed": round(float(yk[idx].mean()), 4)})
        out[name] = rows
    return out


def full_metrics(y: np.ndarray, yhat: np.ndarray, p: np.ndarray | None) -> dict[str, Any]:
    m = from_confusion(confusion(y, yhat))
    if p is not None:
        m["log_loss"], m["brier"] = log_loss(y, p), brier(y, p)
        m["ece_top"], m["reliability_top"] = ece_top(y, p)
        m["reliability_by_class"] = class_reliability(y, p)
    return m


def selectivity(y: np.ndarray, p: np.ndarray, thresholds: tuple[float, ...]) -> list[dict[str, float]]:
    """Pre-registered descriptive study: restrict to rows whose top probability >= tau (coverage vs accuracy / macro-F1 among accepted)."""
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    rows = []
    for t in thresholds:
        m = conf >= t
        if m.sum() < 30:
            rows.append({"tau": t, "coverage": float(m.mean()), "n": int(m.sum())})
            continue
        s = from_confusion(confusion(y[m], pred[m]))
        rows.append({"tau": t, "coverage": float(m.mean()), "n": int(m.sum()), "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                     "pred_freq": s["pred_freq"]})
    return rows


# ------------------------------------------------------------------------------------------ block bootstrap
def block_counts(ts_ms: np.ndarray, y: np.ndarray, yhat: np.ndarray, p: np.ndarray | None, block_s: int) -> np.ndarray:
    """Per time-block sufficient statistics: 9 confusion counts + sum of -log p(y) + n  -> shape (n_blocks, 11)."""
    blk = (np.asarray(ts_ms, dtype="int64") // (block_s * 1000))
    ub, inv = np.unique(blk, return_inverse=True)
    out = np.zeros((len(ub), K * K + 2))
    np.add.at(out, (inv, np.asarray(y) * K + np.asarray(yhat)), 1.0)
    if p is not None:
        ll = -np.log(np.clip(np.asarray(p)[np.arange(len(y)), np.asarray(y)], EPS, 1.0))
        np.add.at(out[:, K * K], inv, ll)
    np.add.at(out[:, K * K + 1], inv, 1.0)
    return out


def _stats(sums: np.ndarray) -> dict[str, np.ndarray]:
    cm = sums[:, : K * K].reshape(-1, K, K)
    tp = np.einsum("bii->bi", cm)
    pred, true = cm.sum(axis=1), cm.sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred > 0, tp / pred, 0.0)
        rec = np.where(true > 0, tp / true, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return {"macro_f1": f1.mean(axis=1), "balanced_accuracy": rec.mean(axis=1), "log_loss": sums[:, K * K] / sums[:, K * K + 1]}


def paired_block_bootstrap(model_blocks: list[np.ndarray], base_blocks: list[np.ndarray], *, B: int, seed: int,
                           chunk: int = 100) -> dict[str, Any]:
    """Resample blocks WITHIN each fold (same indices for model and baseline: paired), pool the folds, return metric replicates.
    ``*_blocks[f]`` are the per-fold arrays from ``block_counts`` (identical block layout for model and baseline)."""
    rng = np.random.default_rng(seed)
    reps = {"model": {"macro_f1": [], "balanced_accuracy": [], "log_loss": []}, "base": {"macro_f1": [], "balanced_accuracy": [], "log_loss": []}}
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        sm = np.zeros((b, K * K + 2))
        sb = np.zeros((b, K * K + 2))
        for mb, bb in zip(model_blocks, base_blocks):
            idx = rng.integers(0, len(mb), (b, len(mb)))
            sm += mb[idx].sum(axis=1)
            sb += bb[idx].sum(axis=1)
        for key, arr in (("model", sm), ("base", sb)):
            st = _stats(arr)
            for k in reps[key]:
                reps[key][k].append(st[k])
    out: dict[str, Any] = {}
    for k in ("macro_f1", "balanced_accuracy", "log_loss"):
        m, b_ = np.concatenate(reps["model"][k]), np.concatenate(reps["base"][k])
        d = m - b_
        out[k] = {"model": _ci(m), "base": _ci(b_), "diff": _ci(d)}
    return out


def _ci(a: np.ndarray) -> dict[str, float]:
    return {"mean": float(a.mean()), "lo95": float(np.percentile(a, 2.5)), "hi95": float(np.percentile(a, 97.5))}


def pooled_point(block_list: list[np.ndarray]) -> dict[str, float]:
    s = sum(b.sum(axis=0) for b in block_list)[None, :]
    st = _stats(s)
    return {k: float(v[0]) for k, v in st.items()}


def label_autocorrelation(y: np.ndarray, step_s: int, max_lag_s: int = 1800) -> list[dict[str, float]]:
    """Autocorrelation of the (binary) NO_TRADE indicator along time, a dependence-time check for the block length."""
    x = (np.asarray(y) == 1).astype("float64")
    x = x - x.mean()
    v = float((x * x).mean())
    out = []
    for lag in range(1, max_lag_s // step_s + 1):
        out.append({"lag_s": lag * step_s, "acf": float((x[:-lag] * x[lag:]).mean() / v) if v > 0 else float("nan")})
    return out

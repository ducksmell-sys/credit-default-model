"""Credit-model metrics used in validation: ranking power, calibration, stability and bootstrap helpers."""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def ks_statistic(y, p):
    """Maximum gap between the cumulative score distributions of defaulters and non-defaulters."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    y_sorted = y[order]
    cum_bad = np.cumsum(y_sorted) / y_sorted.sum()
    cum_good = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def summary_metrics(y, p):
    auc = roc_auc_score(y, p)
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": ks_statistic(y, p),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p))}


def calibration_slope_intercept(y, p):
    """Regress outcomes on logit(p). Slope 1 and intercept 0 means perfectly calibrated."""
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def decile_table(y, p, n_groups=10):
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["group"] = pd.qcut(df["p"].rank(method="first"), n_groups, labels=False) + 1
    out = df.groupby("group").agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return out.reset_index()


def hosmer_lemeshow(y, p, n_groups=10):
    """Hosmer-Lemeshow chi-squared test on decile groups (very sensitive when n is large)."""
    t = decile_table(y, p, n_groups)
    n, pred, obs = t["n"].values, t["predicted"].values, t["observed"].values
    stat = float(np.sum(n * (obs - pred) ** 2 / (pred * (1 - pred))))
    return stat, float(1 - chi2.cdf(stat, n_groups - 2))


def psi(expected, actual, n_bins=10, eps=1e-4):
    """Population stability index. Bins come from the expected (training) sample."""
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    uniq = np.unique(expected)
    if len(uniq) <= n_bins:
        edges = np.concatenate([[-np.inf], (uniq[:-1] + uniq[1:]) / 2, [np.inf]])
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def bootstrap_ci(metric, y, p, n_boot=500, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(metric(y[idx], p[idx]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_auc_difference(y, p_a, p_b, n_boot=500, seed=0, alpha=0.05):
    """AUC(a) - AUC(b) with a paired bootstrap interval (same resamples for both models)."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    point = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    return float(point), float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))

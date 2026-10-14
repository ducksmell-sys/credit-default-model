"""Independent validation of the credit default model, in the structure of a model validation report.

Candidate model: logistic regression PD model. Challenger: gradient boosting. Benchmark: last month's status only.

Acceptance criteria (TOL) are conventional rules of thumb. They were chosen before this full script was
written, but after a smoke run had shown the headline test AUCs; they were not adjusted to those numbers.

Usage:
    python validate_credit_model.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve

from credit_data import FEATURE_GROUPS, PROTECTED, SEED, build_features, data_quality, load_raw, split_data
from credit_metrics import (
    bootstrap_ci, calibration_slope_intercept, decile_table, hosmer_lemeshow, ks_statistic,
    paired_auc_difference, psi, summary_metrics,
)
from credit_models import BENCHMARK_FEATURES, fit_benchmark, fit_gradient_boosting, fit_logistic, predict_pd, standardized_coefficients

TOL = {
    "auc_test_min": 0.72,          # discrimination of the candidate on the held-out test set
    "ks_test_min": 0.35,
    "overfit_gap_max": 0.03,       # train AUC minus test AUC
    "calib_mean_abs_max": 0.01,    # |mean predicted PD - observed default rate| on test
    "calib_slope_range": (0.90, 1.10),
    "psi_max": 0.10,               # train versus test, score and every feature
    "segment_auc_min": 0.65,       # for segments with at least 500 test cases
    "segment_min_n": 500,
    "sign_stability_min": 0.95,    # share of bootstrap refits keeping the sign of each top-5 coefficient
    "challenger_margin": 0.01,     # challenger must beat the candidate by this much AUC (CI excluding 0)
    "air_min": 0.80,               # adverse impact ratio screen on approval rates
    "noise_auc_drop_max": 0.01,    # AUC loss when amounts are perturbed by 10% noise
    "approve_share": 0.80,         # business cut-off: approve the 80% lowest-PD applicants
}

# Directional expectations from credit logic: more delay / utilization -> more default, more payment / limit -> less.
EXPECTED_SIGN = {
    "PAY1_LATE1": +1, "PAY1_LATE2": +1, "PAY1_LATE3P": +1, "MAX_DELAY": +1, "N_LATE_MONTHS": +1,
    "DELAY_2": +1, "DELAY_3": +1, "DELAY_4": +1, "DELAY_5": +1, "DELAY_6": +1,
    "UTIL_1": +1, "UTIL_AVG": +1, "PAY_RATIO_1": -1, "PAY_RATIO_AVG": -1, "LOG_PAYAMT_1": -1, "LOG_LIMIT": -1,
}

BLUE, RED, GRAY, ORANGE = "#4C72B0", "#C44E52", "#8C8C8C", "#DD8452"


def section(title):
    print(f"\n=== {title} ===")


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved chart to '{path}'")


def vif_table(F):
    """Variance inflation factor of each feature given all the others (above 10 is a common warning level)."""
    Z = (F - F.mean()) / F.std()
    out = {}
    for c in Z.columns:
        others = Z.drop(columns=c)
        r2 = LinearRegression().fit(others, Z[c]).score(others, Z[c])
        out[c] = 1 / (1 - r2) if r2 < 0.999999 else np.inf
    return pd.Series(out).sort_values(ascending=False)


def segment_definitions(raw_te, raw_tr, f_te):
    """Boolean masks over the test set. Quartile cut-offs come from the training data."""
    q = raw_tr["LIMIT_BAL"].quantile([0.25, 0.5, 0.75]).values
    util = f_te["UTIL_1"]
    return {
        "Limit: lowest quartile": raw_te["LIMIT_BAL"] <= q[0],
        "Limit: 2nd quartile": (raw_te["LIMIT_BAL"] > q[0]) & (raw_te["LIMIT_BAL"] <= q[1]),
        "Limit: 3rd quartile": (raw_te["LIMIT_BAL"] > q[1]) & (raw_te["LIMIT_BAL"] <= q[2]),
        "Limit: highest quartile": raw_te["LIMIT_BAL"] > q[2],
        "Education: graduate school": raw_te["EDUCATION"] == 1,
        "Education: university": raw_te["EDUCATION"] == 2,
        "Education: high school": raw_te["EDUCATION"] == 3,
        "Last month: not late (PAY_1 <= 0)": raw_te["PAY_1"] <= 0,
        "Last month: late (PAY_1 >= 1)": raw_te["PAY_1"] >= 1,
        "Utilization: below 30%": util < 0.3,
        "Utilization: 30% to 80%": (util >= 0.3) & (util < 0.8),
        "Utilization: 80% or more": util >= 0.8,
    }


def group_definitions(raw_te):
    age = raw_te["AGE"]
    return {
        "SEX": {"male": raw_te["SEX"] == 1, "female": raw_te["SEX"] == 2},
        "AGE": {"under 30": age < 30, "30 to 44": (age >= 30) & (age < 45), "45 and over": age >= 45},
        "MARRIAGE": {"married": raw_te["MARRIAGE"] == 1, "single": raw_te["MARRIAGE"] == 2,
                     "other / unknown": raw_te["MARRIAGE"].isin([0, 3])},
    }


def main():
    results = {}

    # ---------------------------------------------------------------- data
    X_raw, y = load_raw()
    section("1. Data quality")
    for k, v in data_quality(X_raw, y).items():
        print(f"  {k:<42} {v:>10,.4f}" if isinstance(v, float) else f"  {k:<42} {v:>10,}")

    F = build_features(X_raw)
    f_tr, f_va, f_te, y_tr, y_va, y_te = split_data(F, y)
    raw_tr, raw_te = X_raw.loc[f_tr.index], X_raw.loc[f_te.index]
    print(f"  split (train / validation / test): {len(f_tr):,} / {len(f_va):,} / {len(f_te):,}; "
          f"default rate {y_tr.mean():.4f} / {y_va.mean():.4f} / {y_te.mean():.4f}")
    print(f"  model features: {F.shape[1]}; excluded protected attributes: {', '.join(PROTECTED)}")

    lr, gb, bm = fit_logistic(f_tr, y_tr), fit_gradient_boosting(f_tr, y_tr), fit_benchmark(f_tr, y_tr)
    cols = {"Logistic (candidate)": list(F.columns), "Gradient boosting (challenger)": list(F.columns),
            "Benchmark (PAY_1 only)": BENCHMARK_FEATURES}
    models = {"Logistic (candidate)": lr, "Gradient boosting (challenger)": gb, "Benchmark (PAY_1 only)": bm}
    pd_te = {n: predict_pd(m, f_te[cols[n]]) for n, m in models.items()}
    pd_tr = {n: predict_pd(m, f_tr[cols[n]]) for n, m in models.items()}
    pd_va = {n: predict_pd(m, f_va[cols[n]]) for n, m in models.items()}
    cand, chal, bench = "Logistic (candidate)", "Gradient boosting (challenger)", "Benchmark (PAY_1 only)"

    # ---------------------------------------------------------------- conceptual soundness
    section("2. Conceptual soundness: coefficient signs (logistic, standardized inputs)")
    coefs = standardized_coefficients(lr, F.columns)
    wrong = []
    print(f"  {'feature':<16} {'coef':>8} {'odds ratio per 1 SD':>20}  expectation")
    for name, c in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        exp = EXPECTED_SIGN.get(name)
        tag = "" if exp is None else ("as expected" if np.sign(c) == exp else "SIGN MISMATCH")
        if tag == "SIGN MISMATCH":
            wrong.append(name)
        print(f"  {name:<16} {c:>+8.3f} {np.exp(c):>20.3f}  {tag}")
    n_checked = len(EXPECTED_SIGN)
    print(f"  signs checked: {n_checked}; mismatches: {len(wrong)} {wrong}")
    results["signs"] = len(wrong) == 0
    vif = vif_table(f_tr)
    print("  Variance inflation factors (largest eight; above 10 suggests collinearity):")
    print("  " + ", ".join(f"{k} {v:.1f}" for k, v in vif.head(8).items()))
    print(f"  features with VIF above 10: {int((vif > 10).sum())} of {len(vif)}")

    # ---------------------------------------------------------------- discrimination
    section("3. Discrimination (ranking power)")
    print(f"  {'model':<32} {'set':<11} {'AUC':>7} {'Gini':>7} {'KS':>7}")
    summ = {}
    for n in models:
        for label, yy, pp in [("train", y_tr, pd_tr[n]), ("validation", y_va, pd_va[n]), ("test", y_te, pd_te[n])]:
            s = summary_metrics(yy, pp)
            summ[(n, label)] = s
            print(f"  {n:<32} {label:<11} {s['auc']:>7.4f} {s['gini']:>7.4f} {s['ks']:>7.4f}")
    lo, hi = bootstrap_ci(lambda a, b: roc_auc_score(a, b), y_te, pd_te[cand], n_boot=500, seed=1)
    print(f"  candidate test AUC {summ[(cand, 'test')]['auc']:.4f}, 95% bootstrap interval [{lo:.4f}, {hi:.4f}]")
    gap = summ[(cand, "train")]["auc"] - summ[(cand, "test")]["auc"]
    gap_gb = summ[(chal, "train")]["auc"] - summ[(chal, "test")]["auc"]
    print(f"  overfitting gap (train AUC - test AUC): candidate {gap:+.4f}, challenger {gap_gb:+.4f}")
    results["auc"] = summ[(cand, "test")]["auc"] >= TOL["auc_test_min"]
    results["ks"] = summ[(cand, "test")]["ks"] >= TOL["ks_test_min"]
    results["overfit"] = gap <= TOL["overfit_gap_max"]

    # ---------------------------------------------------------------- challenger
    section("4. Challenger and benchmark comparison (test set, paired bootstrap)")
    d_gb, d_lo, d_hi = paired_auc_difference(y_te, pd_te[chal], pd_te[cand], n_boot=500, seed=2)
    d_bm, b_lo, b_hi = paired_auc_difference(y_te, pd_te[cand], pd_te[bench], n_boot=500, seed=3)
    print(f"  challenger minus candidate AUC: {d_gb:+.4f}  95% interval [{d_lo:+.4f}, {d_hi:+.4f}]")
    print(f"  candidate minus benchmark AUC:  {d_bm:+.4f}  95% interval [{b_lo:+.4f}, {b_hi:+.4f}]")
    chal_wins = d_gb >= TOL["challenger_margin"] and d_lo > 0
    print(f"  challenger clears the +{TOL['challenger_margin']:.2f} AUC margin with the interval above 0: {chal_wins}")
    results["challenger_not_better"] = not chal_wins

    # ---------------------------------------------------------------- calibration
    section("5. Calibration (test set)")
    print(f"  {'model':<32} {'mean PD':>8} {'observed':>9} {'slope':>7} {'intercept':>10} {'Brier':>7} {'HL stat':>8} {'HL p':>8}")
    calib = {}
    for n in (cand, chal):
        slope, icpt = calibration_slope_intercept(y_te, pd_te[n])
        hl, hl_p = hosmer_lemeshow(y_te, pd_te[n])
        calib[n] = (slope, icpt)
        print(f"  {n:<32} {pd_te[n].mean():>8.4f} {y_te.mean():>9.4f} {slope:>7.3f} {icpt:>+10.3f} "
              f"{summ[(n, 'test')]['brier']:>7.4f} {hl:>8.2f} {hl_p:>8.4f}")
    results["calib_mean"] = abs(pd_te[cand].mean() - y_te.mean()) <= TOL["calib_mean_abs_max"]
    results["calib_slope"] = TOL["calib_slope_range"][0] <= calib[cand][0] <= TOL["calib_slope_range"][1]
    dec = decile_table(y_te, pd_te[cand])
    print("  Candidate decile table (1 = lowest predicted PD):")
    print("  " + dec.assign(predicted=dec["predicted"].round(4), observed=dec["observed"].round(4)).to_string(index=False).replace("\n", "\n  "))

    # ---------------------------------------------------------------- stability
    section("6. Stability: PSI between training and test (candidate score and features)")
    psi_score = psi(pd_tr[cand], pd_te[cand])
    psi_feat = {c: psi(f_tr[c], f_te[c]) for c in F.columns}
    worst = max(psi_feat, key=psi_feat.get)
    print(f"  score PSI {psi_score:.5f}; largest feature PSI {psi_feat[worst]:.5f} ({worst}); threshold {TOL['psi_max']}")
    print("  Note: the split is random, so PSI is expected to be near zero. This test shows the monitoring")
    print("  machinery works, not that the model is stable over time (the data has no time dimension).")
    results["psi"] = max(psi_score, max(psi_feat.values())) < TOL["psi_max"]

    # ---------------------------------------------------------------- segments
    section("7. Performance by segment (test set, candidate)")
    segs = segment_definitions(raw_te, raw_tr, f_te)
    seg_rows = []
    print(f"  {'segment':<36} {'n':>6} {'observed':>9} {'mean PD':>8} {'gap':>7} {'AUC':>7}")
    for name, mask in segs.items():
        yy, pp = y_te[mask.values], pd_te[cand][mask.values]
        auc = roc_auc_score(yy, pp) if yy.nunique() == 2 else np.nan
        seg_rows.append((name, int(mask.sum()), yy.mean(), pp.mean(), auc))
        print(f"  {name:<36} {int(mask.sum()):>6} {yy.mean():>9.4f} {pp.mean():>8.4f} {pp.mean() - yy.mean():>+7.4f} {auc:>7.4f}")
    big = [r for r in seg_rows if r[1] >= TOL["segment_min_n"]]
    weak = [r[0] for r in big if r[4] < TOL["segment_auc_min"]]
    print(f"  segments below AUC {TOL['segment_auc_min']} (n >= {TOL['segment_min_n']}): {weak}")
    results["segments"] = len(weak) == 0

    # ---------------------------------------------------------------- coefficient stability
    section("8. Coefficient stability (200 bootstrap refits on the training data)")
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(200):
        idx = rng.integers(0, len(f_tr), len(f_tr))
        m = fit_logistic(f_tr.iloc[idx], y_tr.iloc[idx])
        boot.append([standardized_coefficients(m, F.columns)[c] for c in F.columns])
    boot = pd.DataFrame(boot, columns=F.columns)
    top5 = sorted(coefs, key=lambda k: -abs(coefs[k]))[:5]
    stab = {c: float((np.sign(boot[c]) == np.sign(coefs[c])).mean()) for c in top5}
    for c in top5:
        print(f"  {c:<16} full-sample {coefs[c]:>+7.3f}  bootstrap mean {boot[c].mean():>+7.3f}  std {boot[c].std():.3f}  "
              f"sign kept in {stab[c]:.1%} of refits")
    flips = [c for c in F.columns if 0.05 < (np.sign(boot[c]) == np.sign(coefs[c])).mean() < 0.95]
    print(f"  features whose sign is unstable (kept in 5%-95% of refits): {flips}")
    results["sign_stability"] = min(stab.values()) >= TOL["sign_stability_min"]

    # ---------------------------------------------------------------- sensitivity
    section("9. Sensitivity: feature-group ablation and input noise")
    base_auc = summ[(cand, "test")]["auc"]
    ablation = {}
    print(f"  {'group removed':<42} {'test AUC':>9} {'change':>8}")
    for g, gcols in FEATURE_GROUPS.items():
        keep = [c for c in F.columns if c not in gcols]
        m = fit_logistic(f_tr[keep], y_tr)
        a = roc_auc_score(y_te, predict_pd(m, f_te[keep]))
        ablation[g] = a
        print(f"  {g:<42} {a:>9.4f} {a - base_auc:>+8.4f}")
    amount_cols = ["LIMIT_BAL"] + [f"BILL_AMT{k}" for k in range(1, 7)] + [f"PAY_AMT{k}" for k in range(1, 7)]
    noise_rng = np.random.default_rng(7)
    noisy_aucs = []
    for _ in range(20):
        noisy = raw_te.copy()
        for c in amount_cols:
            noisy[c] = noisy[c] * np.exp(noise_rng.normal(0, 0.10, len(noisy)))
        noisy_aucs.append(roc_auc_score(y_te, predict_pd(lr, build_features(noisy))))
    drop = base_auc - float(np.mean(noisy_aucs))
    print(f"  10% multiplicative noise on limit, bills and payments (20 draws): mean AUC {np.mean(noisy_aucs):.4f}, drop {drop:+.4f}")
    results["noise"] = drop <= TOL["noise_auc_drop_max"]

    # ---------------------------------------------------------------- fairness
    section("10. Fairness screen (protected attributes are not model inputs)")
    cutoff = np.quantile(pd_va[cand], TOL["approve_share"])
    approve = pd_te[cand] <= cutoff
    print(f"  cut-off: approve if PD <= {cutoff:.4f} (the {TOL['approve_share']:.0%} quantile on validation); "
          f"overall test approval rate {approve.mean():.1%}")
    print(f"  {'attribute / group':<26} {'n':>6} {'approval':>9} {'AIR':>6} {'observed':>9} {'mean PD':>8} {'cal z':>6} {'AUC':>7}")
    fair_rows, air_fail, cal_flags = [], [], []
    for attr, groups in group_definitions(raw_te).items():
        rates = {g: approve[m.values].mean() for g, m in groups.items()}
        ref = max(rates.values())
        for g, m in groups.items():
            yy, pp = y_te[m.values], pd_te[cand][m.values]
            air = rates[g] / ref
            z = (yy.mean() - pp.mean()) / np.sqrt(pp.mean() * (1 - pp.mean()) / len(yy))
            fair_rows.append((attr, g, int(m.sum()), rates[g], air))
            if air < TOL["air_min"]:
                air_fail.append(f"{attr}:{g}")
            if abs(z) >= 2 and len(yy) >= 500:
                cal_flags.append(f"{attr}:{g} (observed - predicted = {yy.mean() - pp.mean():+.3f}, z = {z:+.1f})")
            print(f"  {attr + ' / ' + g:<26} {int(m.sum()):>6} {rates[g]:>9.1%} {air:>6.2f} {yy.mean():>9.4f} {pp.mean():>8.4f} {z:>+6.1f} {roc_auc_score(yy, pp):>7.4f}")
    print(f"  groups below the {TOL['air_min']:.2f} adverse impact ratio screen: {air_fail}")
    print(f"  groups (n >= 500) whose calibration gap is 2 or more standard errors: {cal_flags}")
    results["fairness"] = len(air_fail) == 0

    # ---------------------------------------------------------------- business view
    section("11. Business view: approval share against default rate among approved (candidate, test)")
    order = np.argsort(pd_te[cand])
    y_sorted = y_te.values[order]
    print(f"  {'approve share':>14} {'bad rate approved':>18} {'bad rate rejected':>18}")
    for share in (0.5, 0.6, 0.7, 0.8, 0.9):
        k = int(share * len(y_sorted))
        print(f"  {share:>14.0%} {y_sorted[:k].mean():>18.2%} {y_sorted[k:].mean():>18.2%}")
    print(f"  overall default rate {y_te.mean():.2%}")

    # ---------------------------------------------------------------- remediation check
    section("12. Follow-up: does a simpler specification resolve the sign finding?")
    dropped = [f"DELAY_{k}" for k in range(2, 7)] + ["UTIL_AVG", "PAY_RATIO_1"] + FEATURE_GROUPS["Education"]
    keep = [c for c in F.columns if c not in dropped]
    v2 = fit_logistic(f_tr[keep], y_tr)
    c2 = standardized_coefficients(v2, keep)
    bad2 = [k for k, e in EXPECTED_SIGN.items() if k in c2 and np.sign(c2[k]) != e]
    p2_va, p2_te = predict_pd(v2, f_va[keep]), predict_pd(v2, f_te[keep])
    print(f"  dropped: {dropped}")
    print(f"  {'':<22} {'features':>9} {'val AUC':>8} {'test AUC':>9} {'test KS':>8} {'sign mismatches':>16}")
    print(f"  {'candidate':<22} {F.shape[1]:>9} {summ[(cand, 'validation')]['auc']:>8.4f} {summ[(cand, 'test')]['auc']:>9.4f} {summ[(cand, 'test')]['ks']:>8.4f} {len(wrong):>16}")
    print(f"  {'simpler specification':<22} {len(keep):>9} {roc_auc_score(y_va, p2_va):>8.4f} {roc_auc_score(y_te, p2_te):>9.4f} {ks_statistic(y_te, p2_te):>8.4f} {len(bad2):>16}")
    print(f"  remaining mismatches: {bad2}; VIF above 10: {int((vif_table(f_tr[keep]) > 10).sum())}")
    print("  (compared on validation AUC; the test figure is shown only as confirmation)")

    # ---------------------------------------------------------------- summary
    section("Summary against pre-set acceptance criteria")
    labels = [
        ("auc", f"Test AUC >= {TOL['auc_test_min']}"), ("ks", f"Test KS >= {TOL['ks_test_min']}"),
        ("overfit", f"Train-test AUC gap <= {TOL['overfit_gap_max']}"),
        ("signs", "All coefficient signs match credit logic"),
        ("calib_mean", f"|mean PD - observed| <= {TOL['calib_mean_abs_max']}"),
        ("calib_slope", f"Calibration slope in {TOL['calib_slope_range']}"),
        ("psi", f"PSI (train vs test) < {TOL['psi_max']}"),
        ("segments", f"Segment AUC >= {TOL['segment_auc_min']} (n >= {TOL['segment_min_n']})"),
        ("sign_stability", f"Top-5 coefficient signs stable in >= {TOL['sign_stability_min']:.0%} of refits"),
        ("noise", f"AUC drop under 10% noise <= {TOL['noise_auc_drop_max']}"),
        ("fairness", f"Adverse impact ratio >= {TOL['air_min']} for all groups"),
        ("challenger_not_better", f"Challenger does NOT beat candidate by >= {TOL['challenger_margin']} AUC"),
    ]
    for key, text in labels:
        print(f"  {'PASS' if results[key] else 'FAIL':<5} {text}")

    # ---------------------------------------------------------------- charts
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for n, color in [(cand, BLUE), (chal, RED), (bench, GRAY)]:
        fpr, tpr, _ = roc_curve(y_te, pd_te[n])
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{n}: AUC {summ[(n, 'test')]['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curves on the Held-Out Test Set")
    ax.legend(loc="lower right", fontsize=8)
    savefig(fig, "roc_curves.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for n, color in [(cand, BLUE), (chal, RED)]:
        d = decile_table(y_te, pd_te[n])
        ax1.plot(d["predicted"], d["observed"], "o-", color=color, label=n)
    top = max(dec["predicted"].max(), dec["observed"].max()) * 1.1
    ax1.plot([0, top], [0, top], "k:", label="Perfect calibration")
    ax1.set_xlabel("Mean predicted PD (decile)")
    ax1.set_ylabel("Observed default rate")
    ax1.set_title("Calibration by Predicted-PD Decile")
    ax1.legend(fontsize=8)
    x = np.arange(1, 11)
    ax2.bar(x - 0.2, dec["predicted"], 0.4, color=BLUE, label="Predicted (candidate)")
    ax2.bar(x + 0.2, dec["observed"], 0.4, color=ORANGE, label="Observed")
    ax2.set_xlabel("Risk decile (1 = lowest predicted PD)")
    ax2.set_ylabel("Default rate")
    ax2.set_title("Rank Ordering: Predicted vs Observed by Decile")
    ax2.legend(fontsize=8)
    savefig(fig, "calibration_deciles.png")

    p_c = pd_te[cand]
    fig, ax = plt.subplots(figsize=(8, 5))
    grid = np.linspace(p_c.min(), p_c.max(), 300)
    cdf_bad = np.array([(p_c[y_te.values == 1] <= g).mean() for g in grid])
    cdf_good = np.array([(p_c[y_te.values == 0] <= g).mean() for g in grid])
    k = int(np.argmax(np.abs(cdf_bad - cdf_good)))
    ax.plot(grid, cdf_good, color=BLUE, linewidth=2, label="Did not default")
    ax.plot(grid, cdf_bad, color=RED, linewidth=2, label="Defaulted")
    ax.vlines(grid[k], cdf_bad[k], cdf_good[k], color="black", linewidth=2)
    ax.text(grid[k] + 0.01, (cdf_bad[k] + cdf_good[k]) / 2, f"KS = {summ[(cand, 'test')]['ks']:.3f}", fontsize=10)
    ax.set_xlabel("Predicted PD (candidate)")
    ax.set_ylabel("Cumulative share of customers")
    ax.set_title("KS: Score Distributions of Defaulters and Non-Defaulters")
    ax.legend(loc="lower right")
    savefig(fig, "ks_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    names = [r[0] for r in seg_rows][::-1]
    aucs = [r[4] for r in seg_rows][::-1]
    ax.barh(names, aucs, color=[RED if (a < TOL["segment_auc_min"]) else BLUE for a in aucs])
    ax.axvline(TOL["segment_auc_min"], color="black", linestyle="--", linewidth=1.5, label=f"Minimum {TOL['segment_auc_min']}")
    ax.axvline(base_auc, color=GRAY, linestyle=":", linewidth=1.5, label=f"Overall {base_auc:.3f}")
    ax.set_xlim(0.5, max(aucs) + 0.05)
    ax.set_xlabel("Test AUC within segment")
    ax.set_title("Where the Candidate Ranks Risk Well and Where It Does Not")
    ax.legend(loc="lower right", fontsize=8)
    savefig(fig, "segment_auc.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    top10 = sorted(coefs, key=lambda k: -abs(coefs[k]))[:10]
    ax.boxplot([boot[c] for c in top10][::-1], vert=False, tick_labels=top10[::-1], showfliers=False)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Standardized coefficient across 200 bootstrap refits")
    ax.set_title("Coefficient Stability: Ten Largest Drivers")
    savefig(fig, "coefficient_stability.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    labels_f = [f"{r[0]}: {r[1]}" for r in fair_rows]
    ax.bar(range(len(fair_rows)), [r[3] * 100 for r in fair_rows],
           color=[RED if r[4] < TOL["air_min"] else BLUE for r in fair_rows])
    ax.axhline(approve.mean() * 100, color="black", linestyle="--", linewidth=1.5, label=f"Overall {approve.mean():.0%}")
    ax.set_xticks(range(len(fair_rows)))
    ax.set_xticklabels(labels_f, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Approval rate (%)")
    ax.set_title("Fairness Screen: Approval Rate by Group (attributes not used by the model)")
    ax.legend()
    savefig(fig, "fairness_screen.png")


if __name__ == "__main__":
    main()

"""Data loading, quality checks, feature engineering and splits for the consumer credit default model.

Data: UCI "Default of credit card clients" (Taiwan, 2005), 30,000 customers, target = default in the
next month. It is consumer credit, used here as a stand-in for loan-level default data. Columns are
anonymized (x1..x23) on OpenML, so they are named below following the UCI documentation.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

OPENML_ID = 42477
SEED = 42
RAW_COLUMNS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + [f"PAY_{i}" for i in range(1, 7)]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
# Attributes a lender may not use in credit decisions (or must treat with care). Excluded from the
# model and only used afterwards for a fairness screen.
PROTECTED = ["SEX", "MARRIAGE", "AGE"]


def load_raw():
    data = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto")
    X = data.data.astype(float).copy()
    X.columns = RAW_COLUMNS
    y = data.target.astype(int).rename("default")
    return X, y


def data_quality(X, y):
    """Counts a validator would flag. PAY codes -2 and 0 are not documented by UCI."""
    return {
        "rows": len(X),
        "default_rate": float(y.mean()),
        "missing_cells": int(X.isna().sum().sum()),
        "duplicate_rows": int(X.duplicated().sum()),
        "education_undocumented_codes (0, 5, 6)": int(X["EDUCATION"].isin([0, 5, 6]).sum()),
        "marriage_undocumented_code (0)": int((X["MARRIAGE"] == 0).sum()),
        "pay_1_undocumented_codes (-2, 0)": int(X["PAY_1"].isin([-2, 0]).sum()),
        "negative_bill_amt1": int((X["BILL_AMT1"] < 0).sum()),
        "bill_amt1_over_limit": int((X["BILL_AMT1"] > X["LIMIT_BAL"]).sum()),
    }


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_features(X):
    """Model features. Protected attributes are not used. PAY_1 (most recent month) is one-hot coded
    because its relationship to default is not monotone (-1 'paid duly' defaults more than 0 'revolving')."""
    f = pd.DataFrame(index=X.index)
    f["LOG_LIMIT"] = np.log(X["LIMIT_BAL"])

    edu = X["EDUCATION"].astype(int).replace({0: 4, 5: 4, 6: 4})  # undocumented codes grouped as 'other'
    for k in (1, 2, 3):
        f[f"EDU_{k}"] = (edu == k).astype(float)

    pay1 = X["PAY_1"].astype(int)
    for code, name in [(-2, "PAY1_NO_USE"), (-1, "PAY1_DULY"), (1, "PAY1_LATE1"), (2, "PAY1_LATE2")]:
        f[name] = (pay1 == code).astype(float)  # reference level: 0 (revolving credit)
    f["PAY1_LATE3P"] = (pay1 >= 3).astype(float)

    for k in range(2, 7):
        f[f"DELAY_{k}"] = X[f"PAY_{k}"].clip(lower=0)
    pays = X[[f"PAY_{k}" for k in range(1, 7)]]
    f["MAX_DELAY"] = pays.clip(lower=0).max(axis=1)
    f["N_LATE_MONTHS"] = (pays > 0).sum(axis=1)

    bills = X[[f"BILL_AMT{k}" for k in range(1, 7)]]
    f["UTIL_1"] = (X["BILL_AMT1"] / X["LIMIT_BAL"]).clip(-0.5, 1.5)
    f["UTIL_AVG"] = (bills.mean(axis=1) / X["LIMIT_BAL"]).clip(-0.5, 1.5)

    # Payment in month k is made against the statement of month k+1 in the file's ordering
    ratios = [(X[f"PAY_AMT{k}"] / X[f"BILL_AMT{k + 1}"].clip(lower=1)).clip(0, 1) for k in range(1, 6)]
    f["PAY_RATIO_1"] = ratios[0]
    f["PAY_RATIO_AVG"] = pd.concat(ratios, axis=1).mean(axis=1)

    f["LOG_BILL_1"] = _signed_log(X["BILL_AMT1"])
    f["LOG_PAYAMT_1"] = np.log1p(X["PAY_AMT1"])
    return f


FEATURE_GROUPS = {
    "Recent status (PAY_1 dummies)": ["PAY1_NO_USE", "PAY1_DULY", "PAY1_LATE1", "PAY1_LATE2", "PAY1_LATE3P"],
    "Delay history (months 2-6, max, count)": [f"DELAY_{k}" for k in range(2, 7)] + ["MAX_DELAY", "N_LATE_MONTHS"],
    "Utilization": ["UTIL_1", "UTIL_AVG"],
    "Payment behaviour": ["PAY_RATIO_1", "PAY_RATIO_AVG", "LOG_PAYAMT_1"],
    "Limit and balance size": ["LOG_LIMIT", "LOG_BILL_1"],
    "Education": ["EDU_1", "EDU_2", "EDU_3"],
}


def split_data(F, y, seed=SEED):
    """Stratified 60 / 20 / 20 train / validation / test split (no time dimension exists in this data)."""
    f_tr, f_tmp, y_tr, y_tmp = train_test_split(F, y, test_size=0.4, stratify=y, random_state=seed)
    f_va, f_te, y_va, y_te = train_test_split(f_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return f_tr, f_va, f_te, y_tr, y_va, y_te

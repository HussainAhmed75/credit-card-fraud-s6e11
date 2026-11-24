"""Loading, cleaning checks and features for Kaggle Playground S6E11
(Predicting Credit Card Fraud).

There is no time column in this data, so there is no "future" to leak.
The leakage risks here are different and are handled explicitly:
  * the row `id` is never a feature,
  * nothing is computed from the target outside a training fold
    (no target encoding on the full data; categoricals stay categorical),
  * no statistic is computed on the test set (no train+test frequency encoding),
  * the internal hold-out set is split off once and never used for any choice.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

TARGET = "is_fraud"
ID = "id"

CATEGORICAL = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOLEAN = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
           "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
NUMERIC_RAW = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
               "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
               "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score",
               "prior_disputes"]
# row-level engineered features: each uses only the row itself
ENGINEERED = ["red_flag_count", "amount_to_balance", "log_amount", "is_night"]
NUMERIC = NUMERIC_RAW + BOOLEAN + ENGINEERED
FEATURES = NUMERIC + CATEGORICAL


def find_file(name: str, roots=("data", "../data")) -> str:
    for r in roots:
        p = os.path.join(r, name)
        if os.path.exists(p):
            return p
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(name)


def load(train_path: str, test_path: str):
    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)
    return tr, te


def prepare(df: pd.DataFrame, categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Add row-level features and fix dtypes. Category levels are taken from
    `categories` when given (so train and test share the same codes)."""
    d = df.copy()
    for c in BOOLEAN:
        d[c] = d[c].astype(float)
    d["red_flag_count"] = d[BOOLEAN].sum(axis=1)
    d["amount_to_balance"] = d["amount_usd"] / (d["account_balance_usd"].abs() + 1.0)
    d["log_amount"] = np.log1p(d["amount_usd"].clip(lower=0))
    d["is_night"] = d["time_of_day_hour"].isin([0, 1, 2, 3, 4, 5]).astype(float)
    cats = categories or {c: sorted(d[c].dropna().astype(str).unique()) for c in CATEGORICAL}
    for c in CATEGORICAL:
        d[c] = pd.Categorical(d[c].astype(str), categories=cats[c])
    if TARGET in d:
        d[TARGET] = d[TARGET].astype(int)
    return d, cats


def data_checks(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    cols = [c for c in tr.columns if c not in (ID, TARGET)]
    overlap = tr[cols].merge(te[cols].drop_duplicates(), on=cols, how="inner")
    return {
        "train_rows": int(len(tr)), "test_rows": int(len(te)),
        "n_raw_features": int(len(cols)),
        "missing_values_train": int(tr.isna().sum().sum()),
        "missing_values_test": int(te.isna().sum().sum()),
        "duplicate_rows_train": int(tr[cols].duplicated().sum()),
        "train_rows_identical_to_a_test_row": int(len(overlap)),
        "fraud_rate_pct": float(tr[TARGET].mean() * 100),
        "n_fraud": int(tr[TARGET].sum()),
    }

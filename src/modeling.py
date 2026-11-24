"""Splits, baselines, models, metrics, tuning and calibration."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

from features import CATEGORICAL, FEATURES, NUMERIC, TARGET

SEED = 42
N_THREADS = 2
HOLDOUT_FRAC = 0.20
N_FOLDS = 5


# ---------------------------------------------------------------- splits
def holdout_split(d: pd.DataFrame):
    """One stratified 80/20 split. The 20% is touched once, at the very end."""
    dev_idx, hold_idx = train_test_split(d.index, test_size=HOLDOUT_FRAC, stratify=d[TARGET],
                                         random_state=SEED)
    return d.loc[dev_idx], d.loc[hold_idx]


def folds(y, n=N_FOLDS, seed=SEED):
    return list(StratifiedKFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


# ---------------------------------------------------------------- metrics
def metrics(y, p, top_frac=0.01) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    k = max(1, int(round(len(p) * top_frac)))
    top = np.argsort(-p, kind="mergesort")[:k]
    caught = y[top].sum()
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "recall_at_top1pct": float(caught / y.sum() * 100),
        "precision_at_top1pct": float(caught / k * 100),
        "brier": float(brier_score_loss(y, pc)),
        "log_loss": float(log_loss(y, pc)),
        "n": int(len(y)),
    }


def accuracy_at_05(y, p) -> float:
    return float(((np.asarray(p) >= 0.5).astype(int) == np.asarray(y)).mean() * 100)


# ---------------------------------------------------------------- baselines
def baseline_majority(d):
    """Always 'not fraud' (a constant score)."""
    return np.zeros(len(d))


def baseline_one_feature(d, col="merchant_risk_score"):
    """ONE strong feature used directly as the score."""
    return d[col].to_numpy(dtype=float)


def baseline_rule(d):
    """Hand-made analyst rule: count of red flags, ties broken by merchant risk."""
    return d["red_flag_count"].to_numpy() * 1000 + d["merchant_risk_score"].to_numpy()


# ---------------------------------------------------------------- models
def X_tree(d):
    return d[FEATURES]


def X_codes(d):
    X = d[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].cat.codes.astype(float)
    return X


def make_logreg():
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000, C=1.0))])


def lgbm_params(**over):
    p = dict(objective="binary", learning_rate=0.05, n_estimators=4000, num_leaves=31,
             min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
             reg_lambda=1.0, random_state=SEED, n_jobs=N_THREADS, deterministic=True,
             force_row_wise=True, verbose=-1)
    p.update(over)
    return p


def fit_lgbm(Xtr, ytr, Xva=None, yva=None, **over):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(**lgbm_params(**over))
    if Xva is not None:
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
    else:
        m.fit(Xtr, ytr)
    return m


def fit_xgb(Xtr, ytr, Xva, yva):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=4000, learning_rate=0.05, max_depth=5, subsample=0.8,
                          colsample_bytree=0.7, min_child_weight=5, tree_method="hist",
                          enable_categorical=True, max_cat_to_onehot=1, eval_metric="auc",
                          early_stopping_rounds=200, random_state=SEED, n_jobs=N_THREADS)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def fit_rf(Xtr, ytr, n_sub=200_000):
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(Xtr), size=min(n_sub, len(Xtr)), replace=False)
    m = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_features=0.4,
                               n_jobs=N_THREADS, random_state=SEED)
    m.fit(Xtr.iloc[idx], np.asarray(ytr)[idx])
    return m


def cv_oof(d, kind, fold_list, **over):
    """Out-of-fold predictions on the development set. Every model and every
    preprocessing step is fit on the training part of each fold only."""
    y = d[TARGET].to_numpy()
    oof = np.zeros(len(d))
    iters = []
    for tr_i, va_i in fold_list:
        a, b = d.iloc[tr_i], d.iloc[va_i]
        if kind == "logreg":
            m = make_logreg().fit(a[FEATURES], a[TARGET])
            oof[va_i] = m.predict_proba(b[FEATURES])[:, 1]
        elif kind == "rf":
            m = fit_rf(X_codes(a), a[TARGET])
            oof[va_i] = m.predict_proba(X_codes(b))[:, 1]
        elif kind == "xgb":
            m = fit_xgb(X_tree(a), a[TARGET], X_tree(b), b[TARGET])
            oof[va_i] = m.predict_proba(X_tree(b))[:, 1]
            iters.append(int(m.best_iteration))
        elif kind == "lgbm":
            m = fit_lgbm(X_tree(a), a[TARGET], X_tree(b), b[TARGET], **over)
            oof[va_i] = m.predict_proba(X_tree(b))[:, 1]
            iters.append(int(m.best_iteration_))
        else:
            raise ValueError(kind)
    return oof, iters


def timed(fn, *a, **k):
    t = time.time()
    out = fn(*a, **k)
    return out, round(time.time() - t, 1)


# ---------------------------------------------------------------- tuning
def tune_lgbm(d, n_trials=20, n_folds=3):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fl = folds(d[TARGET].to_numpy(), n=n_folds, seed=SEED + 1)

    def objective(trial):
        over = dict(
            learning_rate=0.1,
            num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 1000, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
        )
        oof, iters = cv_oof(d, "lgbm", fl, **over)
        trial.set_user_attr("mean_iter", float(np.mean(iters)))
        return roc_auc_score(d[TARGET], oof)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study


# ---------------------------------------------------------------- calibration
def reliability(y, p, n_bins=10):
    """Bins by predicted-probability quantile; returns mean predicted vs observed rate."""
    q = pd.qcut(pd.Series(p), q=n_bins, duplicates="drop")
    g = pd.DataFrame({"y": np.asarray(y), "p": p, "bin": q.values}).groupby("bin", observed=True)
    return g.agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size")).reset_index(drop=True)

"""Leakage and split tests.  Run:  pytest -q tests/"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from features import CATEGORICAL, FEATURES, ID, TARGET, find_file, load, prepare  # noqa: E402
import modeling as mdl  # noqa: E402


@pytest.fixture(scope="module")
def data():
    tr, te = load(find_file("train.csv", (os.path.join(ROOT, "data"),)), find_file("test.csv", (os.path.join(ROOT, "data"),)))
    d, cats = prepare(tr)
    t, _ = prepare(te, categories=cats)
    return d, t


def test_id_and_target_are_not_features():
    assert ID not in FEATURES and TARGET not in FEATURES


def test_features_are_row_level_only(data):
    """Each feature of a row must depend on that row alone: shuffling or dropping
    all OTHER rows cannot change it (no group statistics, no target encoding)."""
    d, _ = data
    sub = d.sample(2000, random_state=0)
    raw = sub.drop(columns=["red_flag_count", "amount_to_balance", "log_amount", "is_night"])
    raw = raw.assign(**{c: raw[c].astype(str) for c in CATEGORICAL})
    raw2, _ = prepare(raw.iloc[::-1], categories={c: list(d[c].cat.categories) for c in CATEGORICAL})
    a = sub[FEATURES].assign(**{c: sub[c].cat.codes for c in CATEGORICAL})
    b = raw2.loc[sub.index, FEATURES].assign(**{c: raw2.loc[sub.index, c].cat.codes for c in CATEGORICAL})
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_features_do_not_use_the_target(data):
    d, _ = data
    flipped = d.copy()
    flipped[TARGET] = 1 - flipped[TARGET]
    raw = flipped.drop(columns=["red_flag_count", "amount_to_balance", "log_amount", "is_night"])
    raw = raw.assign(**{c: raw[c].astype(str) for c in CATEGORICAL})
    again, _ = prepare(raw, categories={c: list(d[c].cat.categories) for c in CATEGORICAL})
    num = [f for f in FEATURES if f not in CATEGORICAL]
    assert np.allclose(d[num].to_numpy(float), again[num].to_numpy(float))


def test_holdout_is_disjoint_and_stratified(data):
    d, _ = data
    dev, hold = mdl.holdout_split(d)
    assert set(dev.index).isdisjoint(hold.index)
    assert len(dev) + len(hold) == len(d)
    assert abs(dev[TARGET].mean() - hold[TARGET].mean()) < 0.001


def test_cv_folds_never_overlap_and_exclude_holdout(data):
    d, _ = data
    dev, hold = mdl.holdout_split(d)
    fl = mdl.folds(dev[TARGET].to_numpy())
    seen = np.zeros(len(dev), int)
    for tr_i, va_i in fl:
        assert set(tr_i).isdisjoint(va_i)
        seen[va_i] += 1
    assert (seen == 1).all()          # every dev row validated exactly once
    hold_ids = set(hold[ID])
    assert hold_ids.isdisjoint(set(dev[ID]))


def test_logreg_preprocessing_is_inside_the_pipeline():
    """Scaling / one-hot are fit inside each CV fold because they live in the Pipeline."""
    m = mdl.make_logreg()
    assert type(m).__name__ == "Pipeline" and m.steps[0][0] == "pre"


def test_no_train_row_appears_in_test(data):
    d, t = data
    raw = [c for c in d.columns if c not in (ID, TARGET, "red_flag_count", "amount_to_balance", "log_amount", "is_night")]
    a = d[raw].sample(20000, random_state=1).astype(str)
    m = a.merge(t[raw].astype(str), on=raw, how="inner")
    assert len(m) == 0

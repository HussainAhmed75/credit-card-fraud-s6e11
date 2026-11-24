"""Streamlit app: hold-out results explorer + what-if fraud scorer.
Run from the repo root:  streamlit run app/streamlit_app.py"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from features import prepare  # noqa: E402

P = lambda *p: os.path.join(ROOT, *p)
st.set_page_config(page_title="Credit-card fraud model", layout="wide")


@st.cache_data
def load():
    m = json.load(open(P("results", "metrics.json"), encoding="utf-8"))
    hp = pd.read_csv(P("results", "holdout_predictions.csv.gz"))
    meta = json.load(open(P("models", "model_meta.json")))
    return m, hp, meta


@st.cache_resource
def model():
    return lgb.Booster(model_file=P("models", "lgbm_full.txt"))


M, HP, META = load()
B = model()
H = M["headline"]
st.title("Catching credit-card fraud")
st.warning("Synthetic data (Kaggle Playground S6E11). Patterns describe the generated data, not a real bank.")
c = st.columns(5)
c[0].metric("Hold-out ROC AUC", f"{H['holdout_auc']:.4f}")
c[1].metric("Hold-out PR AUC", f"{H['holdout_pr_auc']:.3f}")
c[2].metric("Fraud caught in top 1%", f"{H['holdout_recall_top1pct']:.1f}%")
c[3].metric("Precision in top 1%", f"{H['holdout_precision_top1pct']:.1f}%")
c[4].metric("Accuracy of 'never fraud'", f"{H['majority_holdout_accuracy']:.2f}%")

t1, t2 = st.tabs(["Results explorer", "What-if scorer"])
with t1:
    a, b = st.columns(2)
    cat = a.selectbox("Merchant category", ["All"] + sorted(HP["merchant_category"].unique()))
    review = b.slider("Review the riskiest X% of transactions", 0.5, 10.0, 1.0, 0.5)
    d = HP if cat == "All" else HP[HP.merchant_category == cat]
    thr = np.quantile(HP["p"], 1 - review / 100)
    flagged = d[d.p >= thr]
    caught = int(flagged["is_fraud"].sum()); total = int(d["is_fraud"].sum())
    k = st.columns(4)
    k[0].metric("Transactions", f"{len(d):,}")
    k[1].metric("Flagged for review", f"{len(flagged):,}")
    k[2].metric("Fraud caught", f"{caught:,} of {total:,}" if total else "0")
    k[3].metric("Precision of flags", f"{(caught / len(flagged) * 100) if len(flagged) else 0:.1f}%")
    s = d.sample(min(len(d), 6000), random_state=0)
    fig = px.histogram(s, x="p", color=s["is_fraud"].map({0: "not fraud", 1: "fraud"}), nbins=60, log_y=True,
                       barmode="overlay", color_discrete_map={"fraud": "#B23A48", "not fraud": "#A7B0BE"},
                       labels={"p": "Predicted fraud probability", "color": ""})
    fig.add_vline(x=thr, line_dash="dash", annotation_text="review threshold")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(flagged.sort_values("p", ascending=False).head(300), width="stretch")

with t2:
    st.write("Describe a transaction; the model trained on all the competition training data scores it.")
    D, Rg = META["defaults"], META["ranges"]
    row = {}
    cols = st.columns(4)
    for i, cname in enumerate(META["numeric_raw"]):
        lo, hi = Rg[cname]
        isint = float(D[cname]).is_integer() and hi - lo < 200
        row[cname] = cols[i % 4].number_input(cname.replace("_", " "), value=float(D[cname]), min_value=float(min(lo, D[cname])),
                                              max_value=float(max(hi, D[cname])), step=1.0 if isint else 0.1)
    cols = st.columns(5)
    for i, cname in enumerate(META["categorical"]):
        opts = META["categories"][cname]
        row[cname] = cols[i].selectbox(cname.replace("_", " "), opts, index=opts.index(D[cname]))
    cols = st.columns(6)
    for i, cname in enumerate(META["boolean"]):
        row[cname] = cols[i].checkbox(cname.replace("_", " "), False)
    X, _ = prepare(pd.DataFrame([row]), categories=META["categories"])
    p = float(B.predict(X[META["features"]])[0])
    r = st.columns(3)
    r[0].metric("Predicted fraud probability", f"{p * 100:.2f}%")
    r[1].metric("vs average transaction", f"{p / META['fraud_rate']:.1f}x")
    r[2].metric("Would be reviewed (top 1%)?", "Yes" if p >= META["review_threshold_prob"] else "No")

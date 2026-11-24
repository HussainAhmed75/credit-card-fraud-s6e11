"""Builds dashboard/fraud_dashboard.html — one self-contained offline Plotly page.
Every number comes from results/metrics.json or results/holdout_predictions.csv.gz."""
from __future__ import annotations

import html
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402


def _layout(fig, title, sub=None, height=460):
    t = f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:#5B6676'>{sub}</span>" if sub else "")
    fig.update_layout(**style.PLOTLY_LAYOUT, title=dict(text=t, x=0.01, xanchor="left"), height=height)
    return fig


def build(root: str) -> str:
    M = json.load(open(os.path.join(root, "results", "metrics.json"), encoding="utf-8"))
    hp = pd.read_csv(os.path.join(root, "results", "holdout_predictions.csv.gz"))
    H = M["headline"]
    figs = []

    # 1 model comparison
    names = [n for n in ["Always 'not fraud'", "One feature (merchant risk score)", "Red-flag rule", "Random Forest",
                         "Logistic Regression", "XGBoost", "LightGBM", "LightGBM (tuned)",
                         "Blend: tuned LightGBM + LogReg (rank avg)"] if n in M["cv"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="ROC AUC (CV)", x=names, y=[M["cv"][n]["roc_auc"] for n in names], marker_color=style.ACCENT_2,
                         text=[f"{M['cv'][n]['roc_auc']:.3f}" for n in names], textposition="outside"))
    fig.add_trace(go.Bar(name="PR AUC (CV)", x=names, y=[M["cv"][n]["pr_auc"] for n in names], marker_color=style.ACCENT,
                         text=[f"{M['cv'][n]['pr_auc']:.3f}" for n in names], textposition="outside"))
    fig.update_layout(barmode="group", yaxis_range=[0, 1.1], uniformtext=dict(minsize=9, mode="show"))
    figs.append(_layout(fig, "Models vs baselines (5-fold CV)", "PR AUC is the harder, more honest score when fraud is rare", 500))

    # 2 gain curve (hold-out) for chosen model's LightGBM probabilities
    y = hp["is_fraud"].to_numpy(); p = hp["p"].to_numpy()
    o = np.argsort(-p, kind="mergesort"); c = np.cumsum(y[o]) / y.sum(); x = np.arange(1, len(y) + 1) / len(y)
    k = int(len(x) * 0.2)
    fig = go.Figure(go.Scatter(x=x[:k] * 100, y=c[:k] * 100, mode="lines", line=dict(color=style.ACCENT, width=3),
                               hovertemplate="review %{x:.2f}% → catch %{y:.1f}% of fraud<extra></extra>"))
    fig.update_layout(xaxis_title="% of transactions reviewed (riskiest first)", yaxis_title="% of all fraud caught", yaxis_range=[0, 100])
    figs.append(_layout(fig, "How much fraud do you catch for the effort?", "Hold-out set, tuned LightGBM — hover to read any point"))

    # 3 explorer by merchant category
    cats = sorted(hp["merchant_category"].unique())
    fig = go.Figure(); buttons = []
    hp["bin"] = pd.qcut(hp["p"].rank(method="first"), 10, labels=False)
    for i, cat in enumerate(["All"] + cats):
        s = hp if cat == "All" else hp[hp.merchant_category == cat]
        g = s.groupby("bin").agg(pred=("p", "mean"), obs=("is_fraud", "mean"), n=("p", "size"))
        vis = i == 0
        fig.add_trace(go.Bar(x=g.index + 1, y=g["obs"] * 100, name="Observed fraud %", marker_color=style.ACCENT_2, visible=vis,
                             customdata=g["n"], hovertemplate="decile %{x}<br>observed %{y:.2f}%<br>%{customdata} txns<extra></extra>"))
        fig.add_trace(go.Scatter(x=g.index + 1, y=g["pred"] * 100, name="Predicted fraud %", mode="lines+markers",
                                 line=dict(color=style.ACCENT, width=3), visible=vis))
        mask = [False] * (2 * (len(cats) + 1)); mask[2 * i] = mask[2 * i + 1] = True
        buttons.append(dict(label=cat, method="update", args=[{"visible": mask},
                       {"title.text": f"<b>Results explorer — {html.escape(cat)}</b><br><span style='font-size:12px;color:#5B6676'>{len(s):,} hold-out transactions · {s.is_fraud.sum():,} fraud</span>"}]))
    fig.update_layout(updatemenus=[dict(buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.2, yanchor="top")],
                      xaxis_title="Risk decile (1 = safest, 10 = riskiest; deciles of the whole hold-out)", yaxis_title="Fraud rate (%)")
    figs.append(_layout(fig, "Results explorer — All", f"{len(hp):,} hold-out transactions · {int(hp.is_fraud.sum()):,} fraud", 500))

    # 4 flags
    fr = pd.DataFrame(M["flag_rates"]).T.sort_values("with_pct")
    fig = go.Figure([go.Bar(y=fr.index, x=fr["without_pct"], orientation="h", name="flag absent", marker_color=style.MUTED),
                     go.Bar(y=fr.index, x=fr["with_pct"], orientation="h", name="flag present", marker_color=style.ACCENT,
                            text=[f"{v:.1f}%" for v in fr["with_pct"]], textposition="outside", cliponaxis=False)])
    fig.update_layout(barmode="group", xaxis_title="Fraud rate (%)")
    figs.append(_layout(fig, "Red flags and fraud rate", "Training data", 480))

    k1 = [(f"{H['holdout_auc']:.4f}", "ROC AUC on the locked hold-out"),
          (f"{H['holdout_pr_auc']:.3f}", "PR AUC on the hold-out"),
          (f"{H['holdout_recall_top1pct']:.1f}%", "of fraud caught reviewing the riskiest 1%"),
          (f"{H['holdout_precision_top1pct']:.1f}%", "of those reviewed are really fraud"),
          (f"{H['majority_holdout_accuracy']:.2f}%", "accuracy of always saying 'not fraud'")]
    parts = [f.to_html(full_html=False, include_plotlyjs="inline" if i == 0 else False, config={"displaylogo": False})
             for i, f in enumerate(figs)]
    case_rows = "".join(f"<tr><td>{c['case']}</td><td>{c['predicted_prob_pct']}%</td><td>{c['actual']}</td><td>{c['amount_usd']}</td>"
                        f"<td>{html.escape(c['merchant_category'])}</td><td>{html.escape(c['auth_method'])}</td><td>{c['red_flags']}</td>"
                        f"<td>{html.escape(', '.join(c['top_shap']))}</td></tr>" for c in M["case_studies"])
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Credit-card fraud model — dashboard</title><style>body{{font-family:'DejaVu Sans',Arial,sans-serif;color:{style.INK};margin:0;background:#F7F8FA}}
.wrap{{max-width:1150px;margin:0 auto;padding:28px}} .sub{{color:#5B6676;margin-bottom:22px}}
.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:18px}} .kpi{{background:#fff;border-radius:10px;padding:14px;border:1px solid {style.GRID}}}
.kpi b{{font-size:24px;color:{style.ACCENT}}} .kpi span{{display:block;font-size:13px;color:#5B6676}}
.card{{background:#fff;border-radius:10px;padding:8px;border:1px solid {style.GRID};margin-bottom:16px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid {style.GRID};padding:6px;text-align:left}}
.warn{{background:#FFF7E6;border:1px solid #F0D9A8;border-radius:10px;padding:10px 14px;margin-bottom:16px;font-size:14px}}
.foot{{font-size:12px;color:#7A8494}}</style></head><body><div class="wrap">
<h1>Catching credit-card fraud</h1><div class="sub">Kaggle Playground S6E11 · {M['data']['train_rows']:,} training transactions · fraud rate {M['facts']['fraud_rate_pct']}%</div>
<div class="warn"><b>Synthetic data:</b> Kaggle generated this dataset from another one. Patterns describe the generated data, not a real bank.</div>
<div class="kpis">{''.join(f'<div class="kpi"><b>{a}</b><span>{b}</span></div>' for a, b in k1)}</div>
{''.join(f'<div class="card">{p}</div>' for p in parts)}
<div class="card" style="padding:16px"><h3>Case studies (hold-out)</h3><table><tr><th>Case</th><th>Predicted</th><th>Actual</th><th>Amount $</th><th>Merchant</th><th>Auth</th><th>Red flags</th><th>Top SHAP</th></tr>{case_rows}</table></div>
<p class="foot">Data: Kaggle Playground Series S6E11 (CC BY 4.0). Code: MIT. Built by Hussain Ahmed.</p></div></body></html>"""
    out = os.path.join(root, "dashboard", "fraud_dashboard.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write(page)
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

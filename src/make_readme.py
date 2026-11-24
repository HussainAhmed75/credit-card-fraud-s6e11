"""Renders README.md from results/metrics.json (no hand-typed numbers)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = json.load(open(os.path.join(ROOT, "results", "metrics.json"), encoding="utf-8"))
LB = {}
lb_path = os.path.join(ROOT, "results", "kaggle_leaderboard.json")
if os.path.exists(lb_path):
    LB = json.load(open(lb_path, encoding="utf-8"))
H, F, D, S, T, CAL = M["headline"], M["facts"], M["data"], M["splits"], M["tuning"], M["calibration_lgbm_holdout"]
GH = "https://github.com/HussainAhmed75/credit-card-fraud-s6e11"
KAGGLE_NB = "https://www.kaggle.com/code/hussianahmed/credit-card-fraud-honest-model-s6e11"


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def row(name, m):
    acc = m.get("accuracy_at_0.5")
    return (f"| {name} | {fmt(m['roc_auc'])} | {fmt(m['pr_auc'], 3)} | {m['recall_at_top1pct']:.1f}% | "
            f"{m['precision_at_top1pct']:.1f}% | {('—' if acc is None else f'{acc:.2f}%')} |")


order = ["Always 'not fraud'", "One feature (merchant risk score)", "Red-flag rule", "Logistic Regression", "Random Forest",
         "XGBoost", "LightGBM", "LightGBM (tuned)", "Blend: tuned LightGBM + LogReg (rank avg)"]
cv_rows = "\n".join(row(n, M["cv"][n]) for n in order if n in M["cv"])
ho_rows = "\n".join(row(n, M["holdout"][n]) for n in order if n in M["holdout"])
g = M["gain_holdout"][H["chosen_model"]]
gain_row = " | ".join(f"{k}: **{v:.1f}%**" for k, v in sorted(g.items(), key=lambda kv: float(kv[0].rstrip("%"))))
case_rows = "\n".join(f"| {c['case']} | {c['predicted_prob_pct']}% | {'fraud' if c['actual'] else 'not fraud'} | ${c['amount_usd']:,.2f} | "
                      f"{c['merchant_category']} | {c['auth_method']} | {c['red_flags']} | {', '.join(c['top_shap'])} |"
                      for c in M["case_studies"])
lb_line = ""
if LB:
    lb_line = (f"\n**Kaggle public leaderboard:** {LB['public_score']:.5f} (scored by Kaggle on ~20% of the test set, "
               f"checked on {LB['checked_on']}; the private score is only revealed after the competition ends). "
               f"The #1 public score at that time was {LB['top_public_score_seen']:.5f}, so this is a solid, honest baseline, not a top-of-leaderboard entry.\n")

readme = f"""# Catching credit-card fraud — an honest model for Kaggle Playground S6E11

> ⚠️ **Synthetic data.** Kaggle generated this competition's data from another dataset. Every pattern below describes
> the generated data, not a real bank's customers.

Predict the probability that a card transaction is fraud. Only **{F['fraud_rate_pct']}%** of transactions are fraud, so a
model that always says "not fraud" is **{F['majority_class_accuracy_pct']}% accurate** and completely useless —
which is why this project reports ROC AUC, PR AUC and "how much fraud do you catch if you review the riskiest 1%".

**Headline (locked 20% hold-out, scored once):** ROC AUC **{H['holdout_auc']:.4f}**, PR AUC **{H['holdout_pr_auc']:.3f}**.
Reviewing the riskiest 1% of transactions catches **{H['holdout_recall_top1pct']:.1f}%** of all fraud, and
**{H['holdout_precision_top1pct']:.1f}%** of the reviewed cases are real fraud ({H['lift_top1pct_x']}x the base rate).
A hand-made red-flag rule gets ROC AUC {H['rule_holdout_auc']:.4f} and catches {H['rule_recall_top1pct']:.1f}% at the same effort.
{lb_line}
![](figures/06_model_comparison.png)

## What the data says
1. Transactions with **no authentication** are fraud {F['no_auth_fraud_pct']}% of the time vs {F['biometric_fraud_pct']}% with biometrics ({F['no_auth_vs_biometric_x']}x).
2. An **IP-country mismatch** lifts the fraud rate from {F['ip_match_fraud_pct']}% to {F['ip_mismatch_fraud_pct']}%.
3. **{F['riskiest_category']}** are the riskiest purchase category ({F['riskiest_category_pct']}%); {F['safest_category']} the safest ({F['safest_category_pct']}%).
4. With zero red flags the fraud rate is {F['zero_flags_fraud_pct']}%; with three or more it is **{F['three_plus_flags_fraud_pct']}%** ({F['three_plus_flags_share_pct']}% of transactions).
5. The strongest single number is `{F['best_single_feature']}` (AUC {F['best_single_feature_auc']} alone). Time of day carries almost nothing (AUC {F['time_of_day_auc']}).
6. The dataset even has an `is_ai_generated_scam_attempt` flag: {F['ai_scam_flag_n']:,} transactions, {F['ai_scam_flag_fraud_pct']}% of them fraud.

| | |
|---|---|
| ![](figures/01_auth_method.png) | ![](figures/02_red_flags.png) |
| ![](figures/03_merchant_category.png) | ![](figures/04_flag_count.png) |

## Data and cleaning
* {D['train_rows']:,} training rows, {D['test_rows']:,} test rows, {D['n_raw_features']} raw columns. Licence: CC BY 4.0 (Kaggle).
* Checks: {D['missing_values_train']} missing values, {D['duplicate_rows_train']} duplicate rows, {D['train_rows_identical_to_a_test_row']} training rows identical to a test row → **no rows removed**.
* Booleans → 0/1, categories kept as categories, plus four row-level features (`red_flag_count`, `amount_to_balance`, `log_amount`, `is_night`).

## How it was validated (and how leakage was prevented)
* There is **no date column**, so there is no future to protect; a time split would be invented. Stratified K-fold is used instead.
* **{S['holdout_rows']:,} rows (20%)** were locked away as a hold-out and scored once. All choices used 5-fold CV on the other {S['dev_rows']:,}.
* No target encoding, no statistics from the test set, `id` is never a feature. Scaling / one-hot live inside a scikit-learn Pipeline, so they are fit per fold.
* [`tests/test_no_leakage.py`](tests/test_no_leakage.py) checks: features are row-level only (other rows can't change them), features don't use the target,
  hold-out and folds never overlap, preprocessing is inside the pipeline, and no training row appears in the test set.

## Results
### 5-fold CV on the development set (used for every choice)
| Model | ROC AUC | PR AUC | Fraud caught (top 1%) | Precision (top 1%) | Accuracy @0.5 |
|---|---:|---:|---:|---:|---:|
{cv_rows}

XGBoost ({M['cv']['XGBoost']['roc_auc']:.4f}) and LightGBM ({M['cv']['LightGBM']['roc_auc']:.4f}) were practically tied before tuning;
LightGBM was tuned because it trained faster in these runs. Optuna ({T['n_trials']} trials, {T['tuning_folds']}-fold CV) moved LightGBM's CV AUC from {T['cv_auc_untuned']:.5f} to {T['cv_auc_tuned']:.5f}.
Chosen on CV: **{H['chosen_model']}** (CV AUC {H['cv_auc_chosen']:.5f}).

### Locked hold-out (scored once)
| Model | ROC AUC | PR AUC | Fraud caught (top 1%) | Precision (top 1%) | Accuracy @0.5 |
|---|---:|---:|---:|---:|---:|
{ho_rows}

Honest notes: a plain logistic regression (hold-out AUC {H['logreg_holdout_auc']:.4f}) is close to the tuned LightGBM ({H['lgbm_holdout_auc']:.4f}).
Accuracy is shown only to make a point: always saying "not fraud" scores {H['majority_holdout_accuracy']:.2f}%, and the tuned model at a 0.5 threshold scores {H['lgbm_holdout_accuracy_at_05']:.2f}%.
The blend outputs ranks, not probabilities, so its accuracy is not meaningful.

**Share of all fraud caught vs share of transactions reviewed (hold-out):** {gain_row}

| | |
|---|---|
| ![](figures/05_gain_curve.png) | ![](figures/07_calibration.png) |

### Calibration
Tuned LightGBM on the hold-out: Brier score {CAL['brier']:.5f} (a constant fraud-rate guess scores {CAL['brier_constant_rate']:.5f}).
In the riskiest tenth of transactions it predicts {CAL['top_bin_mean_pred_pct']}% fraud and {CAL['top_bin_observed_pct']}% actually is;
the largest gap across the 10 bins is {CAL['max_abs_gap_pct_points']} percentage points.

### Explanation
![](figures/08_shap.png)

| Case | Predicted | Actual | Amount | Merchant | Auth | Red flags | Top SHAP (+ raises risk, − lowers) |
|---|---:|---|---:|---|---|---:|---|
{case_rows}

## Limitations
* **Synthetic data** — real fraud is messier, rarer in places, and changes over time (fraudsters adapt).
* No time column, so there is no "next month" test; a real deployment would need one.
* The 1% review threshold is an example. The right threshold depends on analyst capacity and the cost of a missed fraud vs a blocked customer.
* The extra "original" dataset that many competitors add to training was not used.

## Project structure
```
src/features.py        data checks + row-level features
src/modeling.py        hold-out & CV, baselines, models, metrics, Optuna, calibration
src/style.py           one chart theme
src/make_dashboard.py  offline Plotly dashboard (dashboard/fraud_dashboard.html)
src/make_readme.py     renders this README from results/metrics.json
notebooks/             main notebook (+ self-contained Kaggle version)
app/streamlit_app.py   results explorer + what-if scorer
tests/                 leakage tests
scripts/verify_numbers.py  checks every number in a text file exists in metrics.json
```

## How to run
```bash
pip install -r requirements.txt
# put train.csv and test.csv from the competition into data/
pytest -q tests/
jupyter nbconvert --to notebook --execute --inplace notebooks/fraud_model.ipynb
python src/make_readme.py && python scripts/verify_numbers.py README.md
streamlit run app/streamlit_app.py
```

## License
Code: MIT. Data: Kaggle Playground Series S6E11, CC BY 4.0 (not included in this repo).

---
Built by **Hussain Ahmed** · [GitHub](https://github.com/HussainAhmed75) · [Kaggle](https://www.kaggle.com/hussianahmed) · [Notebook on Kaggle]({KAGGLE_NB})
"""
open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8").write(readme)
print("README.md written")

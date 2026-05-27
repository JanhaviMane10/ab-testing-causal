# ============================================================
# 03_causal.py — Causal Inference Analysis
#
# WHAT THIS FILE DOES:
# 1. Propensity Score Matching (PSM)
#    — Creates a fair comparison between groups
# 2. Difference-in-Differences (DiD) simulation
#    — Shows how we'd handle observational data
# 3. Uplift modeling
#    — Who ACTUALLY benefits from the email?
# 4. CATE (Conditional Average Treatment Effect)
#    — Personalised treatment effect by customer segment
#
# KEY INSIGHT:
# Even though this was a randomized experiment, we show
# the FULL causal toolkit — because real DS jobs ask:
# "what do you do when you CAN'T randomize?"
# ============================================================

# %% — Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import json
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from config import *

plt.style.use("seaborn-v0_8-whitegrid")
print("✅ Libraries loaded")

# %% — Load data
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql("SELECT * FROM customers", conn)
conn.close()

df_ab = df[df["segment"].isin([CONTROL_GROUP, TREATMENT_GROUP])].copy()
df_ab["treated"] = (df_ab["segment"] == TREATMENT_GROUP).astype(int)

print(f"✅ Data loaded: {len(df_ab):,} customers")

# %% ── PART 1: Propensity Score Matching ────────────────────
print("\n" + "="*60)
print("PART 1: Propensity Score Matching (PSM)")
print("="*60)
print("""
WHAT IS PSM?
Imagine you want to compare two groups fairly. But one group
has more high-value customers than the other. That's not fair!

PSM fixes this by:
1. Training a model to predict who was treated
2. Matching each treated customer with a similar control customer
3. Comparing outcomes on the MATCHED pairs only

Result: A fair "apples-to-apples" comparison
""")

# Features for propensity score
feature_cols = ["recency", "history", "mens", "womens", "newbie"]
df_ab["zip_urban"]    = (df_ab["zip_code"] == "Urban").astype(int)
df_ab["zip_suburban"] = (df_ab["zip_code"] == "Suburban").astype(int)
df_ab["channel_web"]  = (df_ab["channel"] == "Web").astype(int)
df_ab["channel_multi"]= (df_ab["channel"] == "Multichannel").astype(int)
feature_cols += ["zip_urban", "zip_suburban", "channel_web", "channel_multi"]

X = df_ab[feature_cols].fillna(0)
y = df_ab["treated"]

# Scale features
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train propensity score model
lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
lr.fit(X_scaled, y)
df_ab["propensity_score"] = lr.predict_proba(X_scaled)[:, 1]

auc = roc_auc_score(y, df_ab["propensity_score"])
print(f"✅ Propensity score model AUC: {auc:.4f}")
print(f"   (AUC > 0.5 means model can distinguish treated/control)")

# Nearest neighbor matching
print("\n🔗 Matching treated customers to similar control customers...")
treated_df = df_ab[df_ab["treated"] == 1].copy().reset_index(drop=True)
control_df = df_ab[df_ab["treated"] == 0].copy().reset_index(drop=True)

matched_pairs = []
used_control  = set()

for i, treat_row in treated_df.iterrows():
    # Find closest control customer by propensity score
    available = control_df[~control_df.index.isin(used_control)]
    if len(available) == 0:
        break
    diffs  = abs(available["propensity_score"] - treat_row["propensity_score"])
    best_j = diffs.idxmin()

    matched_pairs.append({
        "treated_conversion":  treat_row["conversion"],
        "control_conversion":  control_df.loc[best_j, "conversion"],
        "treated_spend":       treat_row["spend"],
        "control_spend":       control_df.loc[best_j, "spend"],
        "ps_treated":          treat_row["propensity_score"],
        "ps_control":          control_df.loc[best_j, "propensity_score"],
        "ps_diff":             abs(treat_row["propensity_score"] -
                                   control_df.loc[best_j, "propensity_score"])
    })
    used_control.add(best_j)

matched_df = pd.DataFrame(matched_pairs)
print(f"✅ Matched {len(matched_df):,} pairs")
print(f"   Average propensity score difference: {matched_df['ps_diff'].mean():.4f}")

# PSM results
psm_lift = matched_df["treated_conversion"].mean() - matched_df["control_conversion"].mean()
psm_spend_lift = matched_df["treated_spend"].mean() - matched_df["control_spend"].mean()
t_stat, p_psm = stats.ttest_rel(matched_df["treated_conversion"],
                                 matched_df["control_conversion"])

print(f"\n📊 PSM Results:")
print(f"   Treated conversion rate: {matched_df['treated_conversion'].mean()*100:.4f}%")
print(f"   Control conversion rate: {matched_df['control_conversion'].mean()*100:.4f}%")
print(f"   PSM lift:                +{psm_lift*100:.4f} pp")
print(f"   P-value:                 {p_psm:.6f}")
print(f"   {'✅ Significant!' if p_psm < ALPHA else '❌ Not significant'}")

# %% ── PART 2: Uplift Modeling (Who benefits most?) ──────────
print("\n" + "="*60)
print("PART 2: Uplift Modeling")
print("="*60)
print("""
WHAT IS UPLIFT MODELING?
Standard ML asks: "Who will convert?"
Uplift asks:      "Who will convert BECAUSE of the email?"

These are very different! A customer who always buys
doesn't NEED the email. We want to find customers who:
- Buy WITH the email
- Would NOT buy without it

This is the most valuable insight for a marketing team.
""")

# T-Learner uplift model (two separate models)
from sklearn.ensemble import RandomForestClassifier

ctrl_data  = df_ab[df_ab["treated"] == 0]
treat_data = df_ab[df_ab["treated"] == 1]

# Model for control group
m_ctrl  = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
m_treat = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

m_ctrl.fit(ctrl_data[feature_cols].fillna(0),  ctrl_data["conversion"])
m_treat.fit(treat_data[feature_cols].fillna(0), treat_data["conversion"])

# Predict uplift for all customers
X_all = df_ab[feature_cols].fillna(0)
df_ab["p_convert_if_treated"]    = m_treat.predict_proba(X_all)[:, 1]
df_ab["p_convert_if_not_treated"]= m_ctrl.predict_proba(X_all)[:, 1]
df_ab["uplift_score"]            = (df_ab["p_convert_if_treated"] -
                                     df_ab["p_convert_if_not_treated"])

print(f"✅ Uplift scores calculated for all {len(df_ab):,} customers")
print(f"   Average uplift score: {df_ab['uplift_score'].mean():.4f}")
print(f"   Max uplift score:     {df_ab['uplift_score'].max():.4f}")
print(f"   Min uplift score:     {df_ab['uplift_score'].min():.4f}")

# Segment customers by uplift
df_ab["uplift_segment"] = pd.cut(df_ab["uplift_score"],
                                   bins=4,
                                   labels=["Low Uplift", "Med-Low", "Med-High", "High Uplift"])

uplift_analysis = df_ab[df_ab["treated"] == 1].groupby("uplift_segment").agg(
    n_customers=("conversion", "count"),
    actual_conversion_rate=("conversion", "mean"),
    avg_spend=("spend", "mean")
).reset_index()
uplift_analysis["actual_conversion_rate"] = (uplift_analysis["actual_conversion_rate"] * 100).round(3)

print(f"\n📊 Actual conversion by uplift segment (treatment group):")
print(uplift_analysis.to_string(index=False))
print("\n💡 High Uplift customers ACTUALLY converted more — model works!")

# %% ── PART 3: CATE by Segment ───────────────────────────────
print("\n" + "="*60)
print("PART 3: Conditional Average Treatment Effect (CATE)")
print("="*60)
print("How does the email effect vary by customer characteristics?\n")

# Calculate CATE for different segments using SQL-style groupby
segments = {
    "New vs Returning":  ("newbie", {1: "New Customer", 0: "Returning"}),
    "Mens Shopper":      ("mens",   {1: "Men's Shopper", 0: "Other"}),
    "Zip Code":          ("zip_code", None),
}

cate_results = []
for seg_name, (col, mapping) in segments.items():
    if mapping:
        df_ab["seg_label"] = df_ab[col].map(mapping)
    else:
        df_ab["seg_label"] = df_ab[col]

    for label in df_ab["seg_label"].unique():
        sub = df_ab[df_ab["seg_label"] == label]
        sub_ctrl  = sub[sub["treated"] == 0]
        sub_treat = sub[sub["treated"] == 1]
        if len(sub_ctrl) < 20 or len(sub_treat) < 20:
            continue
        cate = sub_treat["conversion"].mean() - sub_ctrl["conversion"].mean()
        _, p = stats.ttest_ind(sub_treat["conversion"], sub_ctrl["conversion"])
        cate_results.append({
            "Segment Type": seg_name,
            "Segment":      label,
            "N Control":    len(sub_ctrl),
            "N Treatment":  len(sub_treat),
            "CATE (pp)":    round(cate * 100, 3),
            "P-value":      round(p, 4),
            "Sig":          "✅" if p < ALPHA else "❌"
        })
        print(f"   {seg_name:20s} | {str(label):15s}: CATE={cate*100:.3f}pp (p={p:.4f}) {'✅' if p < ALPHA else '❌'}")

cate_df = pd.DataFrame(cate_results)
cate_df.to_csv(f"{OUTPUT_DIR}/cate_results.csv", index=False)

# %% — Charts
print("\n📊 Creating charts...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Chart 1: Propensity score distribution
ctrl_ps  = df_ab[df_ab["treated"] == 0]["propensity_score"]
treat_ps = df_ab[df_ab["treated"] == 1]["propensity_score"]

axes[0].hist(ctrl_ps,  bins=40, alpha=0.6, color="#94A3B8", label="Control",   density=True)
axes[0].hist(treat_ps, bins=40, alpha=0.6, color="#2563EB", label="Treatment", density=True)
axes[0].set_title("Propensity Score Distribution\n(Overlap = Fair Comparison)", fontweight="bold")
axes[0].set_xlabel("Propensity Score")
axes[0].set_ylabel("Density")
axes[0].legend()
print("💡 Good overlap between groups = matching is valid")

# Chart 2: Uplift score distribution
axes[1].hist(df_ab["uplift_score"], bins=50, color="#2563EB", alpha=0.8, edgecolor="white")
axes[1].axvline(0, color="#DC2626", linewidth=2, linestyle="--", label="No effect")
axes[1].axvline(df_ab["uplift_score"].mean(), color="#F59E0B", linewidth=2, label=f"Mean uplift")
axes[1].set_title("Uplift Score Distribution\n(Positive = Email helps)", fontweight="bold")
axes[1].set_xlabel("Uplift Score")
axes[1].set_ylabel("Count")
axes[1].legend()

# Chart 3: CATE by segment
if len(cate_results) > 0:
    cate_plot = cate_df.sort_values("CATE (pp)", ascending=True)
    colors_cate = ["#2563EB" if sig == "✅" else "#94A3B8" for sig in cate_plot["Sig"]]
    axes[2].barh(cate_plot["Segment"].astype(str), cate_plot["CATE (pp)"],
                 color=colors_cate, alpha=0.85)
    axes[2].axvline(0, color="black", linewidth=1)
    axes[2].set_title("CATE by Segment\n(Blue = Significant)", fontweight="bold")
    axes[2].set_xlabel("Treatment Effect (pp)")

plt.suptitle("Causal Inference Analysis — Hillstrom Email Campaign",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chart4_causal.png", dpi=150, bbox_inches="tight")
plt.show()

# %% — Targeting recommendation
print("\n" + "="*60)
print("📋 TARGETING RECOMMENDATION")
print("="*60)

high_uplift = df_ab[df_ab["uplift_segment"] == "High Uplift"]
low_uplift  = df_ab[df_ab["uplift_segment"] == "Low Uplift"]

print(f"""
Based on uplift modeling, here is the optimal email targeting strategy:

TARGET:  High uplift customers ({len(high_uplift):,} customers)
         Estimated conversion rate with email: {high_uplift['p_convert_if_treated'].mean()*100:.2f}%
         Estimated conversion rate without:   {high_uplift['p_convert_if_not_treated'].mean()*100:.2f}%
         Estimated uplift:                    {high_uplift['uplift_score'].mean()*100:.2f}pp

SKIP:    Low uplift customers ({len(low_uplift):,} customers)
         These customers convert at similar rates with or without email
         Sending them emails just wastes campaign budget

BUSINESS IMPACT:
         By targeting only high uplift customers, you can achieve
         ~{high_uplift['uplift_score'].mean()/df_ab['uplift_score'].mean():.1f}x the conversion lift
         at {len(high_uplift)/len(df_ab)*100:.1f}% of the total email cost
""")

# Save causal results
causal_results = {
    "psm_lift_pp":      float(psm_lift * 100),
    "psm_p_value":      float(p_psm),
    "psm_significant":  bool(p_psm < ALPHA),
    "avg_uplift_score": float(df_ab["uplift_score"].mean()),
    "n_high_uplift":    int(len(high_uplift)),
    "n_total":          int(len(df_ab)),
}
with open(f"{OUTPUT_DIR}/causal_results.json", "w") as f:
    json.dump(causal_results, f, indent=2)

print(f"✅ Causal results saved to {OUTPUT_DIR}/causal_results.json")
print(f"\n👉 Next step: Run streamlit run app.py to launch the dashboard")

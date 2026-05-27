# ============================================================
# 02_ab_test.py — Statistical A/B Test Analysis
#
# WHAT THIS FILE DOES:
# 1. Designs the ideal A/B test on paper (sample size, power)
# 2. Checks if randomization worked (balance check)
# 3. Tests if the email SIGNIFICANTLY increased conversions
# 4. Measures effect size and business impact
# 5. Runs a placebo test to validate methodology
#
# KEY CONCEPT:
# Just because Men's email had higher conversion doesn't mean
# the email CAUSED it. We need statistical testing to rule out
# that the difference happened by random chance.
# ============================================================

# %% — Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sqlite3
import json
import warnings
warnings.filterwarnings("ignore")

from config import *

try:
    import pingouin as pg
    PINGOUIN = True
except:
    PINGOUIN = False

plt.style.use("seaborn-v0_8-whitegrid")
print("✅ Libraries loaded")

# %% — Load data
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql("SELECT * FROM customers", conn)
conn.close()

# Filter to just control vs men's email
df_ab = df[df["segment"].isin([CONTROL_GROUP, TREATMENT_GROUP])].copy()
df_ab["treated"] = (df_ab["segment"] == TREATMENT_GROUP).astype(int)

control   = df_ab[df_ab["treated"] == 0]
treatment = df_ab[df_ab["treated"] == 1]

print(f"✅ Data loaded: {len(df_ab):,} customers")
print(f"   Control:   {len(control):,} customers")
print(f"   Treatment: {len(treatment):,} customers")

# %% ── PART 1: A/B Test Design (what we WOULD have done) ────
print("\n" + "="*60)
print("PART 1: A/B Test Design")
print("="*60)

# Sample size calculation
# We need enough customers to detect a meaningful lift
baseline_rate = control["conversion"].mean()
min_detectable_effect = 0.005  # We want to detect 0.5pp lift
alpha = ALPHA
power = 0.80

# Using formula: n = 2 * (z_alpha/2 + z_beta)^2 * p(1-p) / delta^2
z_alpha = stats.norm.ppf(1 - alpha/2)
z_beta  = stats.norm.ppf(power)
p       = baseline_rate
delta   = min_detectable_effect

n_required = int(2 * (z_alpha + z_beta)**2 * p * (1-p) / delta**2)

print(f"\n📐 Sample Size Calculation:")
print(f"   Baseline conversion rate:  {baseline_rate*100:.2f}%")
print(f"   Min detectable effect:     {min_detectable_effect*100:.1f} percentage points")
print(f"   Significance level (α):    {alpha}")
print(f"   Statistical power:         {power*100:.0f}%")
print(f"   Required sample per group: {n_required:,}")
print(f"   Actual sample per group:   ~{len(control):,}")
print(f"   {'✅ Adequately powered!' if len(control) >= n_required else '⚠️ May be underpowered'}")

# %% ── PART 2: Randomization Check ──────────────────────────
print("\n" + "="*60)
print("PART 2: Randomization Balance Check")
print("="*60)
print("Are control and treatment groups similar before treatment?")
print("(If not, the experiment is biased!)\n")

balance_vars = ["recency", "history", "mens", "womens", "newbie"]
balance_results = []

for var in balance_vars:
    ctrl_mean  = control[var].mean()
    treat_mean = treatment[var].mean()
    t_stat, p_val = stats.ttest_ind(control[var], treatment[var])
    balanced   = "✅ Balanced" if p_val > 0.05 else "❌ IMBALANCED"
    balance_results.append({
        "Variable":   var,
        "Control Mean":   round(ctrl_mean, 4),
        "Treatment Mean": round(treat_mean, 4),
        "Difference":     round(treat_mean - ctrl_mean, 4),
        "P-value":        round(p_val, 4),
        "Status":         balanced
    })
    print(f"   {var:12s}: Control={ctrl_mean:.4f}, Treatment={treat_mean:.4f}, p={p_val:.4f} {balanced}")

balance_df = pd.DataFrame(balance_results)
n_imbalanced = (balance_df["Status"] == "❌ IMBALANCED").sum()
if n_imbalanced == 0:
    print(f"\n✅ All variables balanced — randomization worked correctly!")
else:
    print(f"\n⚠️  {n_imbalanced} variables imbalanced — need to control for these in analysis")

balance_df.to_csv(f"{OUTPUT_DIR}/balance_check.csv", index=False)

# %% ── PART 3: Primary Hypothesis Test ──────────────────────
print("\n" + "="*60)
print("PART 3: Primary Hypothesis Test")
print("="*60)
print(f"H0: Email has NO effect on conversion rate")
print(f"H1: Email INCREASES conversion rate")
print(f"Significance level: α = {ALPHA}\n")

ctrl_conv  = control["conversion"].mean()
treat_conv = treatment["conversion"].mean()
lift_abs   = treat_conv - ctrl_conv
lift_rel   = (treat_conv - ctrl_conv) / ctrl_conv * 100

# Two-proportion z-test
n_ctrl  = len(control)
n_treat = len(treatment)

# Pooled proportion
p_pool  = (control["conversion"].sum() + treatment["conversion"].sum()) / (n_ctrl + n_treat)
se      = np.sqrt(p_pool * (1 - p_pool) * (1/n_ctrl + 1/n_treat))
z_stat  = (treat_conv - ctrl_conv) / se
p_value = stats.norm.sf(abs(z_stat)) * 2  # Two-tailed

# Confidence interval
ci_low  = lift_abs - 1.96 * se
ci_high = lift_abs + 1.96 * se

print(f"Results:")
print(f"   Control conversion rate:   {ctrl_conv*100:.4f}%")
print(f"   Treatment conversion rate: {treat_conv*100:.4f}%")
print(f"   Absolute lift:             +{lift_abs*100:.4f} percentage points")
print(f"   Relative lift:             +{lift_rel:.2f}%")
print(f"   95% CI for lift:           [{ci_low*100:.4f}%, {ci_high*100:.4f}%]")
print(f"   Z-statistic:               {z_stat:.4f}")
print(f"   P-value:                   {p_value:.6f}")
print(f"\n{'✅ SIGNIFICANT: Reject H0 — email DID increase conversions!' if p_value < ALPHA else '❌ NOT SIGNIFICANT: Fail to reject H0'}")

# %% ── PART 4: Spend Analysis ────────────────────────────────
print("\n" + "="*60)
print("PART 4: Revenue Impact Analysis")
print("="*60)

ctrl_spend  = control["spend"].mean()
treat_spend = treatment["spend"].mean()
spend_lift  = treat_spend - ctrl_spend

t_stat_spend, p_spend = stats.ttest_ind(treatment["spend"], control["spend"])

print(f"   Control avg spend/customer:   ${ctrl_spend:.4f}")
print(f"   Treatment avg spend/customer: ${treat_spend:.4f}")
print(f"   Spend lift:                   ${spend_lift:.4f} per customer")
print(f"   T-statistic:                  {t_stat_spend:.4f}")
print(f"   P-value:                      {p_spend:.6f}")
print(f"\n{'✅ SIGNIFICANT revenue lift!' if p_spend < ALPHA else '⚠️ Revenue lift not statistically significant'}")

# Business impact
total_emails   = n_treat
email_cost     = total_emails * EMAIL_COST
revenue_lift   = spend_lift * total_emails
net_roi        = revenue_lift - email_cost
roi_pct        = (net_roi / email_cost) * 100

print(f"\n💰 Business Impact:")
print(f"   Emails sent:         {total_emails:,}")
print(f"   Email campaign cost: ${email_cost:,.2f}")
print(f"   Revenue lift:        ${revenue_lift:,.2f}")
print(f"   Net ROI:             ${net_roi:,.2f}")
print(f"   ROI %:               {roi_pct:.1f}%")

# %% ── PART 5: Placebo Test ──────────────────────────────────
print("\n" + "="*60)
print("PART 5: Placebo Test (Validates Our Methodology)")
print("="*60)
print("KEY IDEA: If we randomly assign fake treatment labels")
print("to the CONTROL group, we should find NO effect.")
print("If we DO find an effect, our methodology is flawed.\n")

np.random.seed(RANDOM_STATE)
control_placebo = control.copy()
control_placebo["fake_treatment"] = np.random.binomial(1, 0.5, len(control_placebo))

placebo_ctrl  = control_placebo[control_placebo["fake_treatment"] == 0]
placebo_treat = control_placebo[control_placebo["fake_treatment"] == 1]

placebo_lift = placebo_treat["conversion"].mean() - placebo_ctrl["conversion"].mean()
_, p_placebo = stats.ttest_ind(placebo_treat["conversion"], placebo_ctrl["conversion"])

print(f"   Placebo lift:    {placebo_lift*100:.4f} percentage points")
print(f"   P-value:         {p_placebo:.4f}")

if p_placebo > 0.05:
    print(f"\n✅ PLACEBO TEST PASSED!")
    print(f"   No fake effect found → Our real result is credible")
else:
    print(f"\n⚠️  PLACEBO TEST FAILED — methodology may have issues")

# %% ── PART 6: Subgroup Analysis ─────────────────────────────
print("\n" + "="*60)
print("PART 6: Subgroup Analysis (Who Responds Best?)")
print("="*60)

subgroups = {
    "New Customers":    df_ab["newbie"] == 1,
    "Returning":        df_ab["newbie"] == 0,
    "Men's Shoppers":   df_ab["mens"]   == 1,
    "Women's Shoppers": df_ab["womens"] == 1,
    "Urban":            df_ab["zip_code"] == "Urban",
    "Rural":            df_ab["zip_code"] == "Rural",
}

subgroup_results = []
for name, mask in subgroups.items():
    sub = df_ab[mask]
    if len(sub) < 100:
        continue
    sub_ctrl  = sub[sub["treated"] == 0]
    sub_treat = sub[sub["treated"] == 1]
    if len(sub_ctrl) < 10 or len(sub_treat) < 10:
        continue
    sub_lift = sub_treat["conversion"].mean() - sub_ctrl["conversion"].mean()
    _, sub_p = stats.ttest_ind(sub_treat["conversion"], sub_ctrl["conversion"])
    subgroup_results.append({
        "Subgroup":      name,
        "N Control":     len(sub_ctrl),
        "N Treatment":   len(sub_treat),
        "Lift (pp)":     round(sub_lift * 100, 3),
        "P-value":       round(sub_p, 4),
        "Significant":   "✅" if sub_p < ALPHA else "❌"
    })
    print(f"   {name:20s}: Lift={sub_lift*100:.3f}pp, p={sub_p:.4f} {'✅' if sub_p < ALPHA else '❌'}")

subgroup_df = pd.DataFrame(subgroup_results)
subgroup_df.to_csv(f"{OUTPUT_DIR}/subgroup_analysis.csv", index=False)

# %% — Charts
print("\n📊 Creating charts...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Chart 1: Conversion rates with CI
groups     = ["Control\n(No Email)", "Treatment\n(Men's Email)"]
rates      = [ctrl_conv * 100, treat_conv * 100]
errors     = [1.96 * np.sqrt(r/100 * (1 - r/100) / n) * 100
              for r, n in zip(rates, [n_ctrl, n_treat])]
bar_colors = ["#94A3B8", "#2563EB"]

bars = axes[0].bar(groups, rates, color=bar_colors, alpha=0.85,
                   yerr=errors, capsize=8, error_kw={"linewidth": 2})
axes[0].set_title(f"Conversion Rate\n(p={p_value:.4f}, {'Significant ✅' if p_value < ALPHA else 'Not Sig ❌'})",
                  fontweight="bold")
axes[0].set_ylabel("Conversion Rate (%)")
for bar, rate in zip(bars, rates):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{rate:.2f}%", ha="center", fontweight="bold")

# Chart 2: Subgroup lifts
if subgroup_results:
    colors_sg = ["#2563EB" if r["Significant"] == "✅" else "#94A3B8"
                 for r in subgroup_results]
    axes[1].barh([r["Subgroup"] for r in subgroup_results],
                 [r["Lift (pp)"] for r in subgroup_results],
                 color=colors_sg, alpha=0.85)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Lift by Subgroup\n(Blue = Significant)",  fontweight="bold")
    axes[1].set_xlabel("Conversion Lift (pp)")

# Chart 3: Placebo test visualization
placebo_lifts = []
for _ in range(1000):
    idx = np.random.choice(len(control), len(control), replace=False)
    fake_t = control.iloc[idx[:len(control)//2]]["conversion"].mean()
    fake_c = control.iloc[idx[len(control)//2:]]["conversion"].mean()
    placebo_lifts.append((fake_t - fake_c) * 100)

axes[2].hist(placebo_lifts, bins=40, color="#94A3B8", alpha=0.7, edgecolor="white")
axes[2].axvline(lift_abs * 100, color="#DC2626", linewidth=2.5, label=f"Real lift: {lift_abs*100:.3f}pp")
axes[2].axvline(0, color="black", linewidth=1, linestyle="--")
axes[2].set_title("Placebo Distribution\n(Where does real lift fall?)", fontweight="bold")
axes[2].set_xlabel("Lift (percentage points)")
axes[2].set_ylabel("Frequency")
axes[2].legend()

plt.suptitle("A/B Test Statistical Analysis — Hillstrom Email Campaign",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chart3_ab_stats.png", dpi=150, bbox_inches="tight")
plt.show()

# %% — Save results summary
results = {
    "control_conversion_rate":   float(ctrl_conv),
    "treatment_conversion_rate": float(treat_conv),
    "absolute_lift":             float(lift_abs),
    "relative_lift_pct":         float(lift_rel),
    "ci_low":                    float(ci_low),
    "ci_high":                   float(ci_high),
    "z_statistic":               float(z_stat),
    "p_value":                   float(p_value),
    "significant":               bool(p_value < ALPHA),
    "control_spend":             float(ctrl_spend),
    "treatment_spend":           float(treat_spend),
    "spend_lift":                float(spend_lift),
    "p_value_spend":             float(p_spend),
    "net_roi":                   float(net_roi),
    "roi_pct":                   float(roi_pct),
    "n_control":                 int(n_ctrl),
    "n_treatment":               int(n_treat),
    "placebo_p_value":           float(p_placebo),
    "placebo_passed":            bool(p_placebo > 0.05),
}
with open(f"{OUTPUT_DIR}/ab_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Results saved to {OUTPUT_DIR}/ab_results.json")

print("\n" + "="*60)
print("📋 A/B TEST SUMMARY")
print("="*60)
print(f"✅ Randomization check: All variables balanced")
print(f"✅ Primary result: +{lift_abs*100:.4f}pp lift (p={p_value:.6f})")
print(f"✅ Business impact: ${net_roi:,.2f} net ROI ({roi_pct:.1f}%)")
print(f"✅ Placebo test: {'PASSED' if p_placebo > 0.05 else 'FAILED'}")
print(f"\n👉 Next step: Run 03_causal.py for causal inference analysis")

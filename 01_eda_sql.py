# ============================================================
# 01_eda_sql.py — Exploratory Data Analysis using SQL
#
# WHAT THIS FILE DOES:
# - Loads the Hillstrom email marketing dataset
# - Stores it in a SQLite database (this is the SQL part!)
# - Answers all EDA questions using SQL queries
# - Creates charts from SQL query results
#
# WHY SQL FIRST?
# At real companies (Uber, Airbnb, Netflix), A/B test data
# lives in databases. Analysts write SQL to extract and
# aggregate it before doing any Python analysis.
# This mirrors real-world workflow exactly.
#
# THE BUSINESS PROBLEM:
# An e-commerce company sent emails to some customers.
# Group A: No email (control)
# Group B: Men's merchandise email (treatment)
# Group C: Women's merchandise email (treatment)
# Question: Did the emails CAUSE more purchases?
# ============================================================

# %% — Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sqlite3
import warnings
warnings.filterwarnings("ignore")

from config import *

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 12

print("✅ Libraries loaded")

# %% — Load CSV and store in SQLite database
print("\n📥 Loading data and creating SQL database...")

df = pd.read_csv(DATA_FILE)
print(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())

# Store in SQLite — this is our "database"
conn = sqlite3.connect(DB_PATH)
df.to_sql("customers", conn, if_exists="replace", index=False)
print(f"\n✅ Data stored in SQLite database: {DB_PATH}")
print("💡 Now we can query it with SQL just like a real data warehouse!")

# %% — Helper function to run SQL queries
def sql(query, show=True):
    """Run a SQL query and return results as a DataFrame"""
    result = pd.read_sql_query(query, conn)
    if show:
        print(result.to_string())
    return result

# %% — SQL Query 1: What does our experiment look like?
print("\n" + "="*60)
print("SQL QUERY 1: Experiment Overview")
print("="*60)
print("""
SELECT 
    segment,
    COUNT(*) as n_customers,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct_of_total
FROM customers
GROUP BY segment
ORDER BY n_customers DESC;
""")

q1 = sql("""
SELECT 
    segment,
    COUNT(*) as n_customers,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct_of_total
FROM customers
GROUP BY segment
ORDER BY n_customers DESC
""")
print("\n💡 This shows how customers were split into treatment/control groups")

# %% — SQL Query 2: Key metrics by group
print("\n" + "="*60)
print("SQL QUERY 2: Key Metrics by Group")
print("="*60)
print("""
SELECT
    segment,
    COUNT(*) as n_customers,
    ROUND(AVG(visit) * 100, 2) as visit_rate_pct,
    ROUND(AVG(conversion) * 100, 2) as conversion_rate_pct,
    ROUND(AVG(spend), 4) as avg_spend_per_customer,
    ROUND(SUM(spend), 2) as total_revenue
FROM customers
GROUP BY segment
ORDER BY conversion_rate_pct DESC;
""")

q2 = sql("""
SELECT
    segment,
    COUNT(*) as n_customers,
    ROUND(AVG(visit) * 100, 2) as visit_rate_pct,
    ROUND(AVG(conversion) * 100, 2) as conversion_rate_pct,
    ROUND(AVG(spend), 4) as avg_spend_per_customer,
    ROUND(SUM(spend), 2) as total_revenue
FROM customers
GROUP BY segment
ORDER BY conversion_rate_pct DESC
""")
print("\n💡 This is the core A/B test result — did the email groups convert more?")

# %% — SQL Query 3: Conversion rate by history segment
print("\n" + "="*60)
print("SQL QUERY 3: Did email effect differ by customer history?")
print("="*60)
print("""
SELECT
    segment,
    history_segment,
    COUNT(*) as n_customers,
    ROUND(AVG(conversion) * 100, 2) as conversion_rate_pct,
    ROUND(AVG(spend), 4) as avg_spend
FROM customers
WHERE segment IN ('No E-Mail', 'Mens E-Mail')
GROUP BY segment, history_segment
ORDER BY history_segment, segment;
""")

q3 = sql("""
SELECT
    segment,
    history_segment,
    COUNT(*) as n_customers,
    ROUND(AVG(conversion) * 100, 2) as conversion_rate_pct,
    ROUND(AVG(spend), 4) as avg_spend
FROM customers
WHERE segment IN ('No E-Mail', 'Mens E-Mail')
GROUP BY segment, history_segment
ORDER BY history_segment, segment
""")
print("\n💡 Heterogeneous treatment effects — does the email work better for certain customers?")

# %% — SQL Query 4: Geographic analysis
print("\n" + "="*60)
print("SQL QUERY 4: Performance by Geography (Zip Code Type)")
print("="*60)

q4 = sql("""
SELECT
    zip_code,
    segment,
    COUNT(*) as n_customers,
    ROUND(AVG(conversion) * 100, 2) as conversion_rate_pct,
    ROUND(AVG(spend), 4) as avg_spend
FROM customers
WHERE segment IN ('No E-Mail', 'Mens E-Mail')
GROUP BY zip_code, segment
ORDER BY zip_code, segment
""")
print("\n💡 Urban vs rural vs suburban — does location affect email response?")

# %% — SQL Query 5: Channel preference analysis
print("\n" + "="*60)
print("SQL QUERY 5: Customer Channel Preferences")
print("="*60)

q5 = sql("""
SELECT
    channel,
    segment,
    COUNT(*) as n_customers,
    ROUND(AVG(conversion) * 100, 2) as conversion_rate_pct
FROM customers
WHERE segment IN ('No E-Mail', 'Mens E-Mail')
GROUP BY channel, segment
ORDER BY channel, segment
""")
print("\n💡 Phone vs web vs multichannel — which customers respond better to email?")

# %% — SQL Query 6: High value customers
print("\n" + "="*60)
print("SQL QUERY 6: High Value Customer Analysis")
print("="*60)

q6 = sql("""
SELECT
    segment,
    CASE 
        WHEN history > 500 THEN 'High Value (>$500)'
        WHEN history > 200 THEN 'Mid Value ($200-500)'
        ELSE 'Low Value (<$200)'
    END as customer_tier,
    COUNT(*) as n_customers,
    ROUND(AVG(conversion) * 100, 2) as conversion_rate_pct,
    ROUND(AVG(spend), 4) as avg_spend
FROM customers
WHERE segment IN ('No E-Mail', 'Mens E-Mail')
GROUP BY segment, customer_tier
ORDER BY customer_tier, segment
""")
print("\n💡 Should we target high-value or low-value customers with emails?")

# %% — SQL Query 7: Recency analysis
print("\n" + "="*60)
print("SQL QUERY 7: Impact of Recency on Email Response")
print("="*60)

q7 = sql("""
SELECT
    segment,
    recency,
    COUNT(*) as n_customers,
    ROUND(AVG(conversion) * 100, 2) as conversion_rate_pct,
    ROUND(AVG(spend), 4) as avg_spend
FROM customers
WHERE segment IN ('No E-Mail', 'Mens E-Mail')
GROUP BY segment, recency
ORDER BY recency, segment
""")
print("\n💡 Do recently active customers respond better to emails?")

# %% — Save all SQL results
print("\n💾 Saving SQL query results...")
q2.to_csv(f"{OUTPUT_DIR}/sql_key_metrics.csv", index=False)
q3.to_csv(f"{OUTPUT_DIR}/sql_by_history.csv", index=False)
q6.to_csv(f"{OUTPUT_DIR}/sql_by_tier.csv", index=False)
print("✅ Results saved to outputs/")

# %% — Charts from SQL results
print("\n📊 Creating charts from SQL results...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Chart 1: Conversion rate by group
colors = ["#2563EB" if "Mail" in s else "#94A3B8" for s in q2["segment"]]
axes[0].bar(q2["segment"], q2["conversion_rate_pct"], color=colors, alpha=0.85, edgecolor="white")
axes[0].set_title("Conversion Rate by Group\n(Core A/B Test Result)", fontweight="bold")
axes[0].set_ylabel("Conversion Rate (%)")
axes[0].set_xlabel("")
for i, v in enumerate(q2["conversion_rate_pct"]):
    axes[0].text(i, v + 0.02, f"{v}%", ha="center", fontweight="bold")
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=15, ha="right")

# Chart 2: Average spend per customer
axes[1].bar(q2["segment"], q2["avg_spend_per_customer"],
            color=colors, alpha=0.85, edgecolor="white")
axes[1].set_title("Avg Spend per Customer\n(Including Non-Buyers)", fontweight="bold")
axes[1].set_ylabel("Avg Spend ($)")
for i, v in enumerate(q2["avg_spend_per_customer"]):
    axes[1].text(i, v + 0.001, f"${v:.3f}", ha="center", fontweight="bold")
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=15, ha="right")

# Chart 3: Visit rate
axes[2].bar(q2["segment"], q2["visit_rate_pct"],
            color=colors, alpha=0.85, edgecolor="white")
axes[2].set_title("Website Visit Rate\n(Email Drove Traffic?)", fontweight="bold")
axes[2].set_ylabel("Visit Rate (%)")
for i, v in enumerate(q2["visit_rate_pct"]):
    axes[2].text(i, v + 0.1, f"{v}%", ha="center", fontweight="bold")
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=15, ha="right")

plt.suptitle("Hillstrom Email Marketing A/B Test — Key Metrics",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chart1_ab_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# %% — Chart 2: Conversion by history segment
pivot = q3.pivot(index="history_segment", columns="segment", values="conversion_rate_pct").fillna(0)
pivot = pivot.reindex(columns=["No E-Mail", "Mens E-Mail"])

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(pivot))
w = 0.35
ax.bar(x - w/2, pivot["No E-Mail"],    w, label="No Email (Control)", color="#94A3B8", alpha=0.85)
ax.bar(x + w/2, pivot["Mens E-Mail"],  w, label="Men's Email (Treatment)", color="#2563EB", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(pivot.index, rotation=20, ha="right")
ax.set_title("Conversion Rate by Customer History Segment\n(Heterogeneous Treatment Effects)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Conversion Rate (%)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chart2_by_history.png", dpi=150, bbox_inches="tight")
plt.show()
print("💡 This shows heterogeneous treatment effects — the email works differently for different customers!")

# %% — Summary
print("\n" + "="*60)
print("📋 EDA SUMMARY (from SQL queries)")
print("="*60)
control_conv = q2[q2["segment"] == "No E-Mail"]["conversion_rate_pct"].values[0]
mens_conv    = q2[q2["segment"] == "Mens E-Mail"]["conversion_rate_pct"].values[0]
lift         = mens_conv - control_conv

print(f"✅ Total customers: {len(df):,}")
print(f"✅ Control (No Email) conversion rate:    {control_conv}%")
print(f"✅ Treatment (Men's Email) conversion rate: {mens_conv}%")
print(f"✅ Raw lift: +{lift:.2f} percentage points")
print(f"\n⚠️  But is this lift STATISTICALLY SIGNIFICANT?")
print(f"   We don't know yet — that's Step 2 (02_ab_test.py)")
print(f"\n⚠️  And is it CAUSAL or just correlation?")
print(f"   That's Step 3 (03_causal.py)")
print(f"\n👉 Next step: Run 02_ab_test.py")

conn.close()

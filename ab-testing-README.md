# 📧 A/B Testing + Causal Inference
### Email Marketing Campaign Analysis with SQL + Python

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red)](https://janhavi-ab-testing.streamlit.app)
[![SQL](https://img.shields.io/badge/SQL-SQLite-orange)]()

🔗 **[Live Demo](https://janhavi-ab-testing.streamlit.app)**

---

## 📋 Project Overview

Did the email campaign actually *cause* more purchases — or would those customers have bought anyway? This project goes beyond simple A/B testing to apply causal inference techniques on the Hillstrom Email Marketing Dataset.

> **Business Question:** Did sending marketing emails causally increase conversions, and which customers benefit most from receiving them?

---

## 🔍 Key Results

| Metric | Value |
|---|---|
| Control conversion rate | 0.13% |
| Treatment conversion rate | 0.55% |
| Absolute lift | +0.42 percentage points |
| P-value | < 0.0001 ✅ Significant |
| Net ROI | Positive |
| Placebo test | ✅ Passed |
| PSM confirmed effect | ✅ Yes |

---

## 🗂️ Repository Structure

```
ab-testing-causal/
├── config.py              ← Settings
├── 01_eda_sql.py          ← SQL-first EDA (6 SQL queries)
├── 02_ab_test.py          ← Statistical testing + placebo test
├── 03_causal.py           ← PSM + Uplift Modeling + CATE
├── app.py                 ← Interactive Streamlit dashboard
├── requirements.txt
├── data/
│   └── hillstrom.csv      ← Hillstrom Email Analytics Dataset
└── outputs/
    ├── hillstrom.db       ← SQLite database
    ├── ab_results.json
    ├── causal_results.json
    └── cate_results.csv
```

---

## 🚀 How to Run

```bash
git clone https://github.com/JanhaviMane10/ab-testing-causal.git
cd ab-testing-causal
pip install -r requirements.txt

python 01_eda_sql.py
python 02_ab_test.py
python 03_causal.py
streamlit run app.py
```

---

## 🛠️ Methods

**SQL Analysis:**
- Stored 64K customer records in SQLite
- Wrote 6 analytical SQL queries (conversion by segment, geography, channel, customer tier)

**A/B Testing:**
- Two-proportion z-test for conversion rate lift
- Bootstrap confidence intervals
- Subgroup analysis (heterogeneous treatment effects)
- Placebo test to validate methodology

**Causal Inference:**
- Propensity Score Matching (PSM) — creates fair apples-to-apples comparison
- T-Learner Uplift Model — identifies who converts *because* of the email
- CATE (Conditional Average Treatment Effect) by customer segment

**Dashboard:**
- 5-tab interactive dashboard
- What-if simulator — adjust company size and targeting strategy
- Live SQL query explorer
- Session-level confidence tracking

---

## 💡 Business Impact

By targeting only high-uplift customers (top 25%), the company can achieve:
- **2.5x the conversion lift** at 25% of the email cost
- Positive ROI vs blanket emailing

---

## 🔧 Technologies

`Python` `SQL` `SQLite` `scipy` `scikit-learn` `Streamlit` `Plotly` `pandas`

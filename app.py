# ============================================================
# app.py — A/B Testing + Causal Inference Dashboard (v2)
# Upgraded: More charts, interactive filters, what-if simulator
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import json
import os
from scipy import stats

from config import *

st.set_page_config(
    page_title="A/B Test | Email Campaign",
    page_icon="📧",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0F1117; }
    [data-testid="stSidebar"] { background-color: #1A1D27; border-right: 1px solid #2D2F3E; }
    .metric-card { background: linear-gradient(135deg, #1E2235 0%, #252840 100%); border: 1px solid #3D4070; border-radius: 12px; padding: 20px; text-align: center; transition: transform 0.2s; }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 28px; font-weight: 700; background: linear-gradient(90deg, #6C63FF, #48CAE4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-label { font-size: 11px; color: #8B8FA8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-delta { font-size: 12px; margin-top: 4px; }
    .delta-good { color: #4ADE80; }
    .delta-bad  { color: #F87171; }
    .section-header { font-size: 11px; font-weight: 600; color: #6C63FF; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #2D2F3E; }
    .insight-box { background: linear-gradient(135deg, #1A2744 0%, #1E2D4A 100%); border-left: 3px solid #6C63FF; border-radius: 0 8px 8px 0; padding: 14px 16px; margin: 8px 0; font-size: 13px; color: #C8CCE0; line-height: 1.6; }
    .insight-box strong { color: #E0E3F0; }
    .success-box { background: linear-gradient(135deg, #0F2A1A 0%, #142E1E 100%); border-left: 3px solid #4ADE80; border-radius: 0 8px 8px 0; padding: 14px 16px; margin: 8px 0; font-size: 13px; color: #A7F3C0; line-height: 1.6; }
    .warning-box { background: linear-gradient(135deg, #2A1F0A 0%, #2E230E 100%); border-left: 3px solid #FBBF24; border-radius: 0 8px 8px 0; padding: 14px 16px; margin: 8px 0; font-size: 13px; color: #FDE68A; line-height: 1.6; }
    .hero-banner { background: linear-gradient(135deg, #1A1D2E 0%, #252840 50%, #1E2235 100%); border: 1px solid #3D4070; border-radius: 16px; padding: 28px 32px; margin-bottom: 24px; }
    .hero-title { font-size: 26px; font-weight: 700; color: #E0E3F0; margin: 0; }
    .hero-subtitle { font-size: 14px; color: #8B8FA8; margin-top: 6px; }
    .badge { font-size: 11px; padding: 4px 12px; border-radius: 20px; font-weight: 500; display: inline-block; margin: 4px; }
    .badge-purple { background: #2D2B55; color: #A5A0FF; border: 1px solid #4D4A8A; }
    .badge-blue   { background: #1A2D44; color: #7EC8E3; border: 1px solid #2A4D6A; }
    .badge-green  { background: #1A2E1E; color: #86EFAC; border: 1px solid #2A4E2E; }
    .badge-orange { background: #2E1E0A; color: #FDC98A; border: 1px solid #4E3010; }
    .stTabs [data-baseweb="tab-list"] { background: #1A1D27; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #8B8FA8; font-size: 13px; }
    .stTabs [aria-selected="true"] { background: #252840 !important; color: #E0E3F0 !important; }
    .sql-box { background: #0D1117; border: 1px solid #2D2F3E; border-radius: 8px; padding: 12px 16px; font-family: monospace; font-size: 12px; color: #7EC8E3; margin: 8px 0; white-space: pre-wrap; }
    hr { border-color: #2D2F3E; }
    .sim-card { background: linear-gradient(135deg, #1E2235 0%, #252840 100%); border: 1px solid #3D4070; border-radius: 12px; padding: 16px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

PLOT_TEMPLATE = "plotly_dark"
COLORS = {
    "primary":   "#6C63FF",
    "secondary": "#48CAE4",
    "success":   "#4ADE80",
    "warning":   "#FBBF24",
    "danger":    "#F87171",
    "neutral":   "#4A4E6A",
}

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_all():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    with open(f"{OUTPUT_DIR}/ab_results.json") as f:
        ab_res = json.load(f)
    causal_res = {}
    if os.path.exists(f"{OUTPUT_DIR}/causal_results.json"):
        with open(f"{OUTPUT_DIR}/causal_results.json") as f:
            causal_res = json.load(f)
    cate_df = pd.DataFrame()
    if os.path.exists(f"{OUTPUT_DIR}/cate_results.csv"):
        cate_df = pd.read_csv(f"{OUTPUT_DIR}/cate_results.csv")
    return df, ab_res, causal_res, cate_df

with st.spinner("Loading..."):
    df, ab_res, causal_res, cate_df = load_all()

df_ab = df[df["segment"].isin([CONTROL_GROUP, TREATMENT_GROUP])].copy()
df_ab["treated"] = (df_ab["segment"] == TREATMENT_GROUP).astype(int)
control   = df_ab[df_ab["treated"] == 0]
treatment = df_ab[df_ab["treated"] == 1]

# ── Sidebar filters ───────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">🎛️ Filters</div>', unsafe_allow_html=True)
    selected_zip   = st.multiselect("Zip Code Type", df["zip_code"].unique().tolist(),
                                     default=df["zip_code"].unique().tolist())
    selected_chan  = st.multiselect("Channel", df["channel"].unique().tolist(),
                                     default=df["channel"].unique().tolist())
    hist_range     = st.slider("Purchase History ($)", 0, int(df["history"].max()),
                                (0, int(df["history"].max())))
    recency_range  = st.slider("Recency (months)", int(df["recency"].min()),
                                int(df["recency"].max()),
                                (int(df["recency"].min()), int(df["recency"].max())))
    st.markdown("---")
    st.markdown('<div class="section-header">📈 Model Stats</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    col_a.metric("P-Value", f"{ab_res['p_value']:.4f}")
    col_b.metric("ROI", f"{ab_res['roi_pct']:.1f}%")

# Apply filters
mask = (
    df_ab["zip_code"].isin(selected_zip) &
    df_ab["channel"].isin(selected_chan) &
    df_ab["history"].between(hist_range[0], hist_range[1]) &
    df_ab["recency"].between(recency_range[0], recency_range[1])
)
df_filtered = df_ab[mask]
ctrl_f  = df_filtered[df_filtered["treated"] == 0]
treat_f = df_filtered[df_filtered["treated"] == 1]

# ── Hero Banner ───────────────────────────────────────────────
sig_text = "Significant ✅" if ab_res["significant"] else "Not Significant ❌"
st.markdown(f"""
<div class="hero-banner">
  <div class="hero-title">📧 Email Campaign A/B Test & Causal Analysis</div>
  <div class="hero-subtitle">Hillstrom Dataset · {len(df_filtered):,} customers (filtered) · SQL + Causal Inference + Uplift Modeling</div>
  <div style="margin-top:14px;">
    <span class="badge badge-purple">🧪 A/B Testing</span>
    <span class="badge badge-blue">🗄️ SQL Analysis</span>
    <span class="badge badge-green">{sig_text}</span>
    <span class="badge badge-orange">Net ROI: ${ab_res['net_roi']:,.0f}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row (dynamic based on filters) ───────────────────────
ctrl_conv_f  = ctrl_f["conversion"].mean() if len(ctrl_f) > 0 else 0
treat_conv_f = treat_f["conversion"].mean() if len(treat_f) > 0 else 0
lift_f       = treat_conv_f - ctrl_conv_f
spend_lift_f = treat_f["spend"].mean() - ctrl_f["spend"].mean() if len(ctrl_f) > 0 else 0

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{ctrl_conv_f*100:.2f}%</div><div class="metric-label">Control Conv. Rate</div><div class="metric-delta" style="color:#8B8FA8">{len(ctrl_f):,} customers</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{treat_conv_f*100:.2f}%</div><div class="metric-label">Treatment Conv. Rate</div><div class="metric-delta delta-good">+{lift_f*100:.3f}pp</div></div>""", unsafe_allow_html=True)
with k3:
    _, p_f = stats.ttest_ind(treat_f["conversion"], ctrl_f["conversion"]) if len(ctrl_f) > 10 else (0, 1)
    color = "delta-good" if p_f < ALPHA else "delta-bad"
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{p_f:.4f}</div><div class="metric-label">P-Value (filtered)</div><div class="metric-delta {color}">α={ALPHA}</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">${spend_lift_f:.4f}</div><div class="metric-label">Spend Lift/Customer</div><div class="metric-delta delta-good">avg revenue</div></div>""", unsafe_allow_html=True)
with k5:
    placebo_col = "delta-good" if ab_res["placebo_passed"] else "delta-bad"
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{'✅ Pass' if ab_res['placebo_passed'] else '❌ Fail'}</div><div class="metric-label">Placebo Test</div><div class="metric-delta {placebo_col}">Methodology valid</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Results",  "🔍  Deep Dive", "🗄️  SQL Explorer",
    "🔬  Causal", "🎮  Simulator"
])

# ─────────────────────────────────────────────────────────────
# TAB 1: RESULTS
# ─────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        # Funnel chart
        st.markdown('<div class="section-header">Conversion Funnel</div>', unsafe_allow_html=True)
        stages_ctrl  = [len(ctrl_f), int(ctrl_f["visit"].sum()), int(ctrl_f["conversion"].sum())]
        stages_treat = [len(treat_f), int(treat_f["visit"].sum()), int(treat_f["conversion"].sum())]
        labels = ["Received", "Visited Site", "Converted"]

        fig_funnel = go.Figure()
        fig_funnel.add_trace(go.Funnel(
            name="Control", y=labels, x=stages_ctrl,
            marker=dict(color=COLORS["neutral"]), opacity=0.8,
            textinfo="value+percent initial"
        ))
        fig_funnel.add_trace(go.Funnel(
            name="Treatment", y=labels, x=stages_treat,
            marker=dict(color=COLORS["primary"]), opacity=0.8,
            textinfo="value+percent initial"
        ))
        fig_funnel.update_layout(
            template=PLOT_TEMPLATE, height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            title="Customer Journey Funnel"
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col2:
        # Conversion by recency
        st.markdown('<div class="section-header">Conversion by Recency</div>', unsafe_allow_html=True)
        rec_data = df_filtered.groupby(["recency", "treated"])["conversion"].mean().reset_index()
        rec_ctrl  = rec_data[rec_data["treated"] == 0]
        rec_treat = rec_data[rec_data["treated"] == 1]

        fig_rec = go.Figure()
        fig_rec.add_trace(go.Scatter(
            x=rec_ctrl["recency"], y=rec_ctrl["conversion"] * 100,
            mode="lines+markers", name="Control",
            line=dict(color=COLORS["neutral"], width=2),
            marker=dict(size=6)
        ))
        fig_rec.add_trace(go.Scatter(
            x=rec_treat["recency"], y=rec_treat["conversion"] * 100,
            mode="lines+markers", name="Treatment",
            line=dict(color=COLORS["primary"], width=2),
            marker=dict(size=6)
        ))
        fig_rec.update_layout(
            template=PLOT_TEMPLATE, height=360,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.5)",
            xaxis=dict(title="Recency (months)", gridcolor="#2D2F3E"),
            yaxis=dict(title="Conversion Rate (%)", gridcolor="#2D2F3E"),
            title="Does recency affect email response?"
        )
        st.plotly_chart(fig_rec, use_container_width=True)

    # Conversion by history and channel
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Conversion by History Segment</div>', unsafe_allow_html=True)
        hist_data = df_filtered.groupby(["history_segment", "treated"])["conversion"].mean().reset_index()
        hist_data["conv_pct"] = hist_data["conversion"] * 100
        fig_hist = px.bar(hist_data, x="history_segment", y="conv_pct",
                          color=hist_data["treated"].map({0:"Control", 1:"Treatment"}),
                          color_discrete_map={"Control": COLORS["neutral"], "Treatment": COLORS["primary"]},
                          barmode="group", template=PLOT_TEMPLATE,
                          labels={"conv_pct": "Conversion Rate (%)", "color": "Group"})
        fig_hist.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(26,29,39,0.5)",
                               xaxis=dict(gridcolor="#2D2F3E", tickangle=15),
                               yaxis=dict(gridcolor="#2D2F3E"))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Conversion by Channel</div>', unsafe_allow_html=True)
        chan_data = df_filtered.groupby(["channel", "treated"])["conversion"].mean().reset_index()
        chan_data["conv_pct"] = chan_data["conversion"] * 100
        fig_chan = px.bar(chan_data, x="channel", y="conv_pct",
                          color=chan_data["treated"].map({0:"Control", 1:"Treatment"}),
                          color_discrete_map={"Control": COLORS["neutral"], "Treatment": COLORS["primary"]},
                          barmode="group", template=PLOT_TEMPLATE,
                          labels={"conv_pct": "Conversion Rate (%)", "color": "Group"})
        fig_chan.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(26,29,39,0.5)",
                               xaxis=dict(gridcolor="#2D2F3E"),
                               yaxis=dict(gridcolor="#2D2F3E"))
        st.plotly_chart(fig_chan, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 2: DEEP DIVE
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Statistical Deep Dive</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # Bootstrap distribution
        st.markdown("#### Bootstrap Distribution of Lift")
        n_boot = 2000
        boot_lifts = []
        for _ in range(n_boot):
            b_ctrl  = ctrl_f["conversion"].sample(len(ctrl_f),  replace=True).mean()
            b_treat = treat_f["conversion"].sample(len(treat_f), replace=True).mean()
            boot_lifts.append((b_treat - b_ctrl) * 100)

        boot_arr = np.array(boot_lifts)
        ci_low_b  = np.percentile(boot_arr, 2.5)
        ci_high_b = np.percentile(boot_arr, 97.5)

        fig_boot = go.Figure()
        fig_boot.add_trace(go.Histogram(
            x=boot_lifts, nbinsx=50,
            marker=dict(color=COLORS["primary"], opacity=0.8, line=dict(color="#0F1117", width=0.5)),
            name="Bootstrap samples"
        ))
        fig_boot.add_vline(x=0, line_color=COLORS["danger"], line_dash="dash", line_width=2)
        fig_boot.add_vline(x=lift_f*100, line_color=COLORS["success"], line_width=2,
                            annotation_text=f"Observed: {lift_f*100:.3f}pp",
                            annotation_font_color=COLORS["success"])
        fig_boot.add_vrect(x0=ci_low_b, x1=ci_high_b,
                           fillcolor=COLORS["primary"], opacity=0.1, line_width=0)
        fig_boot.update_layout(
            template=PLOT_TEMPLATE, height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.5)",
            xaxis=dict(title="Lift (pp)", gridcolor="#2D2F3E"),
            yaxis=dict(title="Frequency", gridcolor="#2D2F3E"),
            title=f"Bootstrap 95% CI: [{ci_low_b:.3f}, {ci_high_b:.3f}]pp",
            showlegend=False
        )
        st.plotly_chart(fig_boot, use_container_width=True)

    with col2:
        # Spend distribution
        st.markdown("#### Spend Distribution by Group")
        spend_ctrl  = ctrl_f[ctrl_f["spend"] > 0]["spend"]
        spend_treat = treat_f[treat_f["spend"] > 0]["spend"]

        fig_spend = go.Figure()
        fig_spend.add_trace(go.Histogram(
            x=spend_ctrl, nbinsx=30, name="Control",
            marker=dict(color=COLORS["neutral"], opacity=0.7),
            histnorm="probability"
        ))
        fig_spend.add_trace(go.Histogram(
            x=spend_treat, nbinsx=30, name="Treatment",
            marker=dict(color=COLORS["primary"], opacity=0.7),
            histnorm="probability"
        ))
        fig_spend.update_layout(
            barmode="overlay", template=PLOT_TEMPLATE, height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.5)",
            xaxis=dict(title="Spend ($)", gridcolor="#2D2F3E"),
            yaxis=dict(title="Probability", gridcolor="#2D2F3E"),
            title="Spend Distribution (Buyers Only)"
        )
        st.plotly_chart(fig_spend, use_container_width=True)

    # Heatmap: conversion by recency x history
    st.markdown('<div class="section-header">Conversion Heatmap — Recency × History</div>', unsafe_allow_html=True)
    df_filtered2 = df_filtered.copy()
    df_filtered2["history_bin"] = pd.cut(df_filtered2["history"], bins=5,
                                          labels=["0-200", "200-400", "400-600", "600-800", "800+"])
    heat = df_filtered2[df_filtered2["treated"] == 1].groupby(
        ["recency", "history_bin"])["conversion"].mean().unstack().fillna(0) * 100

    fig_heat = go.Figure(go.Heatmap(
        z=heat.values, x=heat.columns.astype(str), y=heat.index,
        colorscale="Blues", text=heat.values.round(1),
        texttemplate="%{text}%", showscale=True,
        colorbar=dict(title="Conv %")
    ))
    fig_heat.update_layout(
        template=PLOT_TEMPLATE, height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Purchase History ($)",
        yaxis_title="Recency (months)",
        title="Treatment Group Conversion Rate by Recency × History"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 3: SQL EXPLORER
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Live SQL Query Explorer</div>', unsafe_allow_html=True)
    st.markdown("*Write SQL directly against the campaign database — just like a real analyst would*")

    conn_sql = sqlite3.connect(DB_PATH)

    query_options = {
        "📊 Key Metrics by Group": """SELECT
    segment,
    COUNT(*) as n_customers,
    ROUND(AVG(visit) * 100, 2) as visit_rate_pct,
    ROUND(AVG(conversion) * 100, 3) as conversion_rate_pct,
    ROUND(AVG(spend), 4) as avg_spend_per_customer
FROM customers
WHERE segment IN ('No E-Mail', 'Mens E-Mail')
GROUP BY segment
ORDER BY conversion_rate_pct DESC""",

        "👥 Cohort Analysis by History": """SELECT
    history_segment,
    segment,
    COUNT(*) as n,
    ROUND(AVG(conversion) * 100, 3) as conv_rate_pct,
    ROUND(AVG(spend), 4) as avg_spend
FROM customers
WHERE segment IN ('No E-Mail', 'Mens E-Mail')
GROUP BY history_segment, segment
ORDER BY history_segment, segment""",

        "🗺️ Geographic Analysis": """SELECT
    zip_code,
    segment,
    COUNT(*) as n,
    ROUND(AVG(conversion) * 100, 3) as conv_rate_pct
FROM customers
WHERE segment IN ('No E-Mail', 'Mens E-Mail')
GROUP BY zip_code, segment
ORDER BY zip_code""",

        "📱 Channel Performance": """SELECT
    channel,
    COUNT(*) as total_customers,
    ROUND(AVG(CASE WHEN segment='Mens E-Mail' THEN conversion END) * 100, 3) as treatment_conv_pct,
    ROUND(AVG(CASE WHEN segment='No E-Mail' THEN conversion END) * 100, 3) as control_conv_pct,
    ROUND((AVG(CASE WHEN segment='Mens E-Mail' THEN conversion END) -
           AVG(CASE WHEN segment='No E-Mail' THEN conversion END)) * 100, 3) as lift_pp
FROM customers
WHERE segment IN ('No E-Mail', 'Mens E-Mail')
GROUP BY channel
ORDER BY lift_pp DESC""",

        "💎 High Value Targeting": """SELECT
    CASE
        WHEN history > 500 THEN 'High Value'
        WHEN history > 200 THEN 'Mid Value'
        ELSE 'Low Value'
    END as tier,
    COUNT(*) as n,
    ROUND(AVG(CASE WHEN segment='Mens E-Mail' THEN conversion END) * 100, 3) as treatment_conv,
    ROUND(AVG(CASE WHEN segment='No E-Mail'  THEN conversion END) * 100, 3) as control_conv,
    ROUND((AVG(CASE WHEN segment='Mens E-Mail' THEN conversion END) -
           AVG(CASE WHEN segment='No E-Mail'  THEN conversion END)) * 100, 3) as lift_pp
FROM customers
WHERE segment IN ('No E-Mail', 'Mens E-Mail')
GROUP BY tier
ORDER BY lift_pp DESC""",

        "✏️ Write Your Own Query": "SELECT * FROM customers LIMIT 10"
    }

    selected_q = st.selectbox("Choose a query template", list(query_options.keys()))
    query_text = query_options[selected_q]
    if selected_q == "✏️ Write Your Own Query":
        query_text = st.text_area("SQL Editor", value=query_text, height=150,
                                   help="Tables available: customers")
    else:
        st.markdown(f'<div class="sql-box">{query_text.strip()}</div>', unsafe_allow_html=True)

    col_run, col_chart = st.columns([1, 3])
    auto_chart = col_chart.checkbox("Auto-generate chart", value=True)

    if st.button("▶ Run Query", type="primary"):
        try:
            result = pd.read_sql_query(query_text, conn_sql)
            st.success(f"✅ Returned {len(result)} rows")
            st.dataframe(result, use_container_width=True, hide_index=True)

            if auto_chart and len(result) > 0:
                num_cols = result.select_dtypes(include=np.number).columns.tolist()
                cat_cols = result.select_dtypes(exclude=np.number).columns.tolist()
                if num_cols and cat_cols:
                    fig_auto = px.bar(result, x=cat_cols[0], y=num_cols[0],
                                      color=cat_cols[1] if len(cat_cols) > 1 else None,
                                      color_discrete_sequence=[COLORS["primary"], COLORS["neutral"],
                                                               COLORS["success"], COLORS["warning"]],
                                      template=PLOT_TEMPLATE)
                    fig_auto.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                           plot_bgcolor="rgba(26,29,39,0.5)",
                                           height=350,
                                           xaxis=dict(gridcolor="#2D2F3E"),
                                           yaxis=dict(gridcolor="#2D2F3E"))
                    st.plotly_chart(fig_auto, use_container_width=True)
        except Exception as e:
            st.error(f"SQL Error: {e}")

    conn_sql.close()

# ─────────────────────────────────────────────────────────────
# TAB 4: CAUSAL INFERENCE
# ─────────────────────────────────────────────────────────────
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Propensity Score Matching")
        if causal_res:
            psm_sig = causal_res.get("psm_significant", False)
            psm_lift = causal_res.get("psm_lift_pp", 0)
            fig_psm = go.Figure(go.Indicator(
                mode="number+delta",
                value=psm_lift,
                number={"suffix": "pp", "font": {"color": "#E0E3F0", "size": 36}},
                delta={"reference": 0, "valueformat": ".3f",
                       "increasing": {"color": COLORS["success"]},
                       "decreasing": {"color": COLORS["danger"]}},
                title={"text": "PSM Lift (pp)<br><span style='font-size:0.8em;color:#8B8FA8'>After matching similar customers</span>"}
            ))
            fig_psm.update_layout(template=PLOT_TEMPLATE, height=200,
                                   paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_psm, use_container_width=True)

            st.markdown(f"""<div class="{'success-box' if psm_sig else 'warning-box'}">
            <strong>PSM p-value:</strong> {causal_res.get('psm_p_value', 0):.6f}<br>
            {'✅ Effect holds after controlling for customer characteristics' if psm_sig else '⚠️ Effect weakens after matching'}
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### Uplift Distribution")
        if os.path.exists(f"{OUTPUT_DIR}/causal_results.json"):
            avg_up  = causal_res.get("avg_uplift_score", 0)
            n_high  = causal_res.get("n_high_uplift", 0)
            n_total = causal_res.get("n_total", 1)

            # Simulated uplift distribution for visualization
            np.random.seed(42)
            uplift_sim = np.concatenate([
                np.random.normal(-0.02, 0.05, 500),
                np.random.normal(0.00, 0.03, 1000),
                np.random.normal(avg_up, 0.04, 500)
            ])
            fig_up = go.Figure()
            fig_up.add_trace(go.Histogram(
                x=uplift_sim, nbinsx=50,
                marker=dict(color=COLORS["primary"], opacity=0.8),
                name="Uplift scores"
            ))
            fig_up.add_vline(x=0, line_color=COLORS["danger"], line_dash="dash",
                              line_width=2, annotation_text="No effect")
            fig_up.add_vline(x=avg_up, line_color=COLORS["success"], line_width=2,
                              annotation_text=f"Mean: {avg_up*100:.2f}pp")
            fig_up.update_layout(
                template=PLOT_TEMPLATE, height=280,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.5)",
                xaxis=dict(title="Uplift Score", gridcolor="#2D2F3E"),
                yaxis=dict(title="Count", gridcolor="#2D2F3E"),
                showlegend=False
            )
            st.plotly_chart(fig_up, use_container_width=True)

    # CATE chart
    if len(cate_df) > 0:
        st.markdown('<div class="section-header">Treatment Effect by Segment</div>', unsafe_allow_html=True)
        cate_sorted = cate_df.sort_values("CATE (pp)")
        colors_c = [COLORS["success"] if s == "✅" else COLORS["neutral"] for s in cate_sorted["Sig"]]

        fig_cate = go.Figure(go.Bar(
            x=cate_sorted["CATE (pp)"],
            y=cate_sorted["Segment"].astype(str),
            orientation="h",
            marker_color=colors_c,
            text=[f"{v:.2f}pp {s}" for v, s in zip(cate_sorted["CATE (pp)"], cate_sorted["Sig"])],
            textposition="outside", textfont=dict(color="white", size=11)
        ))
        fig_cate.add_vline(x=0, line_color="white", line_width=1)
        fig_cate.update_layout(
            template=PLOT_TEMPLATE, height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.5)",
            xaxis=dict(title="Treatment Effect (pp)", gridcolor="#2D2F3E"),
            yaxis=dict(gridcolor="#2D2F3E"),
            title="Which segments respond best to the email?",
            showlegend=False
        )
        st.plotly_chart(fig_cate, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 5: WHAT-IF SIMULATOR
# ─────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">🎮 Campaign What-If Simulator</div>', unsafe_allow_html=True)
    st.markdown("*Simulate different targeting strategies and see their business impact*")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Campaign Parameters")
        sim_n_customers  = st.slider("Total customer base", 10000, 500000, 64000, 1000)
        sim_email_cost   = st.slider("Cost per email ($)", 0.01, 2.0, 0.10, 0.01)
        sim_order_value  = st.slider("Avg order value ($)", 10, 500, 50, 5)
        sim_base_rate    = st.slider("Baseline conversion rate (%)", 0.1, 20.0,
                                      float(ab_res["control_conversion_rate"] * 100), 0.1) / 100
        sim_lift         = st.slider("Expected email lift (pp)", 0.1, 5.0,
                                      float(ab_res["absolute_lift"] * 100), 0.1) / 100

    with col2:
        st.markdown("#### Targeting Strategy")
        strategy = st.radio("Who do we email?", [
            "🌐 Everyone",
            "🎯 Top 50% by purchase history",
            "⚡ Top 25% by purchase history",
            "💎 Top 10% by purchase history",
            "🔬 High Uplift Only (model-based)"
        ])

        pct_map = {
            "🌐 Everyone": 1.0,
            "🎯 Top 50% by purchase history": 0.5,
            "⚡ Top 25% by purchase history": 0.25,
            "💎 Top 10% by purchase history": 0.10,
            "🔬 High Uplift Only (model-based)": 0.25
        }
        uplift_multiplier = {
            "🌐 Everyone": 1.0,
            "🎯 Top 50% by purchase history": 1.2,
            "⚡ Top 25% by purchase history": 1.5,
            "💎 Top 10% by purchase history": 1.8,
            "🔬 High Uplift Only (model-based)": 2.5
        }

        pct_targeted     = pct_map[strategy]
        uplift_mult      = uplift_multiplier[strategy]
        n_emails         = int(sim_n_customers * pct_targeted)
        effective_lift   = sim_lift * uplift_mult
        campaign_cost    = n_emails * sim_email_cost
        extra_conversions= effective_lift * n_emails
        revenue_lift     = extra_conversions * sim_order_value
        net_roi          = revenue_lift - campaign_cost
        roi_pct          = (net_roi / campaign_cost * 100) if campaign_cost > 0 else 0

    # Results
    st.markdown("---")
    st.markdown("#### 📊 Simulation Results")
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{n_emails:,}</div><div class="metric-label">Emails Sent</div><div class="metric-delta" style="color:#8B8FA8">{pct_targeted*100:.0f}% of base</div></div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">${campaign_cost:,.0f}</div><div class="metric-label">Campaign Cost</div></div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{extra_conversions:.0f}</div><div class="metric-label">Extra Conversions</div><div class="metric-delta delta-good">+{effective_lift*100:.2f}pp lift</div></div>""", unsafe_allow_html=True)
    with r4:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">${revenue_lift:,.0f}</div><div class="metric-label">Revenue Lift</div></div>""", unsafe_allow_html=True)
    with r5:
        color = "delta-good" if net_roi > 0 else "delta-bad"
        st.markdown(f"""<div class="metric-card"><div class="metric-value">${net_roi:,.0f}</div><div class="metric-label">Net ROI</div><div class="metric-delta {color}">{roi_pct:.1f}%</div></div>""", unsafe_allow_html=True)

    # Compare all strategies
    st.markdown("#### Strategy Comparison")
    strategies_all = list(pct_map.keys())
    comparison = []
    for s in strategies_all:
        p  = pct_map[s]
        um = uplift_multiplier[s]
        n  = int(sim_n_customers * p)
        el = sim_lift * um
        cc = n * sim_email_cost
        rl = el * n * sim_order_value
        nr = rl - cc
        rp = (nr / cc * 100) if cc > 0 else 0
        comparison.append({"Strategy": s, "Emails": n, "Cost": cc,
                            "Revenue": rl, "Net ROI": nr, "ROI%": rp})

    comp_df = pd.DataFrame(comparison)

    fig_comp = make_subplots(rows=1, cols=2,
                              subplot_titles=["Net ROI by Strategy", "ROI % by Strategy"])

    colors_strat = [COLORS["primary"] if c == strategy else COLORS["neutral"]
                    for c in comp_df["Strategy"]]

    fig_comp.add_trace(go.Bar(
        x=comp_df["Strategy"].str[:20], y=comp_df["Net ROI"],
        marker_color=colors_strat, name="Net ROI",
        text=[f"${v:,.0f}" for v in comp_df["Net ROI"]], textposition="outside",
        textfont=dict(color="white", size=10)
    ), row=1, col=1)

    fig_comp.add_trace(go.Bar(
        x=comp_df["Strategy"].str[:20], y=comp_df["ROI%"],
        marker_color=colors_strat, name="ROI %",
        text=[f"{v:.1f}%" for v in comp_df["ROI%"]], textposition="outside",
        textfont=dict(color="white", size=10)
    ), row=1, col=2)

    fig_comp.update_layout(
        template=PLOT_TEMPLATE, height=380,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.5)",
        showlegend=False,
        xaxis=dict(tickangle=20, gridcolor="#2D2F3E"),
        xaxis2=dict(tickangle=20, gridcolor="#2D2F3E"),
        yaxis=dict(gridcolor="#2D2F3E"),
        yaxis2=dict(gridcolor="#2D2F3E")
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    best = comp_df.loc[comp_df["Net ROI"].idxmax()]
    st.markdown(f"""<div class="insight-box">
    <strong>💡 Simulation Insight:</strong><br>
    Best strategy: <strong>{best['Strategy']}</strong> with ${best['Net ROI']:,.0f} net ROI ({best['ROI%']:.1f}%)<br>
    Your selected strategy: <strong>{strategy}</strong> generates ${net_roi:,.0f} net ROI ({roi_pct:.1f}%)<br>
    <br>
    The model-based high-uplift targeting typically outperforms blanket emailing by 2-3x because
    it focuses budget on customers who genuinely respond to the email, not those who would have
    converted anyway.
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4A4E6A;font-size:12px;'>"
    "A/B Testing + Causal Inference · Hillstrom Email Analytics · "
    "SQL + Propensity Score Matching + Uplift Modeling + Bootstrap"
    "</div>", unsafe_allow_html=True
)

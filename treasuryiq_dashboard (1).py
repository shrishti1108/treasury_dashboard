"""
TreasuryIQ – BI Module  v2.0
Combined Dashboard: 5 Datasets
  1. treasury_master_dataset__1_.csv   → Dashboards 1, 2, 3  (primary, 3-year history)
  2. interest_shock.csv                → Dashboard 2, 4      (rate shock period)
  3. stress_predictions.csv            → Dashboard 4         (scenario analysis)
  4. borrow_features.csv               → Dashboards 1, 2     (200k borrowing records)
  5. procurement_synthetic_data.csv    → Dashboard 5 (new)   (procurement spend)

Run: streamlit run treasuryiq_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="TreasuryIQ – BI Module v2",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS / THEME
# ──────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0d1117;color:#e6edf3}
[data-testid="stSidebar"]{background:#161b22}
[data-testid="stHeader"]{background:transparent}
.kpi-card{background:linear-gradient(135deg,#1c2333,#21262d);border:1px solid #30363d;
          border-radius:12px;padding:14px 16px;text-align:center;transition:transform .15s}
.kpi-card:hover{transform:translateY(-3px);border-color:#58a6ff}
.kpi-title{font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px}
.kpi-value{font-size:1.45rem;font-weight:700;color:#e6edf3}
.sec-hd{font-size:1rem;font-weight:600;color:#58a6ff;border-left:3px solid #58a6ff;
         padding-left:9px;margin:18px 0 8px}
button[data-baseweb="tab"]{color:#8b949e!important}
button[data-baseweb="tab"][aria-selected="true"]{color:#58a6ff!important;
  border-bottom:2px solid #58a6ff!important}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PLOTLY DEFAULTS
# ──────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(28,35,51,0.6)",
    font=dict(color="#e6edf3", size=11),
    xaxis=dict(gridcolor="#21262d", zeroline=False),
    yaxis=dict(gridcolor="#21262d", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    margin=dict(l=8, r=8, t=32, b=8),
    hovermode="x unified",
)
GREEN="#3fb950"; RED="#f85149"; YELLOW="#d29922"
BLUE="#58a6ff";  PURPLE="#bc8cff"; ORANGE="#ffa657"; CYAN="#39d353"

SCENARIO_COL = {
    "Baseline":         "baseline",
    "Interest Shock":   "interest_shock",
    "Funding Failure":  "funding_failure",
    "Prepayment Surge": "prepayment_surge",
    "Liquidity Shock":  "liquidity_shock",
}
SCENARIO_COLOR = {
    "Baseline":BLUE, "Interest Shock":ORANGE,
    "Funding Failure":PURPLE, "Prepayment Surge":YELLOW, "Liquidity Shock":RED,
}

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all():
    # 1. Treasury Master – primary (2022-01-01 → 2024-12-30, 1095 rows)
    tm = pd.read_csv("treasury_master_dataset__1_.csv", parse_dates=["date"])
    tm = tm.sort_values("date").reset_index(drop=True)

    # 2. Interest Shock – rate-shock period + baseline_prediction column
    ishock = pd.read_csv("interest_shock.csv", parse_dates=["date"])
    ishock = ishock.sort_values("date").reset_index(drop=True)

    # Enrich treasury master with baseline_prediction where available
    tm = tm.merge(ishock[["date","baseline_prediction"]], on="date", how="left")

    # 3. Stress Predictions – forward scenario data
    stress = pd.read_csv("stress_predictions.csv", parse_dates=["date"])
    stress = stress.sort_values("date").reset_index(drop=True)

    # 4. Borrow Features – 200k borrowing records
    borrow = pd.read_csv("borrow_features.csv")

    # 5. Procurement
    proc = pd.read_csv("procurement_synthetic_data.csv",
                       parse_dates=["PO_Date"], dayfirst=True)

    # Derived columns on treasury master
    np.random.seed(42)
    n = len(tm)
    tm["month"]          = tm["date"].dt.to_period("M").dt.to_timestamp()
    tm["year"]           = tm["date"].dt.year
    tm["variance"]       = tm["net_liquidity_position"].diff().fillna(0)
    tm["roll7_vol"]      = tm["variance"].rolling(7,  min_periods=1).std()
    tm["roll30_vol"]     = tm["variance"].rolling(30, min_periods=1).std()
    tm["expected_inflow"]= tm["total_inflows"] * np.random.uniform(0.93, 1.07, n)
    tm["actual_inflow"]  = tm["total_inflows"]
    tm["cf_variance"]    = tm["actual_inflow"] - tm["expected_inflow"]
    tm["cf_var_pct"]     = tm["cf_variance"] / tm["expected_inflow"] * 100

    # Borrow aggregates (pre-computed for speed)
    bagg = {
        "by_source":    borrow.groupby("funding_source")["borrowing_amount"].sum(),
        "by_bucket":    borrow.groupby("maturity_bucket")["borrowing_amount"].sum(),
        "by_reset":     borrow.groupby("interest_reset_type")["borrowing_amount"].sum(),
        "risk_heat":    borrow.groupby(["funding_source","maturity_bucket"])[
                            "rollover_risk_score"].mean().reset_index(),
        "cost_src":     borrow.groupby("funding_source")["cost_of_funds"].mean() * 100,
    }
    return tm, ishock, stress, borrow, bagg, proc

with st.spinner("Loading TreasuryIQ v2 data…"):
    tm, ishock, stress, borrow, bagg, proc = load_all()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 TreasuryIQ v2")
    st.caption("5 datasets · 3-year treasury view")
    st.markdown("---")

    min_d = tm["date"].min().date()
    max_d = tm["date"].max().date()
    dr = st.date_input("📅 Date Range",
                       value=(min_d, max_d), min_value=min_d, max_value=max_d)
    d_start = pd.Timestamp(dr[0] if len(dr)==2 else min_d)
    d_end   = pd.Timestamp(dr[1] if len(dr)==2 else max_d)

    st.markdown("---")
    fund_opts   = ["All"] + sorted(borrow["funding_source"].unique())
    bucket_opts = ["All"] + sorted(borrow["maturity_bucket"].unique())
    reset_opts  = ["All"] + sorted(borrow["interest_reset_type"].unique())
    sel_fund    = st.multiselect("💰 Funding Source",   fund_opts,   default=["All"])
    sel_bucket  = st.multiselect("⏱ Maturity Bucket",  bucket_opts, default=["All"])
    sel_reset   = st.multiselect("🔄 Interest Reset",   reset_opts,  default=["All"])

    st.markdown("---")
    region_opts = ["All"] + sorted(proc["Region"].dropna().unique())
    status_opts = ["All","Open","Closed","Cancelled"]
    sel_region  = st.multiselect("🌍 Procurement Region", region_opts, default=["All"])
    sel_status  = st.multiselect("📋 PO Status",          status_opts, default=["All"])

    st.markdown("---")
    time_mode = st.radio("🕐 Granularity", ["Daily","Monthly"])
    view_mode = st.radio("📊 View Mode",   ["Absolute","Percentage"])
    stress_scenario = st.selectbox("⚡ Active Stress Scenario",
        list(SCENARIO_COL.keys()))

    st.markdown("---")
    st.caption(f"Treasury: {len(tm):,} rows")
    st.caption(f"Borrowings: {len(borrow):,} rows")
    st.caption(f"Stress Scenarios: {len(stress):,} days")
    st.caption(f"Procurement: {len(proc):,} POs")

# ──────────────────────────────────────────────
# FILTER HELPERS
# ──────────────────────────────────────────────
def ftm():
    return tm[(tm["date"]>=d_start)&(tm["date"]<=d_end)].copy()

def fstress():
    return stress[(stress["date"]>=d_start)&(stress["date"]<=d_end)].copy()

def fishock():
    return ishock[(ishock["date"]>=d_start)&(ishock["date"]<=d_end)].copy()

def fborrow():
    b = borrow.copy()
    if "All" not in sel_fund:   b = b[b["funding_source"].isin(sel_fund)]
    if "All" not in sel_bucket: b = b[b["maturity_bucket"].isin(sel_bucket)]
    if "All" not in sel_reset:  b = b[b["interest_reset_type"].isin(sel_reset)]
    return b

def fproc():
    p = proc.copy()
    if "All" not in sel_region: p = p[p["Region"].isin(sel_region)]
    if "All" not in sel_status: p = p[p["PO_Status"].isin(sel_status)]
    return p

def tmode(df, dcol="date"):
    if time_mode == "Monthly":
        df = df.copy()
        df["_p"] = df[dcol].dt.to_period("M").dt.to_timestamp()
        return df, "_p"
    return df, dcol

def cr(x):
    if pd.isna(x): return "N/A"
    c = x / 1e7
    if abs(c) >= 1e5: return f"₹{c/1e5:.2f}L Cr"
    if abs(c) >= 1000: return f"₹{c/1000:.1f}K Cr"
    return f"₹{c:.2f} Cr"

def kpi(title, val, color=None):
    cv = {"green":"color:#3fb950","red":"color:#f85149",
          "yellow":"color:#d29922"}.get(color,"")
    return (f'<div class="kpi-card"><div class="kpi-title">{title}</div>'
            f'<div class="kpi-value" style="{cv}">{val}</div></div>')

def sec(txt):
    st.markdown(f'<div class="sec-hd">{txt}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;font-size:1.85rem;color:#58a6ff;margin-bottom:0'>
🏦 TreasuryIQ – BI Module v2.0</h1>
<p style='text-align:center;color:#8b949e;font-size:.82rem;margin-top:3px'>
Treasury Master &nbsp;·&nbsp; Interest Shock &nbsp;·&nbsp; Stress Predictions
&nbsp;·&nbsp; Borrow Features &nbsp;·&nbsp; Procurement
</p>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Liquidity Overview",
    "📉 ALM & Borrow Risk",
    "📈 Cashflow Variance",
    "🔮 Stress & Forecast",
    "🛒 Procurement",
])

# ══════════════════════════════════════════════════════════
# TAB 1 – LIQUIDITY OVERVIEW
# ══════════════════════════════════════════════════════════
with tab1:
    t = ftm()
    b = fborrow()

    net_liq     = t["net_liquidity_position"].iloc[-1] if len(t) else 0
    total_in    = t["total_inflows"].iloc[-1]           if len(t) else 0
    total_out   = t["total_outflows"].iloc[-1]          if len(t) else 0
    lcr         = t["liquidity_coverage_ratio"].iloc[-1]if len(t) else 0
    risk_sc     = t["liquidity_risk_score"].mean()      if len(t) else 0
    stress_days = int(t["funding_stress_flag"].sum())
    mkt_stress  = int(t["market_stress_flag"].sum())
    avg_cof     = b["cost_of_funds"].mean() * 100       if len(b) else 0

    nl_col  = "green" if net_liq > 0 else "red"
    lcr_col = "green" if lcr > 1.2 else ("yellow" if lcr >= 1.0 else "red")

    cols = st.columns(8)
    for col, (ttl, val, c) in zip(cols, [
        ("Total Inflows",        cr(total_in),             None),
        ("Total Outflows",       cr(total_out),            None),
        ("Net Liquidity",        cr(net_liq),              nl_col),
        ("LCR",                  f"{lcr:.2f}x",            lcr_col),
        ("Avg Cost of Funds",    f"{avg_cof:.2f}%",        None),
        ("Liquidity Risk Score", f"{risk_sc:,.0f}",        None),
        ("Funding Stress Days",  str(stress_days),         "red" if stress_days>30 else "green"),
        ("Market Stress Days",   str(mkt_stress),          "red" if mkt_stress>100 else "yellow"),
    ]):
        col.markdown(kpi(ttl, val, c), unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        sec("📈 Net Liquidity Position Over Time")
        td, dc = tmode(t)
        pd_ = td.groupby(dc)["net_liquidity_position"].mean().reset_index() \
              if time_mode == "Monthly" else td[[dc, "net_liquidity_position"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd_[dc], y=pd_["net_liquidity_position"],
            fill="tozeroy", line=dict(color=BLUE, width=2),
            fillcolor="rgba(88,166,255,0.12)", name="Net Liquidity"))
        fig.add_hline(y=0, line_dash="dash", line_color=RED, opacity=0.5)
        fig.update_layout(**PL, title="Net Liquidity Position")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("📊 Inflows vs Outflows")
        td, dc = tmode(t)
        gp = td.groupby(dc)[["total_inflows","total_outflows"]].mean().reset_index() \
             if time_mode == "Monthly" else td[[dc,"total_inflows","total_outflows"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=gp[dc], y=gp["total_inflows"],  name="Inflows",  marker_color=GREEN, opacity=.85))
        fig2.add_trace(go.Bar(x=gp[dc], y=gp["total_outflows"], name="Outflows", marker_color=RED,   opacity=.85))
        fig2.update_layout(**PL, barmode="group", title="Inflows vs Outflows")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        sec("📉 Repo Rate & CP Rate Trend")
        td, dc = tmode(t)
        rp = td.groupby(dc)[["repo_rate","cp_rate","rate_spread"]].mean().reset_index() \
             if time_mode == "Monthly" else td[[dc,"repo_rate","cp_rate","rate_spread"]]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=rp[dc], y=rp["repo_rate"],   name="Repo Rate",   line=dict(color=BLUE,   width=2)))
        fig3.add_trace(go.Scatter(x=rp[dc], y=rp["cp_rate"],     name="CP Rate",     line=dict(color=ORANGE, width=2)))
        fig3.add_trace(go.Scatter(x=rp[dc], y=rp["rate_spread"], name="Rate Spread", line=dict(color=PURPLE, width=1.5, dash="dot")))
        fig3.update_layout(**PL, title="Rate Environment (Treasury Master)")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        sec("🔥 Liquidity Risk Score")
        td, dc = tmode(t)
        rs = td.groupby(dc)["liquidity_risk_score"].mean().reset_index() \
             if time_mode == "Monthly" else td[[dc,"liquidity_risk_score"]]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=rs[dc], y=rs["liquidity_risk_score"],
            fill="tozeroy", line=dict(color=ORANGE, width=2),
            fillcolor="rgba(255,166,87,0.12)", name="Risk Score"))
        fig4.update_layout(**PL, title="Liquidity Risk Score Trend")
        st.plotly_chart(fig4, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        sec("🗓 Funding & Market Stress Flags")
        td, dc = tmode(t)
        sf = td.groupby(dc)[["funding_stress_flag","market_stress_flag"]].sum().reset_index() \
             if time_mode == "Monthly" else td[[dc,"funding_stress_flag","market_stress_flag"]]
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=sf[dc], y=sf["funding_stress_flag"],  name="Funding Stress", marker_color=RED,    opacity=.8))
        fig5.add_trace(go.Bar(x=sf[dc], y=sf["market_stress_flag"],   name="Market Stress",  marker_color=YELLOW, opacity=.8))
        fig5.update_layout(**PL, barmode="stack", title="Stress Flag Days")
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        sec("💰 Funding Source Mix (Borrow Features)")
        bs = b.groupby("funding_source")["borrowing_amount"].sum().reset_index()
        if view_mode == "Percentage":
            bs["borrowing_amount"] = bs["borrowing_amount"] / bs["borrowing_amount"].sum() * 100
        fig6 = go.Figure(go.Pie(
            labels=bs["funding_source"], values=bs["borrowing_amount"],
            hole=0.5, marker_colors=[BLUE, GREEN, ORANGE, PURPLE],
            textinfo="label+percent"))
        fig6.update_layout(**PL, title="Funding Source Distribution", showlegend=True)
        st.plotly_chart(fig6, use_container_width=True)

    with st.expander("🧠 Interpretation Guide"):
        i1, i2, i3, i4 = st.columns(4)
        i1.error("🔴 Net Liquidity < 0\nImmediate funding required.")
        i2.warning("🟡 LCR < 1.2x\nBuffer thinning – monitor daily.")
        i3.warning("🟡 Rising Rate Spread\nMarket stress or poor funding mix.")
        i4.info("🔵 High Risk Score\nLiquidity sensitivity escalating.")

# ══════════════════════════════════════════════════════════
# TAB 2 – ALM & BORROW RISK
# ══════════════════════════════════════════════════════════
with tab2:
    t = ftm()
    b = fborrow()

    alm_last    = t["alm_gap"].iloc[-1]            if len(t) else 0
    cum_alm     = t["cumulative_alm_gap"].iloc[-1] if len(t) else 0
    total_borr  = b["borrowing_amount"].sum()
    avg_rr      = b["rollover_risk_score"].mean()
    avg_cof2    = b["cost_of_funds"].mean() * 100
    st_borr     = b[b["maturity_bucket"].isin(["1M","3M"])]["borrowing_amount"].sum()
    st_ratio    = st_borr / total_borr if total_borr else 0
    hi_risk_ct  = int((b["rollover_risk_score"] > 0.7).sum())

    alm_col = "green" if alm_last > 0 else "red"
    str_col = "red"   if st_ratio > 0.4 else ("yellow" if st_ratio > 0.25 else "green")

    cols = st.columns(7)
    for col, (ttl, val, c) in zip(cols, [
        ("ALM Gap",              cr(alm_last),               alm_col),
        ("Cumulative ALM Gap",   cr(cum_alm),                None),
        ("Total Borrowings",     cr(total_borr),             None),
        ("Avg Cost of Funds",    f"{avg_cof2:.2f}%",         None),
        ("Avg Rollover Risk",    f"{avg_rr:.3f}",            "red" if avg_rr > 0.6 else "yellow"),
        ("ST Borrowing Ratio",   f"{st_ratio*100:.1f}%",     str_col),
        ("High-Risk Borrowings", f"{hi_risk_ct:,}",          "red" if hi_risk_ct > 50000 else "yellow"),
    ]):
        col.markdown(kpi(ttl, val, c), unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        sec("📊 ALM Gap Trend")
        td, dc = tmode(t)
        ag = td.groupby(dc)["alm_gap"].mean().reset_index() \
             if time_mode == "Monthly" else td[[dc,"alm_gap"]]
        colors = [GREEN if v > 0 else RED for v in ag["alm_gap"]]
        fig = go.Figure(go.Bar(x=ag[dc], y=ag["alm_gap"], marker_color=colors, name="ALM Gap"))
        fig.add_hline(y=0, line_dash="dash", line_color=YELLOW)
        fig.update_layout(**PL, title="ALM Gap Over Time")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("📈 Cumulative ALM Gap")
        td, dc = tmode(t)
        cg = td.groupby(dc)["cumulative_alm_gap"].last().reset_index() \
             if time_mode == "Monthly" else td[[dc,"cumulative_alm_gap"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=cg[dc], y=cg["cumulative_alm_gap"],
            fill="tozeroy", line=dict(color=ORANGE, width=2),
            fillcolor="rgba(255,166,87,0.1)", name="Cumulative Gap"))
        fig2.update_layout(**PL, title="Cumulative ALM Gap")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        sec("📦 Borrowing Volume by Maturity Bucket")
        ORDER = ["1M","3M","6M","1Y","3Y"]
        bkt = b.groupby("maturity_bucket")["borrowing_amount"].sum().reindex(ORDER).fillna(0).reset_index()
        bkt.columns = ["maturity_bucket","borrowing_amount"]
        if view_mode == "Percentage":
            bkt["borrowing_amount"] = bkt["borrowing_amount"] / bkt["borrowing_amount"].sum() * 100
        bcolors = [RED, RED, YELLOW, GREEN, GREEN]
        fig3 = go.Figure(go.Bar(
            x=bkt["maturity_bucket"], y=bkt["borrowing_amount"],
            marker_color=bcolors,
            text=[f"{v:.1f}%" if view_mode=="Percentage" else cr(v) for v in bkt["borrowing_amount"]],
            textposition="outside"))
        fig3.update_layout(**PL, title="Borrowing Volume by Maturity")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        sec("💸 Avg Cost of Funds by Funding Source")
        cof_s = b.groupby("funding_source")["cost_of_funds"].mean() * 100
        fig4 = go.Figure(go.Bar(
            x=cof_s.index, y=cof_s.values,
            marker_color=[BLUE, GREEN, ORANGE, PURPLE],
            text=[f"{v:.2f}%" for v in cof_s.values], textposition="outside"))
        fig4.update_layout(**PL, title="Avg Cost of Funds (%) by Source")
        st.plotly_chart(fig4, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        sec("🔥 Rollover Risk Heatmap (Source × Bucket)")
        rh = b.groupby(["funding_source","maturity_bucket"])["rollover_risk_score"].mean().reset_index()
        pivot = rh.pivot(index="funding_source", columns="maturity_bucket",
                         values="rollover_risk_score")
        ORDER2 = [x for x in ["1M","3M","6M","1Y","3Y"] if x in pivot.columns]
        pivot = pivot.reindex(columns=ORDER2).fillna(0)
        fig5 = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[[0, GREEN],[0.5, YELLOW],[1, RED]],
            text=[[f"{v:.3f}" for v in row] for row in pivot.values],
            texttemplate="%{text}", showscale=True))
        fig5.update_layout(**PL, title="Rollover Risk: Funding Source × Maturity Bucket")
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        sec("⚖️ Fixed vs Floating Borrowings")
        rf = b.groupby("interest_reset_type")["borrowing_amount"].sum().reset_index()
        fig6 = go.Figure(go.Pie(
            labels=rf["interest_reset_type"], values=rf["borrowing_amount"],
            hole=0.55, marker_colors=[BLUE, ORANGE], textinfo="label+percent"))
        fig6.update_layout(**PL, title="Fixed vs Floating Rate Split")
        st.plotly_chart(fig6, use_container_width=True)

    sec("📊 Rollover Risk Distribution by Funding Source")
    fig7 = go.Figure()
    for src, grp in b.groupby("funding_source"):
        fig7.add_trace(go.Histogram(x=grp["rollover_risk_score"],
            name=src, nbinsx=40, opacity=0.7))
    fig7.add_vline(x=0.7, line_dash="dash", line_color=RED, annotation_text="High Risk (0.7)")
    fig7.update_layout(**PL, barmode="overlay",
        title="Rollover Risk Score Distribution")
    st.plotly_chart(fig7, use_container_width=True)

    with st.expander("🧠 Interpretation Guide"):
        c1, c2, c3, c4 = st.columns(4)
        c1.error("🔴 Negative ALM Gap\nAssets < Liabilities – immediate risk.")
        c2.warning("🟡 High ST Ratio\nVulnerable to rate shocks.")
        c3.error("🔴 Rollover Risk > 0.7\nHigh refinancing dependency.")
        c4.info("🔵 High CoF\nFunding mix needs optimisation.")

# ══════════════════════════════════════════════════════════
# TAB 3 – CASHFLOW VARIANCE
# ══════════════════════════════════════════════════════════
with tab3:
    t = ftm()

    total_exp = t["expected_inflow"].sum()
    total_act = t["actual_inflow"].sum()
    total_var = t["cf_variance"].sum()
    var_pct   = total_var / total_exp * 100 if total_exp else 0
    avg_var   = t["cf_variance"].mean()
    vol       = t["cf_variance"].std()
    accuracy  = max(0, 1 - abs(var_pct / 100)) * 100

    var_col = "green" if total_var >= 0 else "red"
    acc_col = "green" if accuracy >= 90 else ("yellow" if accuracy >= 75 else "red")

    cols = st.columns(7)
    for col, (ttl, val, c) in zip(cols, [
        ("Expected Cashflow",   cr(total_exp),         None),
        ("Actual Cashflow",     cr(total_act),         None),
        ("Total Variance",      cr(total_var),         var_col),
        ("Variance %",          f"{var_pct:+.2f}%",   var_col),
        ("Avg Daily Variance",  cr(avg_var),           None),
        ("Volatility (σ)",      cr(vol),               None),
        ("Forecast Accuracy",   f"{accuracy:.1f}%",   acc_col),
    ]):
        col.markdown(kpi(ttl, val, c), unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        sec("📈 Expected vs Actual Cashflow")
        td, dc = tmode(t)
        cf = td.groupby(dc)[["expected_inflow","actual_inflow"]].sum().reset_index() \
             if time_mode == "Monthly" else td[[dc,"expected_inflow","actual_inflow"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cf[dc], y=cf["expected_inflow"],
            name="Expected", line=dict(color=BLUE, width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=cf[dc], y=cf["actual_inflow"],
            name="Actual",   line=dict(color=GREEN, width=2)))
        fig.update_layout(**PL, title="Expected vs Actual")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("📉 Cashflow Variance Bars")
        td, dc = tmode(t)
        vd = td.groupby(dc)["cf_variance"].sum().reset_index() \
             if time_mode == "Monthly" else td[[dc,"cf_variance"]]
        vc = [GREEN if v >= 0 else RED for v in vd["cf_variance"]]
        fig2 = go.Figure(go.Bar(x=vd[dc], y=vd["cf_variance"],
            marker_color=vc, name="Variance"))
        fig2.add_hline(y=0, line_dash="dash", line_color=YELLOW, opacity=0.6)
        fig2.update_layout(**PL, title="Cashflow Variance")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        sec("📊 Variance Distribution (Histogram)")
        m, s = t["cf_variance"].mean(), t["cf_variance"].std()
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=t["cf_variance"], nbinsx=60,
            marker_color=BLUE, opacity=0.75, name="Variance"))
        fig3.add_vline(x=m,     line_color=YELLOW, line_dash="dash", annotation_text="Mean")
        fig3.add_vline(x=m+2*s, line_color=RED,    line_dash="dot",  annotation_text="+2σ")
        fig3.add_vline(x=m-2*s, line_color=RED,    line_dash="dot",  annotation_text="-2σ")
        fig3.update_layout(**PL, title="Variance Distribution")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        sec("📈 Rolling Volatility (7d & 30d)")
        ts = t.sort_values("date")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=ts["date"], y=ts["roll7_vol"],
            name="7-Day σ",  line=dict(color=ORANGE, width=2)))
        fig4.add_trace(go.Scatter(x=ts["date"], y=ts["roll30_vol"],
            name="30-Day σ", line=dict(color=PURPLE, width=2)))
        fig4.update_layout(**PL, title="Rolling Cashflow Volatility")
        st.plotly_chart(fig4, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        sec("📉 Risk Score vs Rate Spread (Dual Axis)")
        td, dc = tmode(t)
        sc = td.groupby(dc)[["liquidity_risk_score","rate_spread"]].mean().reset_index() \
             if time_mode == "Monthly" else td[[dc,"liquidity_risk_score","rate_spread"]]
        fig5 = make_subplots(specs=[[{"secondary_y": True}]])
        fig5.add_trace(go.Scatter(x=sc[dc], y=sc["liquidity_risk_score"],
            name="Risk Score", line=dict(color=RED,  width=2)), secondary_y=False)
        fig5.add_trace(go.Scatter(x=sc[dc], y=sc["rate_spread"],
            name="Rate Spread",line=dict(color=BLUE, width=1.5, dash="dot")), secondary_y=True)
        fig5.update_layout(**PL, title="Risk Score vs Rate Spread")
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        sec("🗓 Monthly Variance Seasonality Heatmap")
        th = t.copy()
        th["month_num"] = th["date"].dt.month
        th["week_m"]    = ((th["date"].dt.day - 1) // 7) + 1
        hm = th.groupby(["month_num","week_m"])["cf_variance"].mean().reset_index()
        hp = hm.pivot(index="month_num", columns="week_m", values="cf_variance").fillna(0)
        mlbls = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig6 = go.Figure(go.Heatmap(
            z=hp.values, x=[f"W{i}" for i in hp.columns],
            y=[mlbls[m-1] for m in hp.index],
            colorscale=[[0, RED],[0.5, YELLOW],[1, GREEN]], showscale=True))
        fig6.update_layout(**PL, title="Seasonality: Avg Variance by Month × Week")
        st.plotly_chart(fig6, use_container_width=True)

    with st.expander("🧠 Interpretation Guide"):
        c1, c2, c3, c4 = st.columns(4)
        c1.error("🔴 Persistent -ve Variance\nInflow overestimation risk.")
        c2.warning("🟡 High Volatility\nUnstable liquidity – buffer needed.")
        c3.info("🔵 Seasonal Spikes\nPredict and plan for month-end.")
        c4.success("🟢 Accuracy > 90%\nForecasting model is reliable.")

# ══════════════════════════════════════════════════════════
# TAB 4 – STRESS & FORECAST
# ══════════════════════════════════════════════════════════
with tab4:
    t     = ftm()
    st_df = fstress()
    is_df = fishock()

    scol      = SCENARIO_COL[stress_scenario]
    scen_val  = st_df[scol].mean()       if (len(st_df) and scol in st_df.columns) else 0
    base_val  = st_df["baseline"].mean() if len(st_df) else 0
    stress_imp = (scen_val - base_val) / abs(base_val) * 100 if base_val else 0

    breach_col_map = {
        "interest_shock":   "breach_interest",
        "funding_failure":  "breach_funding",
        "prepayment_surge": "breach_prepayment",
        "liquidity_shock":  "breach_liquidity",
    }
    bc = breach_col_map.get(scol)
    breach_prob = st_df[bc].mean() * 100 if (bc and bc in st_df.columns and len(st_df)) else 0

    all_scen_cols = [c for c in SCENARIO_COL.values() if c in (st_df.columns if len(st_df) else [])]
    worst = min(st_df[c].min() for c in all_scen_cols) if all_scen_cols else 0
    best  = max(st_df[c].max() for c in all_scen_cols) if all_scen_cols else 0

    bp_col = "red" if breach_prob > 20 else ("yellow" if breach_prob > 5 else "green")
    si_col = "red" if stress_imp < -10 else ("yellow" if stress_imp < 0 else "green")

    cols = st.columns(7)
    for col, (ttl, val, c) in zip(cols, [
        ("Baseline Liquidity",   cr(base_val),               None),
        (stress_scenario,        cr(scen_val),               "red" if scen_val < base_val else "green"),
        ("Stress Impact",        f"{stress_imp:+.2f}%",      si_col),
        ("Breach Probability",   f"{breach_prob:.1f}%",      bp_col),
        ("Worst Case",           cr(worst),                  "red"),
        ("Best Case",            cr(best),                   "green"),
        ("Forecast Days",        str(len(st_df)),            None),
    ]):
        col.markdown(kpi(ttl, val, c), unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        sec("📈 All Stress Scenarios")
        fig = go.Figure()
        if len(st_df):
            for sname, scname in SCENARIO_COL.items():
                if scname in st_df.columns:
                    fig.add_trace(go.Scatter(
                        x=st_df["date"], y=st_df[scname],
                        name=sname, line=dict(color=SCENARIO_COLOR[sname], width=2)))
        fig.add_hline(y=0, line_dash="dash", line_color=RED, opacity=0.4)
        fig.update_layout(**PL, title="Liquidity Under All Stress Scenarios")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("📊 Average Liquidity by Scenario")
        if len(st_df):
            means = {sn: st_df[sc].mean() for sn, sc in SCENARIO_COL.items() if sc in st_df.columns}
            fig2 = go.Figure(go.Bar(
                x=list(means.keys()), y=list(means.values()),
                marker_color=[SCENARIO_COLOR[k] for k in means],
                text=[cr(v) for v in means.values()], textposition="outside"))
            fig2.add_hline(y=0, line_dash="dash", line_color=YELLOW)
            fig2.update_layout(**PL, title="Mean Liquidity by Scenario")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No stress data in selected date range.")

    c3, c4 = st.columns(2)
    with c3:
        sec("⚡ Interest Shock: Rates vs Risk Score")
        if len(is_df):
            grp_is = is_df.copy()
            if time_mode == "Monthly":
                grp_is["_p"] = grp_is["date"].dt.to_period("M").dt.to_timestamp()
                grp_is = grp_is.groupby("_p")[["repo_rate","cp_rate","liquidity_risk_score"]].mean().reset_index()
                dc2 = "_p"
            else:
                dc2 = "date"
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            fig3.add_trace(go.Scatter(x=grp_is[dc2], y=grp_is["repo_rate"],
                name="Repo Rate", line=dict(color=BLUE,   width=2)), secondary_y=False)
            fig3.add_trace(go.Scatter(x=grp_is[dc2], y=grp_is["cp_rate"],
                name="CP Rate",   line=dict(color=ORANGE, width=2)), secondary_y=False)
            fig3.add_trace(go.Scatter(x=grp_is[dc2], y=grp_is["liquidity_risk_score"],
                name="Risk Score",line=dict(color=RED,    width=1.5, dash="dot")), secondary_y=True)
            fig3.update_layout(**PL, title="Interest Shock: Rate vs Risk Score")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Interest shock data not available in selected date range.")

    with c4:
        sec("🎲 Monte Carlo Distribution (5,000 Sims)")
        if len(st_df):
            loc = st_df["baseline"].mean()
            scale = max(st_df["baseline"].std() * 5, loc * 0.03)
            mc = np.random.normal(loc=loc, scale=scale, size=5000)
            fig4 = go.Figure()
            fig4.add_trace(go.Histogram(x=mc, nbinsx=80,
                marker_color=BLUE, opacity=0.75, name="Simulations"))
            fig4.add_vline(x=0,        line_color=RED,    line_dash="dash", annotation_text="Breach")
            fig4.add_vline(x=mc.mean(),line_color=GREEN,  line_dash="dot",  annotation_text=f"Mean")
            fig4.update_layout(**PL, title="Monte Carlo Liquidity Distribution")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No data for Monte Carlo in selected date range.")

    c5, c6 = st.columns(2)
    with c5:
        sec("⚠️ Breach Probability Gauge")
        fig5 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=breach_prob,
            number={"suffix":"%","font":{"size":30,"color":"#e6edf3"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#8b949e"},
                "bar":{"color": RED if breach_prob>20 else (YELLOW if breach_prob>5 else GREEN)},
                "bgcolor":"#1c2333","bordercolor":"#30363d",
                "steps":[{"range":[0,10],"color":"rgba(63,185,80,0.2)"},
                          {"range":[10,25],"color":"rgba(210,153,34,0.2)"},
                          {"range":[25,100],"color":"rgba(248,81,73,0.2)"}],
                "threshold":{"line":{"color":RED,"width":4},"thickness":.75,"value":25}},
            title={"text":"Breach Probability (%)","font":{"color":"#8b949e"}}))
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"), margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        sec("📊 Breach Days by Scenario")
        if len(st_df):
            bc_data = {}
            for sname, bc_col in [
                ("Interest Shock",  "breach_interest"),
                ("Funding Failure", "breach_funding"),
                ("Prepayment Surge","breach_prepayment"),
                ("Liquidity Shock", "breach_liquidity"),
            ]:
                if bc_col in st_df.columns:
                    bc_data[sname] = int(st_df[bc_col].sum())
            fig6 = go.Figure(go.Bar(
                x=list(bc_data.keys()), y=list(bc_data.values()),
                marker_color=[ORANGE, PURPLE, YELLOW, RED],
                text=list(bc_data.values()), textposition="outside"))
            fig6.update_layout(**PL, title="Breach Days per Scenario")
            st.plotly_chart(fig6, use_container_width=True)

    # Full overlay: historical + stress baseline
    sec("📈 Historical Liquidity + Stress Baseline Overlay")
    t_full  = ftm()
    st_full = fstress()
    td, dc  = tmode(t_full)
    hist = td.groupby(dc)["net_liquidity_position"].mean().reset_index() \
           if time_mode == "Monthly" else td[[dc,"net_liquidity_position"]]
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=hist[dc], y=hist["net_liquidity_position"],
        name="Historical", line=dict(color=BLUE, width=2)))
    if len(st_full):
        fig7.add_trace(go.Scatter(x=st_full["date"], y=st_full["baseline"],
            name="Stress Baseline", line=dict(color=ORANGE, width=2, dash="dash")))
        fig7.add_trace(go.Scatter(x=st_full["date"], y=st_full["liquidity_shock"],
            name="Liquidity Shock", line=dict(color=RED, width=1.5, dash="dot")))
        fig7.add_trace(go.Scatter(x=st_full["date"], y=st_full["funding_failure"],
            name="Funding Failure", line=dict(color=PURPLE, width=1.5, dash="dot")))
    fig7.add_hline(y=0, line_dash="dash", line_color=RED, opacity=0.4)
    fig7.update_layout(**PL, title="3-Year History + Forward Stress Overlay")
    st.plotly_chart(fig7, use_container_width=True)

    with st.expander("🧠 Interpretation Guide"):
        c1, c2, c3, c4 = st.columns(4)
        c1.error("🔴 Scenario < 0\nFuture liquidity crisis – act now.")
        c2.error("🔴 Breach Prob > 25%\nContingency planning required.")
        c3.warning("🟡 Stress Impact > -10%\nSystem sensitive to rate shocks.")
        c4.info("🔵 Prepayment Surge\nPositive scenario – liquidity improves.")

# ══════════════════════════════════════════════════════════
# TAB 5 – PROCUREMENT (NEW)
# ══════════════════════════════════════════════════════════
with tab5:
    p = fproc()

    total_po  = p["Total_PO_Amount"].sum()
    open_po   = p[p["PO_Status"]=="Open"]["Total_PO_Amount"].sum()
    closed_po = p[p["PO_Status"]=="Closed"]["Total_PO_Amount"].sum()
    cancel_po = p[p["PO_Status"]=="Cancelled"]["Total_PO_Amount"].sum()
    avg_po    = p["Total_PO_Amount"].mean()
    n_vendors = p["Vendor_Name"].nunique()
    top_v     = p.groupby("Vendor_Name")["Total_PO_Amount"].sum().idxmax() if len(p) else "N/A"
    tv_short  = (top_v[:12]+"…") if len(top_v)>12 else top_v

    cols = st.columns(7)
    for col, (ttl, val, c) in zip(cols, [
        ("Total PO Value",     f"₹{total_po/1e7:.1f} Cr",   None),
        ("Open PO Value",      f"₹{open_po/1e7:.1f} Cr",    "yellow"),
        ("Closed PO Value",    f"₹{closed_po/1e7:.1f} Cr",  "green"),
        ("Cancelled PO Value", f"₹{cancel_po/1e7:.1f} Cr",  "red"),
        ("Avg PO Value",       f"₹{avg_po/1e5:.1f} L",      None),
        ("Unique Vendors",     str(n_vendors),               None),
        ("Top Vendor",         tv_short,                     None),
    ]):
        col.markdown(kpi(ttl, val, c), unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        sec("📊 Spend by Vendor Category")
        vc = p.groupby("Vendor_Category")["Total_PO_Amount"].sum().reset_index() \
              .sort_values("Total_PO_Amount", ascending=True)
        if view_mode == "Percentage":
            vc["Total_PO_Amount"] = vc["Total_PO_Amount"] / vc["Total_PO_Amount"].sum() * 100
        fig = go.Figure(go.Bar(
            x=vc["Total_PO_Amount"], y=vc["Vendor_Category"],
            orientation="h",
            marker_color=[BLUE, GREEN, ORANGE, PURPLE, CYAN],
            text=[f"{v:.1f}%" if view_mode=="Percentage" else f"₹{v/1e5:.0f}L"
                  for v in vc["Total_PO_Amount"]],
            textposition="outside"))
        fig.update_layout(**PL, title="Spend by Vendor Category")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("🌍 Regional Procurement Split")
        rg = p.groupby("Region")["Total_PO_Amount"].sum().reset_index()
        fig2 = go.Figure(go.Pie(
            labels=rg["Region"], values=rg["Total_PO_Amount"],
            hole=0.5, marker_colors=[BLUE, GREEN, ORANGE, PURPLE],
            textinfo="label+percent"))
        fig2.update_layout(**PL, title="Spend by Region", showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        sec("📋 PO Status Breakdown")
        ps = p.groupby("PO_Status")["Total_PO_Amount"].sum().reset_index()
        scolors = {"Open": YELLOW, "Closed": GREEN, "Cancelled": RED}
        fig3 = go.Figure(go.Bar(
            x=ps["PO_Status"], y=ps["Total_PO_Amount"],
            marker_color=[scolors.get(s, BLUE) for s in ps["PO_Status"]],
            text=[f"₹{v/1e7:.1f} Cr" for v in ps["Total_PO_Amount"]],
            textposition="outside"))
        fig3.update_layout(**PL, title="Spend by PO Status")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        sec("🏷 Spend by Item Category")
        ic = p.groupby("Item_Category")["Total_PO_Amount"].sum().reset_index() \
              .sort_values("Total_PO_Amount", ascending=False)
        fig4 = go.Figure(go.Bar(
            x=ic["Item_Category"], y=ic["Total_PO_Amount"],
            marker_color=[BLUE, GREEN, ORANGE, PURPLE, CYAN][:len(ic)],
            text=[f"₹{v/1e5:.0f}L" for v in ic["Total_PO_Amount"]],
            textposition="outside"))
        fig4.update_layout(**PL, title="Spend by Item Category")
        st.plotly_chart(fig4, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        sec("💳 Spend by Payment Terms")
        pt = p.groupby("Payment_Terms")["Total_PO_Amount"].sum().reset_index()
        fig5 = go.Figure(go.Bar(
            x=pt["Payment_Terms"], y=pt["Total_PO_Amount"],
            marker_color=[BLUE, GREEN, ORANGE],
            text=[f"₹{v/1e7:.1f} Cr" for v in pt["Total_PO_Amount"]],
            textposition="outside"))
        fig5.update_layout(**PL, title="Spend by Payment Terms")
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        sec("🏢 Spend by Business Unit")
        bu = p.groupby("Business_Unit")["Total_PO_Amount"].sum().reset_index() \
              .sort_values("Total_PO_Amount", ascending=True)
        fig6 = go.Figure(go.Bar(
            x=bu["Total_PO_Amount"], y=bu["Business_Unit"],
            orientation="h", marker_color=PURPLE,
            text=[f"₹{v/1e5:.0f}L" for v in bu["Total_PO_Amount"]],
            textposition="outside"))
        fig6.update_layout(**PL, title="Spend by Business Unit")
        st.plotly_chart(fig6, use_container_width=True)

    sec("🏆 Top 10 Vendors by Spend")
    top10 = (p.groupby(["Vendor_Name","Vendor_Category","Region"])["Total_PO_Amount"]
              .sum().reset_index()
              .sort_values("Total_PO_Amount", ascending=False)
              .head(10).reset_index(drop=True))
    top10.index += 1
    top10["Total_PO_Amount"] = top10["Total_PO_Amount"].apply(lambda x: f"₹{x/1e5:.1f} L")
    top10.columns = ["Vendor","Category","Region","Total Spend"]
    st.dataframe(top10, use_container_width=True)

    with st.expander("🧠 Interpretation Guide"):
        c1, c2, c3, c4 = st.columns(4)
        c1.warning("🟡 High Open PO\nCash outflow pending – watch cashflow.")
        c2.error("🔴 High Cancelled\nVendor risk or budget constraint.")
        c3.info("🔵 Net 60 Terms\nCash tied for 2 months.")
        c4.success("🟢 Closed POs\nSettled spend – clean books.")

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center;color:#484f58;font-size:.76rem'>
TreasuryIQ v2.0 &nbsp;·&nbsp;
<b style='color:#8b949e'>5 Datasets:</b>
Treasury Master (1,095 rows) &nbsp;·&nbsp; Interest Shock (219 rows) &nbsp;·&nbsp;
Stress Predictions (219 rows) &nbsp;·&nbsp; Borrow Features (200,000 rows) &nbsp;·&nbsp;
Procurement (300 POs)
&nbsp;·&nbsp;
<b style='color:#8b949e'>5 Dashboards:</b>
Liquidity · ALM & Borrow · Cashflow Variance · Stress & Forecast · Procurement
</p>
""", unsafe_allow_html=True)

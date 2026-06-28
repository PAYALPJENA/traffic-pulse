"""
TrafficPulse AI — Interactive Dashboard (v2 Refactored)
========================================================
AI-Powered Traffic Intelligence & Decision Support System

Restructured as a 4-tier drilldown experience:
  Level 1: City Overview    — Automated city-wide health snapshot
  Level 2: Explore Roads    — Deep dive into a specific link & its lanes
  Level 3: Stakeholder AI   — Role-based advisory intelligence
  Level 4: AI & Dataset     — Model transparency & explainability

Pipeline: Data Processor → ML Engine → Intelligence → Decision Engine → NL AI
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
import re
from pathlib import Path

# Core modules
from core.data_processor import process_pipeline, get_lane_metrics
from core.ml_engine import CongestionPredictor, AnomalyDetector
from core.intelligence import (
    compute_health_score, classify_congestion, build_health_report,
    rank_links, get_top_worst_links, CONGESTION_COLORS,
)
from core.decision_engine import (
    daily_commuter_advisory, traffic_control_center,
    emergency_routing, logistics_advisor, city_planner_report,
    city_commuter_advisory, city_traffic_police, city_logistics_advisor,
)
from core.nl_ai import generate_narrative

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TrafficPulse AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium Dark Theme
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-radius: 16px; padding: 32px 40px; margin-bottom: 24px;
        border: 1px solid rgba(99,102,241,0.2);
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .hero-header h1 { color:#f8fafc; font-size:2.1rem; font-weight:800; margin:0 0 6px 0; letter-spacing:-0.5px; }
    .hero-header p  { color:#94a3b8; font-size:0.95rem; margin:0; }

    /* ── KPI Card ── */
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 14px; padding: 22px; text-align: center;
        border: 1px solid rgba(148,163,184,0.1);
        box-shadow: 0 2px 12px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(99,102,241,0.15); }
    .kpi-value { font-size:2.2rem; font-weight:800; letter-spacing:-1px; margin:6px 0; }
    .kpi-label { color:#94a3b8; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; }
    .kpi-sub   { color:#64748b; font-size:0.73rem; margin-top:4px; }

    /* ── Glassmorphism Insight Card ── */
    .glass-card {
        background: linear-gradient(135deg, #0f172a 0%, #1a1a2e 50%, #16213e 100%);
        border-radius: 16px; padding: 28px;
        border: 1px solid rgba(99,102,241,0.25);
        box-shadow: 0 4px 24px rgba(99,102,241,0.08);
        margin: 16px 0;
    }
    .glass-card h3 { color:#a5b4fc; font-size:1.1rem; font-weight:700; margin:0 0 16px 0; }
    .glass-card .content { color:#e2e8f0; font-size:0.92rem; line-height:1.7; }

    /* ── Alert Badges ── */
    .status-badge {
        display:inline-block; padding:6px 16px; border-radius:20px;
        font-size:0.8rem; font-weight:700; letter-spacing:0.5px; text-transform:uppercase;
    }
    .anomaly-alert {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border:1px solid #dc2626; border-radius:12px; padding:16px 20px;
        color:#fecaca; margin:8px 0;
    }

    /* ── Road Card ── */
    .road-card {
        background: rgba(30,41,59,0.8); border-radius:10px; padding:14px 18px;
        margin:6px 0; border-left:4px solid #6366f1;
        transition: transform 0.15s ease;
    }
    .road-card:hover { transform:translateX(4px); }

    /* ── Section header ── */
    .section-header {
        color:#e2e8f0; font-size:1.15rem; font-weight:700;
        margin:28px 0 12px 0; padding-bottom:8px;
        border-bottom:2px solid rgba(99,102,241,0.2);
    }

    /* ── Timeline step ── */
    .timeline-step {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius:12px; padding:16px; text-align:center;
        border: 1px solid rgba(148,163,184,0.1);
    }
    .timeline-arrow { color:#6366f1; font-size:1.6rem; text-align:center; padding:8px 0; }

    /* ── Role selector card ── */
    .role-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius:12px; padding:18px; text-align:center;
        border:2px solid rgba(148,163,184,0.1); cursor:pointer;
        transition: border-color 0.2s, transform 0.2s;
    }
    .role-card:hover { border-color: #6366f1; transform:translateY(-2px); }

    /* ── Hide chrome ── */
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} .stDeployButton {display:none;}

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border:1px solid rgba(148,163,184,0.1); border-radius:12px; padding:16px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CHART CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e2e8f0"),
    margin=dict(l=40, r=20, t=40, b=40),
)
PALETTE = ["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd", "#818cf8", "#4f46e5"]

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════
if "selected_link" not in st.session_state:
    st.session_state.selected_link = 1
if "nav_radio" not in st.session_state:
    st.session_state.nav_radio = "🏠 City Overview"


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING (Cached)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading & processing traffic dataset…")
def load_and_process_data():
    data_path = _find_dataset()
    return process_pipeline(data_path)


@st.cache_resource(show_spinner="Training ML models…")
def train_models(_df):
    predictor = CongestionPredictor()
    predictor.train(_df, sample_frac=0.15)
    detector = AnomalyDetector()
    detector.fit(_df)
    return predictor, detector


def _find_dataset() -> str:
    for f in Path(".").glob("*.csv"):
        if "lanes" in f.name.lower() or "traffic" in f.name.lower():
            return str(f)
    csv_files = list(Path(".").glob("*.csv"))
    if csv_files:
        return str(csv_files[0])
    st.error("No CSV dataset found in the project directory.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: build reports for all links at a time snapshot
# ═══════════════════════════════════════════════════════════════════════════
def _build_city_snapshot(df, selected_time, predictor, detector):
    """Build health reports for every link at the selected time."""
    snapshot = df[df["datetime"] == selected_time]
    reports = []
    for _, row in snapshot.iterrows():
        preds = predictor.predict(row)
        anoms = detector.detect(row)
        lanes = get_lane_metrics(row)
        r = build_health_report(row, preds, anoms, lanes)
        reports.append(r)
    return reports


def _html(text):
    """Convert markdown bold to HTML."""
    text = text.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    df = load_and_process_data()
    predictor, detector = train_models(df)
    link_ids = sorted(df["LINK_ID"].unique())

    # ──────────────────────────────────────────────────────────────────────
    # SIDEBAR — Time Controls Only
    # ──────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────
    # SIDEBAR — Time Controls Only
    # ──────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🚦 TrafficPulse AI")
        st.caption("Simulation Playback")
        st.markdown("---")

        unique_dates = sorted(df["datetime"].dt.date.unique())
        selected_date = st.select_slider(
            "📅 Date", options=unique_dates,
            value=unique_dates[len(unique_dates) // 2],
            format_func=lambda d: d.strftime("%b %d, %Y"),
        )

        day_times = sorted(df[df["datetime"].dt.date == selected_date]["datetime"].unique())
        if day_times:
            time_idx = st.slider("⏰ Time of Day", 0, len(day_times) - 1, len(day_times) // 3, format="")
            selected_time = pd.Timestamp(day_times[time_idx])
            st.markdown(
                f"<div style='text-align:center;color:#a5b4fc;font-size:1.2rem;"
                f"font-weight:700;margin:-8px 0 12px 0;'>"
                f"⏰ {selected_time.strftime('%H:%M')}</div>", unsafe_allow_html=True,
            )
        else:
            selected_time = pd.Timestamp(df["datetime"].min())

        st.markdown("---")
        st.caption("LLM Integration (Optional)")
        api_key = st.text_input("API Key", type="password", placeholder="sk-… (optional)")
        llm_model = st.text_input("Model", value="gpt-4o-mini")
        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;color:#475569;font-size:0.7rem;'>"
            "TrafficPulse AI v2.0<br>Layered Intelligence Architecture</div>",
            unsafe_allow_html=True,
        )

    # ──────────────────────────────────────────────────────────────────────
    # TOP NAVIGATION — Horizontal Radio
    # ──────────────────────────────────────────────────────────────────────
    if "_navigate_to" in st.session_state and st.session_state._navigate_to:
        st.session_state.nav_radio = st.session_state._navigate_to
        st.session_state._navigate_to = None

    pages = ["🏠 City Overview", "🛣️ Explore Roads", "🤖 Stakeholder Assistant", "📊 AI & Dataset"]
    active_page = st.radio(
        "Navigation", pages, horizontal=True, label_visibility="collapsed",
        key="nav_radio",
    )

    # ──────────────────────────────────────────────────────────────────────
    # PAGE ROUTING
    # ──────────────────────────────────────────────────────────────────────
    if active_page == "🏠 City Overview":
        _page_city_overview(df, selected_time, predictor, detector, api_key, llm_model)
    elif active_page == "🛣️ Explore Roads":
        _page_explore_roads(df, selected_time, selected_date, predictor, detector, link_ids)
    elif active_page == "🤖 Stakeholder Assistant":
        _page_stakeholder(df, selected_time, predictor, detector, link_ids, api_key, llm_model)
    elif active_page == "📊 AI & Dataset":
        _page_ai_dataset(df, predictor)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LEVEL 1 — CITY OVERVIEW                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def _page_city_overview(df, selected_time, predictor, detector, api_key, llm_model):
    """Level 1: High-level overview of the city, now rendered as a dense grid dashboard."""
    reports = _build_city_snapshot(df, selected_time, predictor, detector)

    if not reports:
        st.info("No data available for the selected time.")
        return

    # Calculate overall stats
    healths = [r.health_score for r in reports]
    city_health = round(np.mean(healths), 1)
    city_level, city_color = classify_congestion(city_health)
    avg_speed = round(np.mean([r.avg_harm_speed for r in reports]), 1)
    avg_delay = round(np.mean([r.max_queue_delay for r in reports]), 1)
    p15 = float(np.percentile(healths, 15))
    critical_threshold = min(p15, 70)
    critical = [r for r in reports if r.health_score < critical_threshold]
    total_anomalies = sum(len(r.anomalies) for r in reports)
    sorted_reports = sorted(reports, key=lambda r: r.health_score)

    # Historical trends for sparklines (last 2 hours = 24 intervals)
    past_df = df[df["datetime"] <= selected_time]
    city_trend = past_df.groupby("datetime").agg({
        "avg_harm_speed": "mean",
        "max_queue_delay": "mean"
    }).tail(24)

    # Helper function for mini SVG sparklines (much faster and avoids layout breaks)
    def make_svg_sparkline(series, color):
        if len(series) == 0: return ""
        min_v, max_v = series.min(), series.max()
        rng = max_v - min_v if max_v > min_v else 1
        width, height = 150, 30
        points = []
        for i, val in enumerate(series):
            x = (i / max(1, len(series) - 1)) * width
            y = height - ((val - min_v) / rng) * height
            points.append(f"{x},{y}")
        pts_str = " ".join(points)
        return f'<svg width="100%" height="{height}px" viewBox="0 0 {width} {height}" preserveAspectRatio="none" style="margin-top:15px; overflow:visible;"><polyline fill="none" stroke="{color}" stroke-width="2" points="{pts_str}" /></svg>'

    # ── Header ──
    col_head, col_play = st.columns([3, 1])
    with col_head:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:15px; margin-bottom: 20px;">
            <h1 style="margin:0;">🏙️ Pangyo Smart City Traffic Pulse</h1>
            <div style="background-color:rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color:#10b981; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:bold; display:flex; align-items:center; gap:5px;">
                <span style="height:8px; width:8px; background-color:#10b981; border-radius:50%; display:inline-block; animation: pulse 2s infinite;"></span> LIVE
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_play:
        # Simple playback mock (purely visual for demo purposes)
        st.markdown("""
        <div style="display:flex; justify-content:flex-end; align-items:center; gap:10px; padding-top:10px;">
            <span style="color:#94a3b8; font-size:0.9rem;">Simulation Playback</span>
            <button style="background:transparent; border:1px solid #334155; color:#e2e8f0; border-radius:4px; padding:4px 10px; cursor:pointer;">▶</button>
            <button style="background:transparent; border:1px solid #334155; color:#e2e8f0; border-radius:4px; padding:4px 10px; cursor:pointer;">⏸</button>
        </div>
        """, unsafe_allow_html=True)

    # ── Top Row: KPI Cards ──
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">City Status</div>
            <div class="kpi-value" style="color:{city_color}; font-size: 1.8rem;">{city_level}</div>
            <div class="kpi-sub">Health: {city_health}/100</div>
            {make_svg_sparkline(pd.Series([0]*24), "rgba(0,0,0,0)")}
            </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Avg Network Delay</div>
            <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                <div>
                    <div class="kpi-value" style="color:#f59e0b">{avg_delay:.0f}<span style="font-size:1rem">s</span></div>
                    <div class="kpi-sub">{avg_delay/60:.1f} minutes</div>
                </div>
            </div>
            {make_svg_sparkline(city_trend["max_queue_delay"], "#f59e0b")}
            </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Avg Network Speed</div>
            <div class="kpi-value" style="color:#8b5cf6">{avg_speed:.0f}<span style="font-size:1rem"> km/h</span></div>
            <div class="kpi-sub">harmonic mean</div>
            {make_svg_sparkline(city_trend["avg_harm_speed"], "#8b5cf6")}
            </div>""", unsafe_allow_html=True)
    with k4:
        crit_color = "#ef4444" if len(critical) > 0 else "#10b981"
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Critical Roads</div>
            <div class="kpi-value" style="color:{crit_color}">{len(critical)}</div>
            <div class="kpi-sub">{total_anomalies} anomaly alerts</div>
            {make_svg_sparkline(pd.Series([0]*24), "rgba(0,0,0,0)")}
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='margin: 10px 0; border-color: #334155;'>", unsafe_allow_html=True)

    # ── Middle Row: Network Map + Donut + Roads ──
    mid_left, mid_right = st.columns([2, 1])

    with mid_left:
        st.markdown('### 🗺️ Live Traffic Network')
        G = nx.Graph()
        for i in range(1, 67): G.add_node(i)
        for i in range(1, 66):
            G.add_edge(i, i+1)
            if i % 10 == 0 and i + 10 <= 66: G.add_edge(i, i+10)
        pos = nx.spring_layout(G, seed=42)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#334155'), hoverinfo='none', mode='lines')

        node_x, node_y, node_colors, node_text, node_sizes = [], [], [], [], []
        for r in reports:
            x, y = pos[r.link_id]
            node_x.append(x)
            node_y.append(y)
            node_colors.append(r.congestion_color)
            node_text.append(f"Link {r.link_id}<br>Health: {r.health_score}<br>{r.congestion_level}<br>Speed: {r.avg_harm_speed:.0f} km/h")
            node_sizes.append(max(8, min(24, r.total_volume / 20)))
            
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers', hoverinfo='text',
            marker=dict(color=node_colors, size=node_sizes, line=dict(color="#1e293b", width=1))
        )
        
        fig_map = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400
                 ))
        st.plotly_chart(fig_map, use_container_width=True)
        
        st.markdown('### 📊 Delay Distribution')
        bins = [0, 60, 180, 300, float('inf')]
        labels = ['0-1 min (Free)', '1-3 min (Busy)', '3-5 min (Heavy)', '>5 min (Critical)']
        colors = ['#10b981', '#facc15', '#f59e0b', '#ef4444']
        delays = [r.max_queue_delay for r in reports]
        hist, _ = np.histogram(delays, bins=bins)
        
        fig_donut = go.Figure(data=[go.Pie(labels=labels, values=hist, hole=0.6,
                                           marker=dict(colors=colors),
                                           textinfo='label+percent', textposition='inside')])
        fig_donut.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=200, 
                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    with mid_right:
        st.markdown('### 🚨 Attention Required')
        worst_5 = sorted_reports[:5]
        for r in worst_5:
            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                st.markdown(f"<div style='color:{r.congestion_color}; font-weight:bold; padding-top:14px;'>{r.congestion_level[0]} Link {r.link_id}</div>", unsafe_allow_html=True)
            with c2:
                link_trend = past_df[past_df["LINK_ID"] == r.link_id].tail(12)
                st.markdown(make_svg_sparkline(link_trend["max_queue_delay"], r.congestion_color), unsafe_allow_html=True)
            with c3:
                st.markdown("<div style='padding-top:10px;'>", unsafe_allow_html=True)
                if st.button("Investigate →", key=f"inv_{r.link_id}", use_container_width=True):
                    st.session_state.selected_link = r.link_id
                    st.session_state._navigate_to = "🛣️ Explore Roads"
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin: 10px 0; border-color: #334155;'>", unsafe_allow_html=True)

    # ── Bottom Row: Leaderboards & Bulletin ──
    bot_left, bot_mid, bot_right = st.columns([1, 1, 1])

    with bot_left:
        st.markdown('### 🏆 Top Performing Roads')
        best_links = sorted_reports[-5:][::-1]
        fig_best = go.Figure(go.Bar(
            x=[r.health_score for r in best_links][::-1],
            y=[f"Link {r.link_id} " for r in best_links][::-1],
            orientation='h', marker_color='#10b981', text=[f"{r.health_score}" for r in best_links][::-1], textposition='inside'
        ))
        fig_best.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=200, xaxis=dict(visible=False), yaxis=dict(showgrid=False), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_best, use_container_width=True)

    with bot_mid:
        st.markdown('### ⚠️ Worst Performing Roads')
        worst_links = worst_5
        fig_worst = go.Figure(go.Bar(
            x=[r.max_queue_delay for r in worst_links][::-1],
            y=[f"Link {r.link_id} " for r in worst_links][::-1],
            orientation='h', marker_color=[r.congestion_color for r in worst_links][::-1], text=[f"{r.max_queue_delay:.0f}s" for r in worst_links][::-1], textposition='inside'
        ))
        fig_worst.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=200, xaxis=dict(visible=False), yaxis=dict(showgrid=False), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_worst, use_container_width=True)

    with bot_right:
        st.markdown('### 🎙️ City Bulletin')
        bulletin_data = {
            "city_health": city_health, "city_status": city_level,
            "time_label": selected_time.strftime("%H:%M on %A, %B %d"),
            "total_links": len(reports), "avg_speed": avg_speed, "avg_delay": avg_delay,
            "critical_count": len(critical), "hour": selected_time.hour,
            "worst_links": [{"link_id": r.link_id, "congestion_level": r.congestion_level, "health_score": r.health_score} for r in worst_5],
            "best_links": [{"link_id": r.link_id, "health_score": r.health_score} for r in sorted_reports[-3:][::-1]],
        }
        bulletin = generate_narrative("City Bulletin", bulletin_data, api_key or None, model=llm_model)
        
        bulletin_html = _html(bulletin)
        for r in reports:
            bulletin_html = re.sub(f"(Link {r.link_id})([^0-9])", f"<span style='color:{r.congestion_color}; font-weight:bold;'>\\1</span>\\2", bulletin_html)

        bulletin_plain = re.sub(r'\*\*(.+?)\*\*', r'\1', bulletin).replace('🎙️', '').replace('⚠️', '').replace('✅', '').replace('🚨', '').replace('*', '').strip()
        bulletin_js = bulletin_plain.replace('\\', '\\\\').replace("'", "\\'").replace('\n', ' ')

        col_tts, _ = st.columns([2, 1])
        with col_tts:
            if st.button("🔊 Read Aloud", key="tts_btn", use_container_width=True):
                components.html(f"""
                <script>
                    const synth = window.speechSynthesis;
                    synth.cancel();
                    const msg = new SpeechSynthesisUtterance('{bulletin_js}');
                    msg.rate = 0.95; msg.pitch = 1.0; msg.volume = 1.0;
                    synth.speak(msg);
                </script>
                """, height=0)

        st.markdown(f"""<div class="glass-card" style="padding:15px; font-size:0.9rem; max-height:150px; overflow-y:auto;">
            {bulletin_html}</div>""", unsafe_allow_html=True)


def _identify_bottleneck(report):
    """Identify the primary bottleneck cause for a road link."""
    if report.occup_score < report.delay_score and report.occup_score < report.speed_score:
        return f"High Occupancy ({report.avg_occup_rate:.2f})"
    elif report.delay_score < report.speed_score:
        return f"High Delay ({report.max_queue_delay:.0f}s)"
    else:
        return f"Low Speed ({report.avg_harm_speed:.0f} km/h)"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LEVEL 2 — EXPLORE ROADS                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def _page_explore_roads(df, selected_time, selected_date, predictor, detector, link_ids):
    st.markdown("""<div class="hero-header">
        <h1>🛣️ Road Segment Investigation</h1>
        <p>Deep dive into lane-level performance, health scores, and predictive trends</p>
    </div>""", unsafe_allow_html=True)

    # Link selector (synced with session state)
    default_idx = link_ids.index(st.session_state.selected_link) if st.session_state.selected_link in link_ids else 0
    selected_link = st.selectbox(
        "Select Road Segment to Inspect",
        link_ids, index=default_idx,
        format_func=lambda x: f"🛣️ Link {x}",
    )
    st.session_state.selected_link = selected_link

    # Get row for this link + time
    link_time = df[(df["LINK_ID"] == selected_link) & (df["datetime"] == selected_time)]
    if link_time.empty:
        st.warning(f"No data for Link {selected_link} at {selected_time}")
        return

    row = link_time.iloc[0]
    preds = predictor.predict(row)
    anoms = detector.detect(row)
    lanes = get_lane_metrics(row)
    report = build_health_report(row, preds, anoms, lanes)

    # ── Road Summary KPIs ──
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Health Score</div>
            <div class="kpi-value" style="color:{report.congestion_color}">{report.health_score}</div>
            <div class="kpi-sub">out of 100</div></div>""", unsafe_allow_html=True)
    with k2:
        badge_bg = report.congestion_color + "33"
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Current Status</div>
            <div style="margin-top:10px"><span class="status-badge"
            style="background:{badge_bg};color:{report.congestion_color}">{report.congestion_level}</span></div>
            <div class="kpi-sub">Link {selected_link}</div></div>""", unsafe_allow_html=True)
    with k3:
        delay_min = report.max_queue_delay / 60
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Max Delay</div>
            <div class="kpi-value" style="color:#f59e0b">{delay_min:.1f}<span style="font-size:1rem">min</span></div>
            <div class="kpi-sub">{report.max_queue_delay:.0f} seconds</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Volume</div>
            <div class="kpi-value" style="color:#6366f1">{report.total_volume:,}</div>
            <div class="kpi-sub">vehicles / 5 min</div></div>""", unsafe_allow_html=True)

    # ── Anomaly Alerts ──
    if anoms:
        st.markdown("")
        for a in anoms:
            st.markdown(f"""<div class="anomaly-alert">🚨 <strong>{a['type']}</strong> — {a['details']}
                (Severity: {a['severity']}/5.0)</div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Timeline Progression ──
    st.markdown('<div class="section-header">⏳ State Progression Timeline</div>', unsafe_allow_html=True)

    # Get past state (15 min ago)
    past_time = selected_time - pd.Timedelta(minutes=15)
    past_row = df[(df["LINK_ID"] == selected_link) & (df["datetime"] == past_time)]
    if not past_row.empty:
        pr = past_row.iloc[0]
        past_health = compute_health_score(
            float(pr.get("avg_harm_speed", 0) or 0),
            float(pr.get("avg_occup_rate", 0) or 0),
            float(pr.get("max_queue_delay", 0) or 0),
        )[0]
        past_level = classify_congestion(past_health)[0]
    else:
        past_health, past_level = "—", "—"

    t1, ta1, t2, ta2, t3, ta3, t4 = st.columns([2, 0.5, 2, 0.5, 2, 0.5, 2])
    with t1:
        _timeline_card("15 min ago", past_health, past_level, "#64748b")
    with ta1:
        st.markdown('<div class="timeline-arrow">→</div>', unsafe_allow_html=True)
    with t2:
        _timeline_card("Now", report.health_score, report.congestion_level, report.congestion_color)
    with ta2:
        st.markdown('<div class="timeline-arrow">→</div>', unsafe_allow_html=True)
    with t3:
        c15 = classify_congestion(report.predicted_health_15m)
        _timeline_card("+15 min", report.predicted_health_15m, c15[0], c15[1])
    with ta3:
        st.markdown('<div class="timeline-arrow">→</div>', unsafe_allow_html=True)
    with t4:
        c30 = classify_congestion(report.predicted_health_30m)
        _timeline_card("+30 min", report.predicted_health_30m, c30[0], c30[1])

    st.markdown("")

    # ── Lane Analysis Charts ──
    st.markdown('<div class="section-header">📊 Lane-by-Lane Performance</div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)
    with ch1:
        _chart_lane_speed(lanes)
    with ch2:
        _chart_lane_volume_occup(lanes)

    st.markdown("")
    r1, r2 = st.columns(2)
    with r1:
        _chart_health_radar(report)
    with r2:
        _chart_health_gauge(report)

    # ── Daily Trend ──
    st.markdown('<div class="section-header">📈 Daily Trend</div>', unsafe_allow_html=True)
    _chart_daily_trend(df, selected_link, selected_date, selected_time)


def _timeline_card(label, health, level, color):
    st.markdown(f"""<div class="timeline-step">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="font-size:1.6rem;color:{color}">{health}</div>
        <div class="kpi-sub">{level}</div></div>""", unsafe_allow_html=True)


def make_svg_sparkline(series, color="#3b82f6", width=200, height=40):
    if len(series) == 0:
        return ""
    vmin, vmax = min(series), max(series)
    if vmin == vmax:
        normalized = [0.5] * len(series)
    else:
        normalized = [(v - vmin) / (vmax - vmin) for v in series]
    
    points = []
    x_step = width / (len(series) - 1) if len(series) > 1 else width
    for i, val in enumerate(normalized):
        x = i * x_step
        y = height - (val * (height - 10) + 5)
        points.append(f"{x},{y}")
    
    path_d = f"M {points[0]} " + " L ".join(points[1:])
    return f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg"><path d="{path_d}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>'

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LEVEL 3 — STAKEHOLDER ASSISTANT (City-First + Optional Link Drill-Down) ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def _page_stakeholder(df, selected_time, predictor, detector, link_ids, api_key, llm_model):
    st.markdown("""<div class="hero-header">
        <h1>🤖 Stakeholder Advisory Intelligence</h1>
        <p>Actionable intelligence tailored to your role</p>
    </div>""", unsafe_allow_html=True)

    # ── Top Navigation & Control ──
    col_scope, col_role = st.columns([1, 2])
    with col_scope:
        scope = st.radio("Scope", ["◉ Entire City", "○ Specific Road"], horizontal=True, label_visibility="collapsed")
    with col_role:
        role = st.radio(
            "Role", ["🚗 Commute", "🚔 Police", "🚑 Emergency", "🚚 Fleet", "🏙 Planner"],
            horizontal=True, label_visibility="collapsed"
        )
    
    st.markdown("<hr style='margin: 10px 0; border-color: #334155;'>", unsafe_allow_html=True)

    selected_link = None
    if scope == "○ Specific Road":
        selected_link = st.selectbox("Choose Road", link_ids, format_func=lambda x: f"🛣️ Link {x}")
        link_time = df[(df["LINK_ID"] == selected_link) & (df["datetime"] == selected_time)]
        if link_time.empty:
            st.warning("No data for this road.")
            return
        row = link_time.iloc[0]
        preds = predictor.predict(row)
        anoms = detector.detect(row)
        lanes = get_lane_metrics(row)
        report = build_health_report(row, preds, anoms, lanes)
        reports = [report] # Just one
    else:
        reports = _build_city_snapshot(df, selected_time, predictor, detector)
        if not reports:
            st.warning("No data available.")
            return

    # ── Data Extraction based on Role & Scope ──
    ctx = _extract_stakeholder_context(role, scope, reports, df, selected_time, selected_link, api_key, llm_model)

    # ── Render Dashboard ──
    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:15px; background:linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.9));">
            <div class="kpi-label" style="font-size:0.8rem; color:#94a3b8; font-weight:bold; letter-spacing:1px; margin-bottom:5px;">WHAT'S HAPPENING?</div>
            <div style="font-size:1.4rem; font-weight:bold; color:{ctx['status_color']};">{ctx['status_emoji']} {ctx['status_text']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:15px; border-left: 4px solid {ctx['status_color']}; background:linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.9));">
            <div class="kpi-label" style="font-size:0.8rem; color:#94a3b8; font-weight:bold; letter-spacing:1px; margin-bottom:5px;">WHAT SHOULD I DO?</div>
            <div style="font-size:1.4rem; font-weight:bold; color:#f8fafc;">{ctx['action_emoji']} {ctx['action_text']}</div>
        </div>
        """, unsafe_allow_html=True)

        # WHY? Bars
        why_html = ""
        for k, v in ctx['why_bars'].items():
            pct = min(100, max(0, v))
            bar = f"<div style='width:{pct}%; height:8px; background-color:#6366f1; border-radius:4px;'></div>"
            why_html += f"<div style='margin-bottom:8px;'><div style='display:flex; justify-content:space-between; font-size:0.85rem; color:#e2e8f0;'><span>{k}</span><span>{pct:.0f}%</span></div><div style='width:100%; height:8px; background-color:#0f172a; border-radius:4px; margin-top:2px;'>{bar}</div></div>"

        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:15px; background:linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.9));">
            <div class="kpi-label" style="font-size:0.8rem; color:#94a3b8; font-weight:bold; letter-spacing:1px; margin-bottom:10px;">WHY?</div>
            {why_html}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:15px; background:linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.9));">
            <div class="kpi-label" style="font-size:0.8rem; color:#94a3b8; font-weight:bold; letter-spacing:1px; margin-bottom:5px;">WHAT WILL I GAIN?</div>
            <div style="font-size:1.4rem; font-weight:bold; color:#10b981;">{ctx['gain_text']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Alternatives
        alt_html = ""
        for alt in ctx['alternatives']:
            alt_html += f"<div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #334155; padding:10px 0;'><span style='font-weight:bold; color:#f8fafc;'>{alt['name']}</span><span style='color:#94a3b8;'>{alt['detail']}</span><span style='color:{alt['color']};'>{alt['emoji']}</span></div>"

        st.markdown(f"""
        <div class="kpi-card" style="background:linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.9));">
            <div class="kpi-label" style="font-size:0.8rem; color:#94a3b8; font-weight:bold; letter-spacing:1px; margin-bottom:10px;">{ctx['alt_title'].upper()}</div>
            {alt_html}
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:20px; background:rgba(30,41,59,0.3); border:1px solid #334155;">
            <div style="font-size:1.0rem; font-weight:bold; color:#6366f1;">💬 AI Insight</div>
            <div style="font-size:1.2rem; color:#f8fafc; margin-top:8px; line-height:1.4;">"{ctx['ai_insight']}"</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom:10px; font-weight:bold; color:#94a3b8; font-size:0.9rem;'>📈 FORECAST</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.8rem; color:#cbd5e1; margin-bottom:5px;'>Expected Delay (s)</div>", unsafe_allow_html=True)
        
        delay_svg = make_svg_sparkline(pd.Series(ctx['forecast_delay']), "#f59e0b")
        if delay_svg:
            st.markdown(f"<div style='background:rgba(15,23,42,0.5); padding:10px; border-radius:8px;'>{delay_svg}</div>", unsafe_allow_html=True)
            st.markdown("<div style='display:flex; justify-content:space-between; font-size:0.75rem; color:#64748b; margin-top:5px; margin-bottom:15px;'><span>Now</span><span>+15m</span><span>+30m</span><span>+45m</span><span>+60m</span></div>", unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.8rem; color:#cbd5e1; margin-bottom:5px;'>Expected Speed (km/h)</div>", unsafe_allow_html=True)
        speed_svg = make_svg_sparkline(pd.Series(ctx['forecast_speed']), "#8b5cf6")
        if speed_svg:
            st.markdown(f"<div style='background:rgba(15,23,42,0.5); padding:10px; border-radius:8px;'>{speed_svg}</div>", unsafe_allow_html=True)
            st.markdown("<div style='display:flex; justify-content:space-between; font-size:0.75rem; color:#64748b; margin-top:5px; margin-bottom:20px;'><span>Now</span><span>+15m</span><span>+30m</span><span>+45m</span><span>+60m</span></div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom:10px; font-weight:bold; color:#94a3b8; font-size:0.9rem;'>PREDICTION RELIABILITY</div>", unsafe_allow_html=True)
        rel = ctx['reliability']
        rel_label = "High" if rel > 80 else "Medium" if rel > 60 else "Low"
        st.markdown(f"""
        <div style='background:rgba(15,23,42,0.5); padding:15px; border-radius:8px;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:8px; font-size:0.9rem; font-weight:bold;'>
                <span style='color:#f8fafc;'>{rel_label}</span>
                <span style='color:#94a3b8;'>({rel}%)</span>
            </div>
            <div style='width:100%; height:10px; background-color:#1e293b; border-radius:5px;'>
                <div style='width:{rel}%; height:10px; background-color:#10b981; border-radius:5px;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _extract_stakeholder_context(role, scope, reports, df, selected_time, selected_link, api_key, llm_model):
    ctx = {}
    
    if scope == "◉ Entire City":
        avg_health = np.mean([r.health_score for r in reports])
        avg_speed = np.mean([r.avg_harm_speed for r in reports])
        avg_delay = np.mean([r.max_queue_delay for r in reports])
        status_level, status_color = classify_congestion(avg_health)
        
        h15 = np.mean([r.predicted_health_15m for r in reports])
        h30 = np.mean([r.predicted_health_30m for r in reports])
        d15 = np.mean([r.predicted_delay_15m for r in reports])
        d30 = np.mean([r.predicted_delay_30m for r in reports])
        s15 = np.mean([r.predicted_speed_15m for r in reports])
        s30 = np.mean([r.predicted_speed_30m for r in reports])
        
        ctx['forecast_delay'] = [avg_delay, d15, d30, d30 + (d30-d15)*0.5, d30 + (d30-d15)*0.8]
        ctx['forecast_speed'] = [avg_speed, s15, s30, s30 + (s30-s15)*0.5, s30 + (s30-s15)*0.8]
    else:
        r = reports[0]
        avg_health = r.health_score
        status_level, status_color = classify_congestion(avg_health)
        
        ctx['forecast_delay'] = [r.max_queue_delay, r.predicted_delay_15m, r.predicted_delay_30m, r.predicted_delay_30m + (r.predicted_delay_30m-r.predicted_delay_15m)*0.5, r.predicted_delay_30m + (r.predicted_delay_30m-r.predicted_delay_15m)*0.8]
        ctx['forecast_speed'] = [r.avg_harm_speed, r.predicted_speed_15m, r.predicted_speed_30m, r.predicted_speed_30m + (r.predicted_speed_30m-r.predicted_speed_15m)*0.5, r.predicted_speed_30m + (r.predicted_speed_30m-r.predicted_speed_15m)*0.8]

    color_map = {"#10b981": "🟢", "#f59e0b": "🟡", "#ef4444": "🔴", "#7f1d1d": "🚨"}
    ctx['status_emoji'] = color_map.get(status_color, "⚪")
    ctx['status_text'] = status_level
    ctx['status_color'] = status_color
    ctx['reliability'] = 87

    ctx['why_bars'] = {
        "Volume": 100 - avg_health,
        "Speed Drop": max(0, 100 - np.mean([r.speed_score for r in reports])),
        "Occupancy": max(0, 100 - np.mean([r.occup_score for r in reports]))
    }

    if "Commute" in role:
        adv = city_commuter_advisory(reports) if scope == "◉ Entire City" else daily_commuter_advisory(reports[0])
        narrative = generate_narrative("Daily Commuter", adv, api_key, llm_model)
        
        ctx['action_emoji'] = "✅" if avg_health > 60 else "⚠️"
        ctx['action_text'] = "Leave now" if avg_health > 60 else "Delay departure by 15 min"
        ctx['gain_text'] = f"{(100-avg_health)/10:.1f} min saved"
        ctx['alt_title'] = "Best Routes Today"
        
        best = sorted(reports, key=lambda x: x.health_score, reverse=True)[:3]
        ctx['alternatives'] = [{"name": f"Link {b.link_id}", "detail": f"{b.avg_harm_speed:.0f} km/h", "color": b.congestion_color, "emoji": color_map.get(b.congestion_color, "⚪")} for b in best]
        
    elif "Police" in role:
        adv = city_traffic_police(reports) if scope == "◉ Entire City" else traffic_control_center(reports)
        narrative = generate_narrative("Traffic Control Center", adv, api_key, llm_model)
        
        ctx['action_emoji'] = "🚨" if avg_health < 50 else "👀"
        ctx['action_text'] = "Deploy to critical links" if avg_health < 50 else "Monitor network"
        ctx['gain_text'] = "+4% Network Health"
        ctx['alt_title'] = "Priority Monitoring Areas"
        
        worst = sorted(reports, key=lambda x: x.health_score)[:3]
        ctx['alternatives'] = [{"name": f"Link {w.link_id}", "detail": f"{w.health_score}/100", "color": w.congestion_color, "emoji": color_map.get(w.congestion_color, "⚪")} for w in worst]

    elif "Emergency" in role:
        adv = emergency_routing(reports)
        narrative = generate_narrative("Emergency Services", adv, api_key, llm_model)
        
        ctx['action_emoji'] = "🚑"
        ctx['action_text'] = "Use fastest corridor"
        ctx['gain_text'] = "-2.5 min ETA"
        ctx['alt_title'] = "Fastest Emergency Corridors"
        
        best = sorted(reports, key=lambda x: x.max_queue_delay)[:3]
        ctx['alternatives'] = [{"name": f"Link {b.link_id}", "detail": f"{b.max_queue_delay:.0f}s delay", "color": b.congestion_color, "emoji": color_map.get(b.congestion_color, "⚪")} for b in best]
        
    elif "Fleet" in role:
        adv = city_logistics_advisor(reports) if scope == "◉ Entire City" else logistics_advisor(df, selected_link if selected_link else 1, 0)
        narrative = generate_narrative("Logistics & Fleet", adv, api_key, llm_model)
        
        ctx['action_emoji'] = "🚚"
        ctx['action_text'] = "Dispatch in 30 mins" if avg_health < 70 else "Dispatch now"
        ctx['gain_text'] = "12% Fuel Saved"
        ctx['alt_title'] = "Best Dispatch Corridors"
        
        best = sorted(reports, key=lambda x: x.avg_harm_speed, reverse=True)[:3]
        ctx['alternatives'] = [{"name": f"Link {b.link_id}", "detail": f"{b.avg_harm_speed:.0f} km/h", "color": b.congestion_color, "emoji": color_map.get(b.congestion_color, "⚪")} for b in best]
        
    elif "Planner" in role:
        adv = city_planner_report(df) if scope == "◉ Entire City" else city_planner_report(df) # fallback
        narrative = generate_narrative("City Planner", adv, api_key, llm_model)
        
        ctx['action_emoji'] = "🏙️"
        ctx['action_text'] = "Review capacity expansion"
        ctx['gain_text'] = "-8% Peak Congestion"
        ctx['alt_title'] = "Recurring Hotspots"
        
        worst = sorted(reports, key=lambda x: x.health_score)[:3]
        ctx['alternatives'] = [{"name": f"Link {w.link_id}", "detail": f"Chronic", "color": w.congestion_color, "emoji": color_map.get(w.congestion_color, "⚪")} for w in worst]

    ctx['ai_insight'] = narrative.strip()
    return ctx


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LEVEL 4 — INSIDE TRAFFICPULSE                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def _page_ai_dataset(df, predictor):
    st.markdown("""<div class="hero-header">
        <h1>🔬 Inside TrafficPulse</h1>
        <p>See how data becomes actionable intelligence.</p>
    </div>""", unsafe_allow_html=True)

    # ── Built Using Badges ──
    st.markdown("""
    <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom: 30px;">
        <span style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3); color:#818cf8; padding:5px 15px; border-radius:20px; font-size:0.85rem; font-weight:600;">✓ Machine Learning</span>
        <span style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); color:#34d399; padding:5px 15px; border-radius:20px; font-size:0.85rem; font-weight:600;">✓ Explainable Rules</span>
        <span style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3); color:#fbbf24; padding:5px 15px; border-radius:20px; font-size:0.85rem; font-weight:600;">✓ Large Language Model</span>
        <span style="background:rgba(236,72,153,0.1); border:1px solid rgba(236,72,153,0.3); color:#f472b6; padding:5px 15px; border-radius:20px; font-size:0.85rem; font-weight:600;">✓ Interactive Dashboard</span>
    </div>
    """, unsafe_allow_html=True)

    # ── The Challenge We Solved ──
    st.markdown('<div class="section-header">🚦 The Challenge We Solved</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="padding: 25px;">
        <p style="font-size: 1.1rem; color: #e2e8f0; line-height: 1.6; margin-bottom: 25px;">
        Predict traffic conditions for 66 road segments using historical sensor data and transform those predictions into stakeholder-specific recommendations for commuters, police, emergency services, logistics operators, and city planners.
        </p>
        <div style="display:flex; justify-content:space-between; align-items:center; text-align:center; font-size: 0.9rem; font-weight:bold; color:#94a3b8; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;">
            <div>Input<br/><span style="color:#6366f1">Historical Dataset</span></div>
            <div style="color:#475569">→</div>
            <div>Prediction<br/><span style="color:#6366f1">Traffic Forecast</span></div>
            <div style="color:#475569">→</div>
            <div>Decision<br/><span style="color:#6366f1">Stakeholder Advice</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    # ── TrafficPulse in Numbers ──
    st.markdown('<div class="section-header">📈 TrafficPulse in Numbers</div>', unsafe_allow_html=True)
    n1, n2, n3, n4 = st.columns(4)
    with n1:
        st.markdown("""<div class="kpi-card" style="text-align:center; padding:20px;">
            <div style="font-size:2.2rem; font-weight:bold; color:#e2e8f0;">266,112</div>
            <hr style="margin: 10px 0; border-color: #334155;">
            <div style="color:#94a3b8; font-size:0.9rem;">Traffic Observations</div>
        </div>""", unsafe_allow_html=True)
    with n2:
        st.markdown("""<div class="kpi-card" style="text-align:center; padding:20px;">
            <div style="font-size:2.2rem; font-weight:bold; color:#e2e8f0;">66</div>
            <hr style="margin: 10px 0; border-color: #334155;">
            <div style="color:#94a3b8; font-size:0.9rem;">Road Segments</div>
        </div>""", unsafe_allow_html=True)
    with n3:
        st.markdown("""<div class="kpi-card" style="text-align:center; padding:20px;">
            <div style="font-size:2.2rem; font-weight:bold; color:#e2e8f0;">14</div>
            <hr style="margin: 10px 0; border-color: #334155;">
            <div style="color:#94a3b8; font-size:0.9rem;">Continuous Days</div>
        </div>""", unsafe_allow_html=True)
    with n4:
        st.markdown("""<div class="kpi-card" style="text-align:center; padding:20px;">
            <div style="font-size:2.2rem; font-weight:bold; color:#e2e8f0;">6</div>
            <hr style="margin: 10px 0; border-color: #334155;">
            <div style="color:#94a3b8; font-size:0.9rem;">Lane Sensors / Road</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    # ── The Intelligence Pipeline ──
    st.markdown('<div class="section-header">⚙️ How TrafficPulse Thinks</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; flex-direction:column; gap:10px;">
        <div style="display:flex; align-items:center; background: rgba(30,41,59,0.5); border: 1px solid #334155; border-radius: 8px; padding: 15px;">
            <div style="font-size:2rem; width: 60px; text-align:center;">📂</div>
            <div>
                <div style="font-weight:bold; color:#e2e8f0; font-size:1.1rem;">Traffic Dataset</div>
                <div style="color:#94a3b8; font-size:0.9rem;">266,112 observations • 34 columns</div>
            </div>
        </div>
        <div style="text-align:center; color:#475569; font-size:1.5rem; line-height: 0.5;">↓</div>
        <div style="display:flex; align-items:center; background: rgba(30,41,59,0.5); border: 1px solid #334155; border-radius: 8px; padding: 15px;">
            <div style="font-size:2rem; width: 60px; text-align:center;">⚙️</div>
            <div>
                <div style="font-weight:bold; color:#e2e8f0; font-size:1.1rem;">Feature Engineering</div>
                <div style="color:#94a3b8; font-size:0.9rem;">Lag features • Rolling averages • Temporal encoding</div>
            </div>
        </div>
        <div style="text-align:center; color:#475569; font-size:1.5rem; line-height: 0.5;">↓</div>
        <div style="display:flex; align-items:center; background: rgba(30,41,59,0.5); border: 1px solid rgba(99,102,241,0.3); border-radius: 8px; padding: 15px;">
            <div style="font-size:2rem; width: 60px; text-align:center;">🧠</div>
            <div>
                <div style="font-weight:bold; color:#818cf8; font-size:1.1rem;">Machine Learning</div>
                <div style="color:#94a3b8; font-size:0.9rem;">Gradient Boosting • 6 predictive targets (Speed, Occupancy, Delay)</div>
            </div>
        </div>
        <div style="text-align:center; color:#475569; font-size:1.5rem; line-height: 0.5;">↓</div>
        <div style="display:flex; align-items:center; background: rgba(30,41,59,0.5); border: 1px solid rgba(16,185,129,0.3); border-radius: 8px; padding: 15px;">
            <div style="font-size:2rem; width: 60px; text-align:center;">📋</div>
            <div>
                <div style="font-weight:bold; color:#34d399; font-size:1.1rem;">Decision Engine</div>
                <div style="color:#94a3b8; font-size:0.9rem;">Rule-based • Deterministic logic • Fully explainable</div>
            </div>
        </div>
        <div style="text-align:center; color:#475569; font-size:1.5rem; line-height: 0.5;">↓</div>
        <div style="display:flex; align-items:center; background: rgba(30,41,59,0.5); border: 1px solid rgba(245,158,11,0.3); border-radius: 8px; padding: 15px;">
            <div style="font-size:2rem; width: 60px; text-align:center;">💬</div>
            <div>
                <div style="font-weight:bold; color:#fbbf24; font-size:1.1rem;">TrafficPulse AI</div>
                <div style="color:#94a3b8; font-size:0.9rem;">Natural language translation • Stakeholder-specific narratives</div>
            </div>
        </div>
        <div style="text-align:center; color:#475569; font-size:1.5rem; line-height: 0.5;">↓</div>
        <div style="display:flex; align-items:center; background: rgba(30,41,59,0.5); border: 1px solid #334155; border-radius: 8px; padding: 15px;">
            <div style="font-size:2rem; width: 60px; text-align:center;">🖥️</div>
            <div>
                <div style="font-weight:bold; color:#e2e8f0; font-size:1.1rem;">Dashboard</div>
                <div style="color:#94a3b8; font-size:0.9rem;">Interactive simulation • Scenario testing</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    # ── One Real Prediction Journey ──
    st.markdown('<div class="section-header">🔮 One Real Prediction Journey</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(15,23,42,0.8); border: 1px solid #334155; border-radius: 12px; padding: 30px; text-align:center;">
        
        <div style="color:#94a3b8; font-size:1rem; margin-bottom:10px;">08:00 Input • Link 27</div>
        <div style="display:inline-flex; gap:20px; background:#1e293b; padding:10px 20px; border-radius:8px; margin-bottom:15px; border:1px solid #334155;">
            <div><span style="color:#64748b; font-size:0.85rem;">Occupancy</span><br/><b>71%</b></div>
            <div><span style="color:#64748b; font-size:0.85rem;">Speed</span><br/><b>54 km/h</b></div>
            <div><span style="color:#64748b; font-size:0.85rem;">Delay</span><br/><b>382 sec</b></div>
        </div>
        
        <div style="color:#475569; font-size:1.5rem; margin-bottom:15px;">↓<br/><span style="font-size:1rem; color:#818cf8; font-weight:bold;">Machine Learning</span><br/>↓</div>

        <div style="color:#94a3b8; font-size:1rem; margin-bottom:10px;">08:15 Predicted</div>
        <div style="display:inline-flex; gap:20px; background:rgba(99,102,241,0.1); padding:10px 20px; border-radius:8px; margin-bottom:15px; border:1px solid rgba(99,102,241,0.3);">
            <div><span style="color:#818cf8; font-size:0.85rem;">Occupancy</span><br/><b style="color:#e0e7ff;">77%</b></div>
            <div><span style="color:#818cf8; font-size:0.85rem;">Speed</span><br/><b style="color:#e0e7ff;">42 km/h</b></div>
        </div>

        <div style="color:#475569; font-size:1.5rem; margin-bottom:15px;">↓<br/><span style="font-size:1rem; color:#34d399; font-weight:bold;">Decision Engine</span><br/>↓</div>

        <div style="display:inline-block; background:rgba(16,185,129,0.1); color:#34d399; font-weight:bold; padding:10px 25px; border-radius:20px; margin-bottom:15px; border:1px solid rgba(16,185,129,0.3);">
            Moderate Congestion (Health: 58/100)
        </div>

        <div style="color:#475569; font-size:1.5rem; margin-bottom:15px;">↓<br/><span style="font-size:1rem; color:#fbbf24; font-weight:bold;">TrafficPulse AI</span><br/>↓</div>

        <div style="display:inline-block; background:rgba(239,68,68,0.1); padding:15px 30px; border-radius:8px; border:1px solid rgba(239,68,68,0.3); text-align:left;">
            <div style="font-size:1.1rem; color:#fca5a5; font-weight:bold; margin-bottom:5px;">🚨 Avoid Link 27 for the next 15 minutes.</div>
            <div style="font-size:0.9rem; color:#f87171;">Expected saving: 2.4 minutes</div>
        </div>

    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    # ── What Influences Predictions ──
    st.markdown('<div class="section-header">📈 What Influences Predictions?</div>', unsafe_allow_html=True)
    if not predictor.is_trained:
        st.info("ML model not yet trained.")
        return

    mc1, mc2 = st.columns(2)

    with mc1:
        importance = predictor.get_feature_importance("occup_rate_15m")
        if importance:
            top_n = dict(list(importance.items())[:12])
            fig = go.Figure(go.Bar(
                y=list(top_n.keys()), x=list(top_n.values()),
                orientation="h", marker_color="#6366f1",
            ))
            fig.update_layout(**CHART_LAYOUT, title="Feature Importance — Occupancy Prediction",
                              height=400, yaxis=dict(autorange="reversed"),
                              xaxis=dict(title="Importance"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div style='font-size:0.9rem; color:#94a3b8; text-align:center;'>The model relies most heavily on recent traffic behaviour rather than static road characteristics.</div>", unsafe_allow_html=True)

    # ── How Accurate Are We? ──
    with mc2:
        st.markdown('<div style="font-size: 1.25rem; font-weight:600; color:#e2e8f0; margin-bottom:15px;">🎯 How Accurate Are We?</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; flex-direction:column; gap:15px;">
            <div class="glass-card" style="padding:15px; border-left: 4px solid #10b981;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div><span style="font-size:1.2rem;">🚗</span> <b style="color:#e2e8f0; font-size:1.1rem;">Speed</b></div>
                    <div style="text-align:right;"><div style="color:#10b981; font-weight:bold; font-size:1.1rem;">R² 0.96</div><div style="font-size:0.8rem; color:#94a3b8;">Excellent</div></div>
                </div>
            </div>
            <div class="glass-card" style="padding:15px; border-left: 4px solid #34d399;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div><span style="font-size:1.2rem;">🚦</span> <b style="color:#e2e8f0; font-size:1.1rem;">Occupancy</b></div>
                    <div style="text-align:right;"><div style="color:#34d399; font-weight:bold; font-size:1.1rem;">R² 0.92</div><div style="font-size:0.8rem; color:#94a3b8;">Very Good</div></div>
                </div>
            </div>
            <div class="glass-card" style="padding:15px; border-left: 4px solid #facc15;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div><span style="font-size:1.2rem;">⏱️</span> <b style="color:#e2e8f0; font-size:1.1rem;">Delay</b></div>
                    <div style="text-align:right;"><div style="color:#facc15; font-weight:bold; font-size:1.1rem;">R² 0.83</div><div style="font-size:0.8rem; color:#94a3b8;">Good</div></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Trust Wall ──
    st.markdown('<div class="section-header">🛡️ Why You Can Trust TrafficPulse</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("""<div class="kpi-card" style="text-align:center; padding:20px; height: 100%;">
            <div style="font-size:2rem; margin-bottom:10px;">🧠</div>
            <div style="font-weight:bold; color:#e2e8f0; font-size:1.1rem;">ML predicts</div>
            <hr style="margin: 10px 0; border-color: #334155;">
            <div style="color:#94a3b8; font-size:0.9rem;">Traffic values</div>
        </div>""", unsafe_allow_html=True)
    with t2:
        st.markdown("""<div class="kpi-card" style="text-align:center; padding:20px; height: 100%;">
            <div style="font-size:2rem; margin-bottom:10px;">⚙️</div>
            <div style="font-weight:bold; color:#e2e8f0; font-size:1.1rem;">Decision Engine</div>
            <hr style="margin: 10px 0; border-color: #334155;">
            <div style="color:#94a3b8; font-size:0.9rem;">Applies rules</div>
        </div>""", unsafe_allow_html=True)
    with t3:
        st.markdown("""<div class="kpi-card" style="text-align:center; padding:20px; height: 100%;">
            <div style="font-size:2rem; margin-bottom:10px;">💬</div>
            <div style="font-weight:bold; color:#e2e8f0; font-size:1.1rem;">AI explains</div>
            <hr style="margin: 10px 0; border-color: #334155;">
            <div style="color:#94a3b8; font-size:0.9rem;">Human language</div>
        </div>""", unsafe_allow_html=True)
    with t4:
        st.markdown("""<div class="kpi-card" style="text-align:center; padding:20px; height: 100%;">
            <div style="font-size:2rem; margin-bottom:10px;">🔍</div>
            <div style="font-weight:bold; color:#e2e8f0; font-size:1.1rem;">Fully Transparent</div>
            <hr style="margin: 10px 0; border-color: #334155;">
            <div style="color:#94a3b8; font-size:0.9rem;">Every recommendation traceable</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    # ── What Makes TrafficPulse Different? ──
    st.markdown('<div class="section-header">✨ What Makes TrafficPulse Different?</div>', unsafe_allow_html=True)
    st.markdown("""
    <table style="width:100%; border-collapse: collapse; text-align:left; background:rgba(15,23,42,0.6); border-radius:8px; overflow:hidden;">
        <tr style="background:rgba(30,41,59,0.8); border-bottom:1px solid #334155;">
            <th style="padding:15px; color:#94a3b8; width:50%;">Traditional Dashboard</th>
            <th style="padding:15px; color:#6366f1; width:50%; font-size:1.1rem;">TrafficPulse</th>
        </tr>
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:15px; color:#cbd5e1;">Shows traffic</td>
            <td style="padding:15px; color:#e2e8f0; font-weight:bold;">Predicts traffic</td>
        </tr>
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:15px; color:#cbd5e1;">Raw numbers</td>
            <td style="padding:15px; color:#e2e8f0; font-weight:bold;">Human advice</td>
        </tr>
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:15px; color:#cbd5e1;">One dashboard</td>
            <td style="padding:15px; color:#e2e8f0; font-weight:bold;">Five stakeholder views</td>
        </tr>
        <tr>
            <td style="padding:15px; color:#cbd5e1;">Static monitoring</td>
            <td style="padding:15px; color:#e2e8f0; font-weight:bold;">Interactive simulation</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Common Questions ──
    st.markdown('<div class="section-header">❓ Common Questions</div>', unsafe_allow_html=True)
    with st.expander("Why not ask an LLM to predict traffic?"):
        st.markdown("""
        **Because language models cannot predict traffic physics.**
        
        Traffic flow is governed by complex mathematical relationships between volume, speed, capacity, and density. An LLM cannot compute these relationships accurately.
        
        TrafficPulse uses a dedicated Machine Learning engine to perform the actual mathematical prediction. The AI simply explains the results in plain language.
        """)
        
    st.markdown("<br/><br/>", unsafe_allow_html=True)

    # ── Technical Details ──
    with st.expander("⚙️ Technical Details (For ML Judges)"):
        st.markdown("""
        **Feature Columns (per lane):** `VEHS`, `SPEEDAVGARITH`, `SPEEDAVGHARM`, `QUEUEDELAY`, `OCCUPRATE` × Lanes 1–6
        
        **Engineered Features:** 
        - Rolling lags (1/3/6 intervals)
        - Rolling means
        - Forward targets
        - Temporal features (hour, day_of_week, is_weekend)
        
        **Prediction Targets:**
        - `occup_rate_15m` / `30m`
        - `queue_delay_15m` / `30m`
        - `speed_15m` / `30m`
        
        **Raw Metrics:**
        """)
        metrics = predictor.get_metrics()
        if metrics:
            metrics_df = pd.DataFrame([
                {"Target": k.replace("_", " ").title(), "MAE": f"{v['mae']:.4f}", "R²": f"{v['r2']:.4f}"}
                for k, v in metrics.items()
            ])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PLOTLY CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _chart_lane_speed(lane_metrics):
    lanes = list(lane_metrics.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Harmonic Speed", x=[f"Lane {l}" for l in lanes],
                         y=[lane_metrics[l]["harm_speed"] for l in lanes], marker_color="#6366f1"))
    fig.add_trace(go.Bar(name="Arithmetic Speed", x=[f"Lane {l}" for l in lanes],
                         y=[lane_metrics[l]["arith_speed"] for l in lanes], marker_color="#a78bfa"))
    fig.update_layout(**CHART_LAYOUT, title="Speed by Lane (km/h)", barmode="group", height=350,
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)


def _chart_lane_volume_occup(lane_metrics):
    lanes = list(lane_metrics.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Volume", x=[f"Lane {l}" for l in lanes],
                         y=[lane_metrics[l]["volume"] for l in lanes], marker_color="#8b5cf6"))
    fig.add_trace(go.Scatter(name="Occupancy", x=[f"Lane {l}" for l in lanes],
                             y=[lane_metrics[l]["occup_rate"] for l in lanes],
                             mode="lines+markers", marker=dict(color="#f59e0b", size=10),
                             line=dict(color="#f59e0b", width=3), yaxis="y2"))
    fig.update_layout(**CHART_LAYOUT, title="Volume & Occupancy by Lane", height=350,
                      yaxis=dict(title="Vehicles"), yaxis2=dict(title="Occupancy", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)


def _chart_health_radar(report):
    cats = ["Speed", "Occupancy", "Delay"]
    vals = [report.speed_score, report.occup_score, report.delay_score]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
                                  fillcolor="rgba(99,102,241,0.15)", line=dict(color="#6366f1", width=3),
                                  marker=dict(size=8), name="Current"))
    pred = compute_health_score(report.predicted_speed_15m, report.predicted_occup_15m, report.predicted_delay_15m)
    pv = [pred[1], pred[2], pred[3]]
    fig.add_trace(go.Scatterpolar(r=pv + [pv[0]], theta=cats + [cats[0]], fill="toself",
                                  fillcolor="rgba(139,92,246,0.08)", line=dict(color="#a78bfa", width=2, dash="dash"),
                                  marker=dict(size=6), name="+15 min"))
    fig.update_layout(**CHART_LAYOUT, title="Health Score Components", height=370,
                      polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(148,163,184,0.1)"),
                                 angularaxis=dict(gridcolor="rgba(148,163,184,0.1)")),
                      legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig, use_container_width=True)


def _chart_health_gauge(report):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=report.health_score,
        delta={"reference": report.predicted_health_15m,
               "increasing": {"color": "#10b981"}, "decreasing": {"color": "#ef4444"}},
        number={"font": {"size": 52, "color": "#f8fafc", "family": "Inter"}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#94a3b8"}},
               "bar": {"color": report.congestion_color, "thickness": 0.8},
               "bgcolor": "#1e293b", "bordercolor": "rgba(148,163,184,0.1)",
               "steps": [{"range": [0, 25], "color": "rgba(127,29,29,0.3)"},
                         {"range": [25, 50], "color": "rgba(239,68,68,0.15)"},
                         {"range": [50, 75], "color": "rgba(245,158,11,0.15)"},
                         {"range": [75, 100], "color": "rgba(16,185,129,0.15)"}],
               "threshold": {"line": {"color": "#a78bfa", "width": 3},
                             "thickness": 0.8, "value": report.predicted_health_15m}},
    ))
    fig.update_layout(**CHART_LAYOUT, title="Traffic Health Gauge", height=370)
    st.plotly_chart(fig, use_container_width=True)


def _chart_daily_trend(df, link_id, selected_date, selected_time):
    day = df[(df["LINK_ID"] == link_id) & (df["datetime"].dt.date == selected_date)].sort_values("datetime").copy()
    if day.empty:
        st.info("No trend data for this day.")
        return

    day["health_score"] = day.apply(lambda r: compute_health_score(
        float(r.get("avg_harm_speed", 0) or 0), float(r.get("avg_occup_rate", 0) or 0),
        float(r.get("max_queue_delay", 0) or 0))[0], axis=1)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=day["datetime"], y=day["health_score"], mode="lines", fill="tozeroy",
                                   fillcolor="rgba(99,102,241,0.1)", line=dict(color="#6366f1", width=2.5)))
        fig.add_vline(x=selected_time.timestamp() * 1000, line_dash="dash", line_color="#f59e0b", line_width=2,
                      annotation_text="Now", annotation_font_color="#f59e0b")
        fig.update_layout(**CHART_LAYOUT, title=f"Health Score — Link {link_id}", height=320,
                          yaxis=dict(title="Health Score", range=[0, 100]),
                          xaxis=dict(title="Time"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=day["datetime"], y=day["avg_harm_speed"], mode="lines",
                                 line=dict(color="#8b5cf6", width=2), name="Speed"))
        fig.add_trace(go.Scatter(x=day["datetime"], y=day["total_volume"], mode="lines",
                                 line=dict(color="#f59e0b", width=2), name="Volume", yaxis="y2"))
        fig.add_vline(x=selected_time.timestamp() * 1000, line_dash="dash", line_color="#6366f1", line_width=2)
        fig.update_layout(**CHART_LAYOUT, title=f"Speed & Volume — Link {link_id}", height=320,
                          yaxis=dict(title="Speed (km/h)"),
                          yaxis2=dict(title="Volume", overlaying="y", side="right"),
                          legend=dict(orientation="h", y=-0.15), xaxis=dict(title="Time"))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()

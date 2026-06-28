"""
Module 4: Decision Engine
==========================
Translates Traffic Intelligence outputs into stakeholder-specific advisories
using explicit, deterministic, rule-based logic.

This module NEVER calls ML models or generates predictions.
It operates purely on structured outputs from the Intelligence Engine.

Stakeholder Profiles:
  1. Daily Commuter — trip timing and lane observations
  2. Traffic Control Center — intervention alerts
  3. Emergency Services — lowest-delay link segment routing
  4. Logistics & Fleet — optimal dispatch window
  5. City Planner — evidence-based congestion patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from core.intelligence import (
    TrafficHealthReport,
    compute_health_score,
    classify_congestion,
    CONGESTION_LEVELS,
)


# ---------------------------------------------------------------------------
# 1. Daily Commuter Advisory
# ---------------------------------------------------------------------------
def daily_commuter_advisory(report: TrafficHealthReport) -> Dict:
    """
    Generate commuter-facing advisory based on current vs. predicted state.

    Rules:
      - If predicted health worsens by ≥10 points in 15m → recommend delaying
      - Identify lanes with observably higher speeds (not prescriptive)
      - Estimate expected delay based on queue delay metrics

    Returns a structured advisory dict (NOT natural language).
    """
    current_health = report.health_score
    predicted_health_15 = report.predicted_health_15m
    predicted_health_30 = report.predicted_health_30m

    health_delta_15 = predicted_health_15 - current_health
    health_delta_30 = predicted_health_30 - current_health

    # Determine trend
    if health_delta_15 <= -10:
        trend = "worsening"
    elif health_delta_15 >= 10:
        trend = "improving"
    else:
        trend = "stable"

    # Departure recommendation
    if trend == "worsening":
        # Health dropping → suggest leaving now or waiting for improvement
        if health_delta_30 > health_delta_15:
            recommendation = "delay_20min"
            rec_text = "Consider delaying departure by ~20 minutes. Conditions are predicted to worsen shortly but may recover."
        else:
            recommendation = "depart_now"
            rec_text = "Conditions are expected to worsen. Departing soon may be preferable."
    elif trend == "improving":
        recommendation = "delay_15min"
        rec_text = "Conditions are improving. Waiting ~15 minutes may result in a smoother commute."
    else:
        recommendation = "depart_anytime"
        rec_text = "Traffic conditions are stable. No significant timing advantage expected."

    # Lane observations (observational, not prescriptive)
    lane_observations = _observe_lanes(report.lane_metrics)

    return {
        "profile": "Daily Commuter",
        "congestion_level": report.congestion_level,
        "health_score": current_health,
        "trend": trend,
        "health_delta_15m": round(health_delta_15, 1),
        "health_delta_30m": round(health_delta_30, 1),
        "predicted_congestion_15m": report.predicted_congestion_15m,
        "predicted_congestion_30m": report.predicted_congestion_30m,
        "recommendation_code": recommendation,
        "recommendation_text": rec_text,
        "expected_delay_seconds": report.max_queue_delay,
        "lane_observations": lane_observations,
        "anomaly_count": len(report.anomalies),
    }


# ---------------------------------------------------------------------------
# 2. Traffic Control Center
# ---------------------------------------------------------------------------
def traffic_control_center(
    all_reports: List[TrafficHealthReport],
) -> Dict:
    """
    Identify links requiring immediate intervention.

    Rules:
      - Health Score < 40 → flag for intervention
      - Any active anomalies → flag for investigation
      - Rank by severity (lowest health first)

    Returns a structured intervention list.
    """
    intervention_links = []
    anomaly_links = []
    healthy_count = 0

    for r in all_reports:
        if r.health_score < 40:
            intervention_links.append({
                "link_id": r.link_id,
                "health_score": r.health_score,
                "congestion_level": r.congestion_level,
                "max_queue_delay": r.max_queue_delay,
                "avg_occup_rate": r.avg_occup_rate,
                "avg_harm_speed": r.avg_harm_speed,
                "predicted_health_15m": r.predicted_health_15m,
                "trend": "worsening" if r.predicted_health_15m < r.health_score else "stable_or_improving",
            })
        else:
            healthy_count += 1

        if r.anomalies:
            anomaly_links.append({
                "link_id": r.link_id,
                "anomalies": r.anomalies,
                "health_score": r.health_score,
            })

    # Sort by worst health first
    intervention_links.sort(key=lambda x: x["health_score"])

    network_health = np.mean([r.health_score for r in all_reports]) if all_reports else 0

    return {
        "profile": "Traffic Control Center",
        "network_health": round(network_health, 1),
        "total_links": len(all_reports),
        "links_requiring_intervention": intervention_links[:10],
        "intervention_count": len(intervention_links),
        "anomaly_alerts": anomaly_links[:5],
        "anomaly_count": len(anomaly_links),
        "healthy_link_count": healthy_count,
    }


# ---------------------------------------------------------------------------
# 3. Emergency Services Routing
# ---------------------------------------------------------------------------
def emergency_routing(
    all_reports: List[TrafficHealthReport],
    target_link_id: Optional[int] = None,
) -> Dict:
    """
    Identify road segments with the lowest predicted delay for emergency access.

    Instead of recommending specific lanes (we don't know physical constraints),
    we recommend the link segments with lowest occupancy and delay.
    """
    # Rank all links by lowest delay + lowest occupancy
    candidates = []
    for r in all_reports:
        candidates.append({
            "link_id": r.link_id,
            "health_score": r.health_score,
            "congestion_level": r.congestion_level,
            "avg_occup_rate": r.avg_occup_rate,
            "max_queue_delay": r.max_queue_delay,
            "predicted_delay_15m": r.predicted_delay_15m,
            "avg_harm_speed": r.avg_harm_speed,
        })

    # Sort by delay (primary) then occupancy (secondary)
    candidates.sort(key=lambda x: (x["max_queue_delay"], x["avg_occup_rate"]))

    # For the target link specifically, provide lane observations
    target_lane_obs = None
    if target_link_id:
        for r in all_reports:
            if r.link_id == target_link_id:
                target_lane_obs = _observe_lanes_emergency(r.lane_metrics)
                break

    return {
        "profile": "Emergency Services",
        "recommended_segments": candidates[:5],
        "avoid_segments": candidates[-3:] if len(candidates) >= 3 else [],
        "target_link_lane_observations": target_lane_obs,
    }


# ---------------------------------------------------------------------------
# 4. Logistics & Fleet Management
# ---------------------------------------------------------------------------
def logistics_advisor(
    df: pd.DataFrame,
    link_id: int,
    current_time_idx: int,
    max_future_intervals: int = 36,  # 3 hours
) -> Dict:
    """
    Scan future time windows to recommend optimal dispatch timing.

    Looks at the next 36 intervals (3 hours) to find the window with
    the best predicted health score.
    """
    link_data = df[df["LINK_ID"] == link_id].copy()
    if link_data.empty:
        return {
            "profile": "Logistics & Fleet",
            "status": "no_data",
            "recommendation": "Insufficient data for this link.",
        }

    link_data = link_data.sort_values("datetime").reset_index(drop=True)

    # Find current position
    current_mask = link_data.index == current_time_idx
    if not current_mask.any():
        # Fallback: use the closest index
        current_pos = min(current_time_idx, len(link_data) - 1)
    else:
        current_pos = link_data.index[current_mask][0]

    # Scan future windows
    future_windows = []
    end_pos = min(current_pos + max_future_intervals + 1, len(link_data))

    for i in range(current_pos, end_pos):
        row = link_data.iloc[i]
        speed = float(row.get("avg_harm_speed", 0) or 0)
        occup = float(row.get("avg_occup_rate", 0) or 0)
        delay = float(row.get("max_queue_delay", 0) or 0)
        health = compute_health_score(speed, occup, delay)[0]

        minutes_ahead = (i - current_pos) * 5
        future_windows.append({
            "minutes_ahead": minutes_ahead,
            "time_label": f"+{minutes_ahead} min",
            "health_score": health,
            "congestion_level": classify_congestion(health)[0],
            "avg_harm_speed": round(speed, 2),
            "avg_occup_rate": round(occup, 3),
            "max_queue_delay": round(delay, 2),
            "datetime": str(row.get("datetime", "")),
        })

    if not future_windows:
        return {
            "profile": "Logistics & Fleet",
            "status": "no_future_data",
            "recommendation": "No future time windows available for analysis.",
            "windows": [],
        }

    # Find optimal window
    best = max(future_windows, key=lambda w: w["health_score"])

    return {
        "profile": "Logistics & Fleet",
        "status": "ok",
        "current_health": future_windows[0]["health_score"] if future_windows else 0,
        "optimal_window": best,
        "recommendation": (
            f"Optimal dispatch window: {best['time_label']} "
            f"(Health Score: {best['health_score']}, "
            f"{best['congestion_level']})"
        ),
        "future_windows": future_windows,
    }


# ---------------------------------------------------------------------------
# 5. City Planner Evidence Report
# ---------------------------------------------------------------------------
def city_planner_report(
    df: pd.DataFrame,
    link_id: Optional[int] = None,
) -> Dict:
    """
    Generate evidence-based congestion analysis for urban planning.

    Presents:
      - Frequently congested links (health < 50 in >30% of observations)
      - Recurring peak congestion hours
      - Lane utilization imbalance

    No infrastructure recommendations — just data for planners to interpret.
    """
    # Compute health scores for all rows efficiently
    speeds = df["avg_harm_speed"].fillna(0).values
    occups = df["avg_occup_rate"].fillna(0).values
    delays = df["max_queue_delay"].fillna(0).values

    health_scores = []
    for s, o, d in zip(speeds, occups, delays):
        h = compute_health_score(float(s), float(o), float(d))[0]
        health_scores.append(h)
    df = df.copy()
    df["_health"] = health_scores

    # --- Frequently congested links ---
    link_congestion = df.groupby("LINK_ID").agg(
        mean_health=("_health", "mean"),
        pct_below_50=("_health", lambda x: (x < 50).mean() * 100),
        total_intervals=("_health", "count"),
    ).reset_index()

    chronic_links = link_congestion[link_congestion["pct_below_50"] > 30].sort_values(
        "pct_below_50", ascending=False
    )

    chronic_list = []
    for _, row in chronic_links.head(10).iterrows():
        chronic_list.append({
            "link_id": int(row["LINK_ID"]),
            "mean_health_score": round(row["mean_health"], 1),
            "pct_congested": round(row["pct_below_50"], 1),
        })

    # --- Peak congestion hours ---
    hourly = df.groupby("hour")["_health"].mean().reset_index()
    hourly.columns = ["hour", "avg_health"]
    peak_hours = hourly.nsmallest(3, "avg_health")
    peak_list = [
        {"hour": int(r["hour"]), "avg_health": round(r["avg_health"], 1)}
        for _, r in peak_hours.iterrows()
    ]

    # --- Lane utilization imbalance (for specific link) ---
    lane_imbalance = None
    if link_id:
        link_df = df[df["LINK_ID"] == link_id]
        if not link_df.empty:
            lane_volumes = {}
            lane_speeds = {}
            for lane in range(1, 7):
                vcol = f"VEHS(ALL)_{lane}"
                scol = f"SPEEDAVGHARM(ALL)_{lane}"
                lane_volumes[lane] = round(link_df[vcol].mean(), 1)
                lane_speeds[lane] = round(link_df[scol].mean(), 1)

            total_vol = sum(lane_volumes.values()) or 1
            lane_shares = {k: round(v / total_vol * 100, 1) for k, v in lane_volumes.items()}

            # Imbalance metric: coefficient of variation of lane shares
            shares = list(lane_shares.values())
            cv = (np.std(shares) / np.mean(shares) * 100) if np.mean(shares) > 0 else 0

            lane_imbalance = {
                "lane_volumes": lane_volumes,
                "lane_speeds": lane_speeds,
                "lane_share_pct": lane_shares,
                "imbalance_cv": round(cv, 1),
                "imbalance_assessment": (
                    "Significant imbalance" if cv > 50
                    else "Moderate imbalance" if cv > 25
                    else "Balanced utilization"
                ),
            }

    # --- Weekday vs Weekend comparison ---
    weekday = df[df["is_weekend"] == 0]["_health"].mean()
    weekend = df[df["is_weekend"] == 1]["_health"].mean()

    return {
        "profile": "City Planner",
        "chronically_congested_links": chronic_list,
        "peak_congestion_hours": peak_list,
        "lane_utilization": lane_imbalance,
        "weekday_avg_health": round(weekday, 1) if not np.isnan(weekday) else None,
        "weekend_avg_health": round(weekend, 1) if not np.isnan(weekend) else None,
    }


# ---------------------------------------------------------------------------
# Helper: Lane Observations (Observational, Not Prescriptive)
# ---------------------------------------------------------------------------
def _observe_lanes(lane_metrics: Dict) -> List[str]:
    """
    Generate observational statements about lane performance.
    Uses soft language per user directive — never says 'move to lane X'.
    """
    if not lane_metrics:
        return []

    observations = []

    # Find fastest and slowest lanes
    speeds = {lane: m["harm_speed"] for lane, m in lane_metrics.items() if m.get("harm_speed", 0) > 0}
    if not speeds:
        return ["Insufficient lane speed data for observations."]

    fastest_lane = max(speeds, key=speeds.get)
    slowest_lane = min(speeds, key=speeds.get)

    # Find least congested lane
    occups = {lane: m["occup_rate"] for lane, m in lane_metrics.items()}
    least_congested = min(occups, key=occups.get)

    if speeds[fastest_lane] > speeds.get(slowest_lane, 0) * 1.3:
        observations.append(
            f"Lane {fastest_lane} currently shows the highest average speed "
            f"({speeds[fastest_lane]:.1f} km/h)."
        )

    if len(speeds) > 1:
        high_speed_lanes = [l for l, s in speeds.items() if s >= speeds[fastest_lane] * 0.9]
        if len(high_speed_lanes) > 1:
            lane_str = ", ".join(str(l) for l in sorted(high_speed_lanes))
            observations.append(
                f"Lanes {lane_str} currently show higher average speeds and lower congestion."
            )

    observations.append(
        f"Lane {least_congested} has the lowest occupancy rate "
        f"({occups[least_congested]:.2f})."
    )

    return observations


def _observe_lanes_emergency(lane_metrics: Dict) -> List[str]:
    """
    Generate lane observations specifically relevant for emergency services.
    Observational only — the dataset cannot confirm physical access constraints.
    """
    if not lane_metrics:
        return ["No lane-level data available."]

    observations = []

    # Rank by combined low occupancy + low delay
    lane_scores = {}
    for lane, m in lane_metrics.items():
        occup = m.get("occup_rate", 999)
        delay = m.get("queue_delay", 999)
        lane_scores[lane] = occup + delay / 100  # Combined score

    sorted_lanes = sorted(lane_scores, key=lane_scores.get)

    best = sorted_lanes[0]
    observations.append(
        f"Lane {best} currently shows the lowest combined occupancy "
        f"({lane_metrics[best].get('occup_rate', 0):.2f}) and delay "
        f"({lane_metrics[best].get('queue_delay', 0):.1f}s) on this segment."
    )

    if len(sorted_lanes) >= 2:
        second = sorted_lanes[1]
        observations.append(
            f"Lane {second} is the next least congested alternative."
        )

    return observations


# ---------------------------------------------------------------------------
# CITY-LEVEL ADVISORIES (for Stakeholder Assistant city-first UX)
# ---------------------------------------------------------------------------

def city_commuter_advisory(all_reports: List[TrafficHealthReport]) -> Dict:
    """
    City-wide commuter advisory: which roads to avoid, which are clear,
    overall network health, and departure timing guidance.
    """
    if not all_reports:
        return {"profile": "Daily Commuter", "status": "no_data"}

    sorted_reports = sorted(all_reports, key=lambda r: r.health_score)
    avg_health = np.mean([r.health_score for r in all_reports])
    city_level = classify_congestion(avg_health)[0]

    # Roads to avoid (health < 60 or worst 5)
    avoid = [r for r in sorted_reports if r.health_score < 60][:5]
    avoid_list = [{"link_id": r.link_id, "health_score": r.health_score,
                   "congestion_level": r.congestion_level,
                   "delay": round(r.max_queue_delay, 0),
                   "speed": round(r.avg_harm_speed, 0)} for r in avoid]

    # Recommended roads (best health)
    best = sorted_reports[-5:][::-1]
    recommend_list = [{"link_id": r.link_id, "health_score": r.health_score,
                       "congestion_level": r.congestion_level,
                       "speed": round(r.avg_harm_speed, 0)} for r in best]

    # Network trend
    worsening_count = sum(1 for r in all_reports if r.predicted_health_15m < r.health_score - 5)
    improving_count = sum(1 for r in all_reports if r.predicted_health_15m > r.health_score + 5)

    if worsening_count > len(all_reports) * 0.3:
        network_trend = "worsening"
        timing_advice = "Network congestion is building. Consider departing soon or delaying by 30+ minutes."
    elif improving_count > len(all_reports) * 0.3:
        network_trend = "improving"
        timing_advice = "Conditions are improving across the network. Waiting 15–20 minutes may give a smoother commute."
    else:
        network_trend = "stable"
        timing_advice = "Network conditions are stable. No significant timing advantage expected."

    return {
        "profile": "Daily Commuter",
        "city_health": round(avg_health, 1),
        "city_status": city_level,
        "roads_to_avoid": avoid_list,
        "recommended_roads": recommend_list,
        "network_trend": network_trend,
        "timing_advice": timing_advice,
        "worsening_count": worsening_count,
        "improving_count": improving_count,
        "total_links": len(all_reports),
    }


def city_traffic_police(all_reports: List[TrafficHealthReport]) -> Dict:
    """
    City-wide traffic police command: intervention priorities,
    anomaly dashboard, deployment recommendations.
    """
    if not all_reports:
        return {"profile": "Traffic Police", "status": "no_data"}

    avg_health = np.mean([r.health_score for r in all_reports])

    # Intervention list: health < 40
    intervention = sorted(
        [r for r in all_reports if r.health_score < 40],
        key=lambda r: r.health_score
    )
    intervention_list = [{
        "link_id": r.link_id, "health_score": r.health_score,
        "congestion_level": r.congestion_level,
        "delay": round(r.max_queue_delay, 0),
        "speed": round(r.avg_harm_speed, 0),
        "occupancy": round(r.avg_occup_rate, 2),
        "trend": "worsening" if r.predicted_health_15m < r.health_score - 5 else "stable_or_improving",
    } for r in intervention]

    # Watch list: health 40-60
    watch = sorted(
        [r for r in all_reports if 40 <= r.health_score < 60],
        key=lambda r: r.health_score
    )
    watch_list = [{
        "link_id": r.link_id, "health_score": r.health_score,
        "congestion_level": r.congestion_level,
    } for r in watch[:5]]

    # Anomalies across entire network
    all_anomalies = []
    for r in all_reports:
        for a in r.anomalies:
            all_anomalies.append({"link_id": r.link_id, **a})

    return {
        "profile": "Traffic Police",
        "network_health": round(avg_health, 1),
        "total_links": len(all_reports),
        "intervention_required": intervention_list,
        "intervention_count": len(intervention_list),
        "watch_list": watch_list,
        "watch_count": len(watch),
        "anomaly_alerts": all_anomalies[:10],
        "anomaly_count": len(all_anomalies),
        "healthy_count": sum(1 for r in all_reports if r.health_score >= 60),
    }


def city_logistics_advisor(all_reports: List[TrafficHealthReport]) -> Dict:
    """
    City-wide logistics advisory: best corridors for dispatch now,
    overall network congestion assessment.
    """
    if not all_reports:
        return {"profile": "Logistics & Fleet", "status": "no_data"}

    sorted_reports = sorted(all_reports, key=lambda r: r.health_score, reverse=True)
    avg_health = np.mean([r.health_score for r in all_reports])
    avg_delay = np.mean([r.max_queue_delay for r in all_reports])

    best_corridors = [{
        "link_id": r.link_id, "health_score": r.health_score,
        "speed": round(r.avg_harm_speed, 0),
        "delay": round(r.max_queue_delay, 0),
    } for r in sorted_reports[:5]]

    avoid_corridors = [{
        "link_id": r.link_id, "health_score": r.health_score,
        "delay": round(r.max_queue_delay, 0),
    } for r in sorted_reports[-5:][::-1]]

    return {
        "profile": "Logistics & Fleet",
        "network_health": round(avg_health, 1),
        "avg_delay": round(avg_delay, 0),
        "best_corridors": best_corridors,
        "avoid_corridors": avoid_corridors,
        "total_links": len(all_reports),
    }

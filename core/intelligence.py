"""
Module 3: Traffic Intelligence Engine
=======================================
The central analytical layer that computes the Traffic Health Score —
the core metric around which the entire system revolves.

Responsibilities:
  - Compute Traffic Health Score (0–100) per link using weighted metrics
  - Classify congestion levels (Free Flow / Moderate / Heavy / Gridlock)
  - Rank links globally to identify worst-performing segments
  - Extract structured lane matrices for comparison views

This module consumes aggregated data and ML predictions but does NOT
generate recommendations — that's the Decision Engine's job.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Congestion Classification Thresholds
# ---------------------------------------------------------------------------
CONGESTION_LEVELS = {
    "Free Flow":  (75, 100),
    "Moderate":   (50, 74),
    "Heavy":      (25, 49),
    "Gridlock":   (0, 24),
}

CONGESTION_COLORS = {
    "Free Flow":  "#10b981",   # Emerald green
    "Moderate":   "#f59e0b",   # Amber
    "Heavy":      "#ef4444",   # Red
    "Gridlock":   "#7f1d1d",   # Dark red
}

# Health Score weights
WEIGHT_SPEED = 0.40
WEIGHT_OCCUP = 0.35
WEIGHT_DELAY = 0.25


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class TrafficHealthReport:
    """Structured output from the intelligence engine for a single link/time."""
    link_id: int
    health_score: float
    congestion_level: str
    congestion_color: str

    # Component scores (0–100 each)
    speed_score: float
    occup_score: float
    delay_score: float

    # Raw metrics
    avg_harm_speed: float
    avg_occup_rate: float
    max_queue_delay: float
    total_volume: int

    # Predictions (from ML engine)
    predicted_occup_15m: float = 0.0
    predicted_occup_30m: float = 0.0
    predicted_delay_15m: float = 0.0
    predicted_delay_30m: float = 0.0
    predicted_speed_15m: float = 0.0
    predicted_speed_30m: float = 0.0

    # Predicted health scores
    predicted_health_15m: float = 0.0
    predicted_health_30m: float = 0.0
    predicted_congestion_15m: str = ""
    predicted_congestion_30m: str = ""

    # Anomalies
    anomalies: List[Dict] = field(default_factory=list)

    # Lane matrix
    lane_metrics: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Health Score Computation
# ---------------------------------------------------------------------------
def compute_health_score(
    avg_harm_speed: float,
    avg_occup_rate: float,
    max_queue_delay: float,
    free_flow_speed: float = 120.0,
    max_delay_ref: float = 500.0,
) -> Tuple[float, float, float, float]:
    """
    Compute the Traffic Health Score on a 0–100 scale.

    Components:
      - Speed Score (40%): How close to free-flow speed
      - Occupancy Score (35%): Inverse of congestion density
      - Delay Score (25%): Inverse of queue delay

    Args:
        avg_harm_speed: Average harmonic speed across lanes (km/h)
        avg_occup_rate: Average occupancy rate across lanes (0–100 scale typical)
        max_queue_delay: Maximum queue delay across lanes (seconds)
        free_flow_speed: Reference free-flow speed for normalization
        max_delay_ref: Reference max delay for normalization

    Returns:
        Tuple of (health_score, speed_score, occup_score, delay_score)
    """
    # Speed Score: higher speed → higher score
    speed_score = min(100.0, max(0.0, (avg_harm_speed / free_flow_speed) * 100))

    # Occupancy Score: lower occupancy → higher score
    # Occupancy typically 0–~10+ in this dataset, normalize assuming max ~15
    occup_normalized = min(avg_occup_rate / 10.0, 1.0) if avg_occup_rate > 0 else 0.0
    occup_score = max(0.0, (1.0 - occup_normalized) * 100)

    # Delay Score: lower delay → higher score
    delay_normalized = min(max_queue_delay / max_delay_ref, 1.0) if max_queue_delay > 0 else 0.0
    delay_score = max(0.0, (1.0 - delay_normalized) * 100)

    # Weighted composite
    health_score = (
        WEIGHT_SPEED * speed_score +
        WEIGHT_OCCUP * occup_score +
        WEIGHT_DELAY * delay_score
    )

    return (
        round(health_score, 1),
        round(speed_score, 1),
        round(occup_score, 1),
        round(delay_score, 1),
    )


def classify_congestion(health_score: float) -> Tuple[str, str]:
    """
    Map a health score to a congestion level and its display color.

    Returns:
        Tuple of (congestion_level, color_hex)
    """
    for level, (low, high) in CONGESTION_LEVELS.items():
        if low <= health_score <= high:
            return level, CONGESTION_COLORS[level]
    return "Gridlock", CONGESTION_COLORS["Gridlock"]


# ---------------------------------------------------------------------------
# Link Ranking
# ---------------------------------------------------------------------------
def rank_links(
    df: pd.DataFrame,
    time_filter: Optional[str] = None,
    datetime_filter: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Rank all links by health score for a given time point.

    Returns a DataFrame with columns:
      LINK_ID, health_score, congestion_level, avg_harm_speed,
      avg_occup_rate, max_queue_delay, total_volume
    sorted ascending (worst first).
    """
    if datetime_filter is not None:
        snapshot = df[df["datetime"] == datetime_filter].copy()
    elif time_filter is not None:
        snapshot = df[df["TIMEINT"] == time_filter].copy()
    else:
        snapshot = df.copy()

    if snapshot.empty:
        return pd.DataFrame()

    results = []
    for _, row in snapshot.iterrows():
        speed = float(row.get("avg_harm_speed", 0) or 0)
        occup = float(row.get("avg_occup_rate", 0) or 0)
        delay = float(row.get("max_queue_delay", 0) or 0)

        health, s_score, o_score, d_score = compute_health_score(speed, occup, delay)
        level, color = classify_congestion(health)

        results.append({
            "LINK_ID": int(row["LINK_ID"]),
            "health_score": health,
            "congestion_level": level,
            "congestion_color": color,
            "speed_score": s_score,
            "occup_score": o_score,
            "delay_score": d_score,
            "avg_harm_speed": round(speed, 2),
            "avg_occup_rate": round(occup, 3),
            "max_queue_delay": round(delay, 2),
            "total_volume": int(row.get("total_volume", 0)),
        })

    ranked = pd.DataFrame(results)
    ranked.sort_values("health_score", ascending=True, inplace=True)
    ranked.reset_index(drop=True, inplace=True)
    return ranked


def get_top_worst_links(
    df: pd.DataFrame,
    datetime_filter: Optional[pd.Timestamp] = None,
    n: int = 5,
) -> pd.DataFrame:
    """Return the top-N worst-performing links at a given time."""
    ranked = rank_links(df, datetime_filter=datetime_filter)
    return ranked.head(n)


# ---------------------------------------------------------------------------
# Full Health Report for a single link/time
# ---------------------------------------------------------------------------
def build_health_report(
    row: pd.Series,
    predictions: Dict[str, float],
    anomalies: List[Dict],
    lane_metrics: Dict,
) -> TrafficHealthReport:
    """
    Build a complete TrafficHealthReport combining current state,
    ML predictions, anomalies, and lane metrics.

    This is the central structured output that feeds into the Decision Engine.
    """
    speed = float(row.get("avg_harm_speed", 0) or 0)
    occup = float(row.get("avg_occup_rate", 0) or 0)
    delay = float(row.get("max_queue_delay", 0) or 0)
    volume = int(row.get("total_volume", 0) or 0)

    # Current health
    health, s_score, o_score, d_score = compute_health_score(speed, occup, delay)
    level, color = classify_congestion(health)

    # Predicted health scores
    pred_speed_15 = predictions.get("speed_15m", speed)
    pred_occup_15 = predictions.get("occup_rate_15m", occup)
    pred_delay_15 = predictions.get("queue_delay_15m", delay)
    pred_health_15 = compute_health_score(pred_speed_15, pred_occup_15, pred_delay_15)[0]
    pred_level_15 = classify_congestion(pred_health_15)[0]

    pred_speed_30 = predictions.get("speed_30m", speed)
    pred_occup_30 = predictions.get("occup_rate_30m", occup)
    pred_delay_30 = predictions.get("queue_delay_30m", delay)
    pred_health_30 = compute_health_score(pred_speed_30, pred_occup_30, pred_delay_30)[0]
    pred_level_30 = classify_congestion(pred_health_30)[0]

    return TrafficHealthReport(
        link_id=int(row.get("LINK_ID", 0)),
        health_score=health,
        congestion_level=level,
        congestion_color=color,
        speed_score=s_score,
        occup_score=o_score,
        delay_score=d_score,
        avg_harm_speed=round(speed, 2),
        avg_occup_rate=round(occup, 3),
        max_queue_delay=round(delay, 2),
        total_volume=volume,
        predicted_occup_15m=round(pred_occup_15, 3),
        predicted_occup_30m=round(pred_occup_30, 3),
        predicted_delay_15m=round(pred_delay_15, 2),
        predicted_delay_30m=round(pred_delay_30, 2),
        predicted_speed_15m=round(pred_speed_15, 2),
        predicted_speed_30m=round(pred_speed_30, 2),
        predicted_health_15m=pred_health_15,
        predicted_health_30m=pred_health_30,
        predicted_congestion_15m=pred_level_15,
        predicted_congestion_30m=pred_level_30,
        anomalies=anomalies,
        lane_metrics=lane_metrics,
    )

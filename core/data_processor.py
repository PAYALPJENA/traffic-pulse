"""
Module 1: Data Processing & Feature Engineering
================================================
Loads the raw traffic dataset, parses temporal components, aggregates
lane-level metrics into link-level summaries, and engineers rolling
lag features for ML consumption.

Dataset-agnostic: works with any CSV following the lane-schema convention
(VEHS, SPEEDAVGARITH, SPEEDAVGHARM, QUEUEDELAY, OCCUPRATE) × lanes 1–6.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LANE_RANGE = range(1, 7)  # Lanes 1 through 6
METRICS = ["VEHS(ALL)", "SPEEDAVGARITH(ALL)", "SPEEDAVGHARM(ALL)",
           "QUEUEDELAY(ALL)", "OCCUPRATE(ALL)"]

# Rolling window sizes (in number of 5-min intervals)
LAG_15MIN = 3   # 3 × 5 min = 15 min
LAG_30MIN = 6   # 6 × 5 min = 30 min


# ---------------------------------------------------------------------------
# 1. Data Loading & Temporal Parsing
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """
    Load the traffic CSV and parse all temporal components.

    Extracts:
      - datetime from the 'date' column
      - hour, minute, day_of_week, is_weekend
      - time_slot_index: ordinal position of each 5-min interval within a day (0–287)

    Returns a DataFrame sorted by (LINK_ID, datetime) for correct rolling operations.
    """
    df = pd.read_csv(path)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["day_of_week"] = df["datetime"].dt.dayofweek          # 0=Mon … 6=Sun
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Time slot index: ordinal position within a day (0–287 for 5-min intervals)
    df["time_slot_index"] = df["hour"] * 12 + df["minute"] // 5

    # Ensure deterministic ordering for rolling calculations
    df.sort_values(["LINK_ID", "datetime"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ---------------------------------------------------------------------------
# 2. Lane Aggregation → Link-Level Summaries
# ---------------------------------------------------------------------------
def aggregate_lanes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute link-level summary metrics from the 6 per-lane columns.

    Creates:
      - total_volume:       sum of VEHS across all 6 lanes
      - avg_arith_speed:    mean of SPEEDAVGARITH across lanes
      - avg_harm_speed:     weighted harmonic mean of SPEEDAVGHARM (weighted by VEHS)
      - max_queue_delay:    maximum QUEUEDELAY across lanes
      - avg_queue_delay:    mean QUEUEDELAY across lanes
      - avg_occup_rate:     mean OCCUPRATE across lanes
      - max_occup_rate:     max OCCUPRATE across lanes
    """
    # Gather lane columns into arrays for vectorized ops
    vehs_cols = [f"VEHS(ALL)_{i}" for i in LANE_RANGE]
    arith_cols = [f"SPEEDAVGARITH(ALL)_{i}" for i in LANE_RANGE]
    harm_cols = [f"SPEEDAVGHARM(ALL)_{i}" for i in LANE_RANGE]
    delay_cols = [f"QUEUEDELAY(ALL)_{i}" for i in LANE_RANGE]
    occup_cols = [f"OCCUPRATE(ALL)_{i}" for i in LANE_RANGE]

    # Total volume
    df["total_volume"] = df[vehs_cols].sum(axis=1)

    # Arithmetic speed average
    df["avg_arith_speed"] = df[arith_cols].mean(axis=1)

    # Weighted harmonic mean speed (weighted by vehicle count per lane)
    vehs_arr = df[vehs_cols].values.astype(float)
    harm_arr = df[harm_cols].values.astype(float)

    # Avoid division by zero: replace 0 speeds with NaN
    harm_safe = np.where(harm_arr > 0, harm_arr, np.nan)
    weighted_inv = vehs_arr / harm_safe                 # weight / speed
    total_weight = np.nansum(vehs_arr, axis=1)
    sum_weighted_inv = np.nansum(weighted_inv, axis=1)

    # Harmonic mean = total_weight / sum(weight/speed)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["avg_harm_speed"] = np.where(
            sum_weighted_inv > 0,
            total_weight / sum_weighted_inv,
            0.0
        )

    # Queue delay aggregates
    df["max_queue_delay"] = df[delay_cols].max(axis=1)
    df["avg_queue_delay"] = df[delay_cols].mean(axis=1)

    # Occupancy rate aggregates
    df["avg_occup_rate"] = df[occup_cols].mean(axis=1)
    df["max_occup_rate"] = df[occup_cols].max(axis=1)

    return df


# ---------------------------------------------------------------------------
# 3. Feature Engineering (Rolling Lag Windows)
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling lag features per LINK_ID for temporal forecasting.

    For each of [avg_occup_rate, max_queue_delay, avg_harm_speed, total_volume]:
      - lag_1:  value at t-1  (5 min ago)
      - lag_3:  value at t-3  (15 min ago)
      - lag_6:  value at t-6  (30 min ago)
      - roll_3_mean: rolling mean over last 3 intervals (15 min)
      - roll_6_mean: rolling mean over last 6 intervals (30 min)

    Also creates future targets for supervised learning:
      - *_fwd_3: value at t+3  (15 min ahead)
      - *_fwd_6: value at t+6  (30 min ahead)
    """
    target_cols = ["avg_occup_rate", "max_queue_delay", "avg_harm_speed", "total_volume"]

    feature_dfs = []
    for link_id, group in df.groupby("LINK_ID"):
        g = group.copy()
        for col in target_cols:
            # Lag features (past)
            g[f"{col}_lag_1"] = g[col].shift(1)
            g[f"{col}_lag_3"] = g[col].shift(LAG_15MIN)
            g[f"{col}_lag_6"] = g[col].shift(LAG_30MIN)

            # Rolling means (past)
            g[f"{col}_roll_3_mean"] = g[col].rolling(window=LAG_15MIN, min_periods=1).mean()
            g[f"{col}_roll_6_mean"] = g[col].rolling(window=LAG_30MIN, min_periods=1).mean()

            # Future targets (for ML training)
            g[f"{col}_fwd_3"] = g[col].shift(-LAG_15MIN)
            g[f"{col}_fwd_6"] = g[col].shift(-LAG_30MIN)

        feature_dfs.append(g)

    result = pd.concat(feature_dfs, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# 4. Full Pipeline
# ---------------------------------------------------------------------------
def process_pipeline(path: str) -> pd.DataFrame:
    """
    Execute the full data processing pipeline:
      load → aggregate lanes → engineer features.

    Returns the fully enriched DataFrame ready for ML and intelligence layers.
    """
    df = load_data(path)
    df = aggregate_lanes(df)
    df = engineer_features(df)
    return df


# ---------------------------------------------------------------------------
# 5. Utility: Extract per-lane metrics for a single row
# ---------------------------------------------------------------------------
def get_lane_metrics(row: pd.Series) -> dict:
    """
    Extract a structured lane-by-lane metrics dict from a single DataFrame row.

    Returns:
      {
        1: {volume, arith_speed, harm_speed, queue_delay, occup_rate},
        2: {...},
        ...
        6: {...}
      }
    """
    lanes = {}
    for i in LANE_RANGE:
        lanes[i] = {
            "volume": float(row.get(f"VEHS(ALL)_{i}", 0)),
            "arith_speed": float(row.get(f"SPEEDAVGARITH(ALL)_{i}", 0)),
            "harm_speed": float(row.get(f"SPEEDAVGHARM(ALL)_{i}", 0)),
            "queue_delay": float(row.get(f"QUEUEDELAY(ALL)_{i}", 0)),
            "occup_rate": float(row.get(f"OCCUPRATE(ALL)_{i}", 0)),
        }
    return lanes

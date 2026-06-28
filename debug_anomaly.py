import pandas as pd
import numpy as np
from core.data_processor import process_pipeline
from core.ml_engine import AnomalyDetector

df = process_pipeline("Pangyo_14days_lanes_w_arith_adj.csv")

det = AnomalyDetector(z_threshold=2.0)
det.fit(df)

# Check stats for link 19
stats = det._link_stats.get(19, {})
for lane in [1, 2, 3]:
    s = stats.get(lane, {})
    print(f"Lane {lane}: speed_mean={s.get('speed_mean',0):.1f}, speed_std={s.get('speed_std',0):.1f}, occup_mean={s.get('occup_mean',0):.4f}, occup_std={s.get('occup_std',0):.4f}")

# Count anomalies in 500 random rows
count = 0
for _, row in df.sample(500, random_state=42).iterrows():
    anoms = det.detect(row)
    count += len(anoms)
print(f"\nAnomalies in 500 random rows: {count}")

# Check worst-case z-scores for link 19
link19 = df[df["LINK_ID"] == 19]
print(f"\nLink 19 rows: {len(link19)}")
for lane in [1, 2, 3]:
    speed_col = f"SPEEDAVGHARM(ALL)_{lane}"
    occup_col = f"OCCUPRATE(ALL)_{lane}"
    speed_std = link19[speed_col].std()
    occup_std = link19[occup_col].std()
    speed_min = link19[speed_col].min()
    speed_mean = link19[speed_col].mean()
    occup_max = link19[occup_col].max()
    occup_mean = link19[occup_col].mean()

    worst_speed_z = (speed_mean - speed_min) / (speed_std or 1)
    worst_occup_z = (occup_max - occup_mean) / (occup_std or 1)
    print(f"  Lane {lane}: worst_speed_z={worst_speed_z:.2f}, worst_occup_z={worst_occup_z:.2f}")
    print(f"    speed: min={speed_min:.1f} mean={speed_mean:.1f} std={speed_std:.1f}")
    print(f"    occup: max={occup_max:.4f} mean={occup_mean:.4f} std={occup_std:.4f}")

# Check ALL rows for all links
print("\nChecking ALL rows for anomalies (this may take a minute)...")
total = 0
for _, row in df.iterrows():
    total += len(det.detect(row))
print(f"Total anomalies across entire dataset: {total}")

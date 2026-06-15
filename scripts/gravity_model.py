"""
Option C - Gravity Model
SCHM 6201 | Supply Chain Vulnerability - ClearPath Analytics

Scores all 17,822 directed CFS-area corridors using a spatial gravity model:

    gravity_ij = (VAL_i * VAL_j) / distance_ij^2

where VAL is annual CFS shipment value ($K) for the origin (i) and destination
(j) areas, and distance is the great-circle centroid distance in miles.

High-gravity corridors carry the most value over the shortest distance, so a
disruption there propagates the largest economic impact across the network.
This ranking feeds the Network Graph + Rerouting workstreams.

Inputs  (supply_chain_data/05_features/):
    distance_matrix.csv  - 17,822 directed (origin, dest) pairs + distance_miles
    features_master.csv  - per-area VAL, TON, vulnerability_score, etc.

Outputs (supply_chain_data/03_outputs/):
    corridor_gravity_scores.csv  - full ranked edge list (17,822 rows)
    top_corridors_by_gravity.png - top-20 corridor bar chart
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1] / "supply_chain_data"
FEATURES = ROOT / "05_features"
OUTPUTS = ROOT / "03_outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- load
dist = pd.read_csv(FEATURES / "distance_matrix.csv")
feat = pd.read_csv(FEATURES / "features_master.csv")

val = feat.set_index("GEO_ID")["VAL"]

# ---------------------------------------------------------------- join VAL onto both endpoints
df = dist.copy()
df["VAL_origin"] = df["GEO_ID_origin"].map(val)
df["VAL_dest"] = df["GEO_ID_dest"].map(val)

missing = df[["VAL_origin", "VAL_dest"]].isna().any(axis=1).sum()
if missing:
    raise ValueError(f"{missing} corridors have an endpoint with no VAL in features_master")

# ---------------------------------------------------------------- distance floor
# 118 corridors connect co-located zones (metro "parts" / "Remainder of" areas
# that share a centroid) and have distance_miles == 0, which would make gravity
# infinite. Floor distance at the smallest observed inter-zone distance so these
# are treated as the closest-possible corridor, not infinitely close. Flag them
# so the Network Graph / Rerouting steps can treat them explicitly.
DIST_FLOOR = df.loc[df["distance_miles"] > 0, "distance_miles"].min()
df["dist_floored"] = df["distance_miles"] < DIST_FLOOR
df["distance_used"] = df["distance_miles"].clip(lower=DIST_FLOOR)

# ---------------------------------------------------------------- gravity
df["gravity"] = (df["VAL_origin"] * df["VAL_dest"]) / (df["distance_used"] ** 2)

# 0-1 normalized score for downstream edge weighting
g = df["gravity"]
df["gravity_norm"] = (g - g.min()) / (g.max() - g.min())

# rank (1 = highest disruption impact)
df = df.sort_values("gravity", ascending=False).reset_index(drop=True)
df["rank"] = np.arange(1, len(df) + 1)

# ---------------------------------------------------------------- write edge list
cols = [
    "rank", "GEO_ID_origin", "NAME_origin", "GEO_ID_dest", "NAME_dest",
    "VAL_origin", "VAL_dest", "distance_miles", "distance_used", "dist_floored",
    "gravity", "gravity_norm",
]
out = df[cols]
out_path = OUTPUTS / "corridor_gravity_scores.csv"
out.to_csv(out_path, index=False)

# ---------------------------------------------------------------- summary
print("=" * 70)
print("OPTION C - GRAVITY MODEL")
print("=" * 70)
print(f"Corridors scored      : {len(df):,}")
print(f"Distance floor applied : {DIST_FLOOR:.2f} mi  ({int(df['dist_floored'].sum())} co-located corridors)")
print(f"Gravity range          : {g.min():,.0f}  ->  {g.max():,.0f}")
print(f"Output                 : {out_path.relative_to(ROOT.parent)}")
print()
print("Top 15 highest-disruption-impact corridors:")
print("-" * 70)
short = lambda s: s.split(" CFS Area")[0].split(",")[0]
for _, r in df.head(15).iterrows():
    print(f"  {r['rank']:>2}. {short(r.NAME_origin):<28} -> {short(r.NAME_dest):<28} "
          f"| {r.distance_miles:>7.1f} mi | g={r.gravity:,.0f}")

# ---------------------------------------------------------------- plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = df.head(20).iloc[::-1]
    labels = [f"{short(o)} -> {short(d)}"
              for o, d in zip(top.NAME_origin, top.NAME_dest)]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(labels, top["gravity"], color="#0d9488")
    ax.set_xlabel("Gravity score  (VAL_i x VAL_j / distance$^2$)")
    ax.set_title("Top 20 Disruption-Impact Corridors - Gravity Model")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    fig.tight_layout()
    fig_path = OUTPUTS / "top_corridors_by_gravity.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nPlot saved             : {fig_path.relative_to(ROOT.parent)}")
except ImportError:
    print("\n(matplotlib not available - skipped plot)")

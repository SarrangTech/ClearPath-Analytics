"""
Rebuild centroids.csv + distance_matrix.csv from authoritative source.

The original centroids.csv was geocoded by area *name*, which collapsed 66 of
134 CFS areas onto a shared (and often wrong) coordinate -- e.g. Buffalo,
Rochester and Albany all landed on New York City's lat/lon, and Baltimore, MD
landed in Columbia, SC. That corrupted ~74% of corridor distances feeding the
gravity model.

The 2022_CFS_Areas shapefile ships the Census-provided internal point
(INTPTLAT/INTPTLON) for every area -- a guaranteed-inside, authoritative
coordinate. This script reads those directly, rebuilds the centroid table, and
recomputes the full directed distance matrix with the haversine formula.

NAME values are preserved from the original centroids.csv (keyed on GEO_ID) so
downstream joins in the Network Graph / Rerouting workstreams are unchanged --
only the coordinates and distances are corrected.

Originals are backed up to *.orig.csv before being overwritten.
"""

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import shapefile  # pyshp

ROOT = Path(__file__).resolve().parents[1] / "supply_chain_data"
FEATURES = ROOT / "05_features"
SHP = ROOT / "02_shapefiles" / "cfs_areas" / "2022_CFS_Areas.shp"

# ---------------------------------------------------------------- read authoritative internal points
reader = shapefile.Reader(str(SHP))
recs = [r.as_dict() for r in reader.records()]
sf = pd.DataFrame({
    "GEO_ID": [r["CFS22_GE_1"] for r in recs],
    "INTPTLAT": [float(r["INTPTLAT"]) for r in recs],
    "INTPTLON": [float(r["INTPTLON"]) for r in recs],
})

n_unique = sf.groupby(["INTPTLAT", "INTPTLON"]).ngroups
assert n_unique == len(sf), f"expected distinct coords, got {n_unique}/{len(sf)}"

# preserve original NAME mapping (cosmetic; joins key on GEO_ID)
orig = pd.read_csv(FEATURES / "centroids.csv")
sf = sf.merge(orig[["GEO_ID", "NAME"]], on="GEO_ID", how="left")
assert sf["NAME"].notna().all(), "some GEO_IDs missing from original centroids.csv"

cent = sf[["GEO_ID", "NAME", "INTPTLAT", "INTPTLON"]].sort_values("NAME").reset_index(drop=True)

# ---------------------------------------------------------------- haversine distance matrix (all directed pairs)
R_MILES = 3958.7613

lat = np.radians(cent["INTPTLAT"].to_numpy())
lon = np.radians(cent["INTPTLON"].to_numpy())

# pairwise via broadcasting
dlat = lat[:, None] - lat[None, :]
dlon = lon[:, None] - lon[None, :]
a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
dist_mi = 2 * R_MILES * np.arcsin(np.sqrt(a))

geo = cent["GEO_ID"].to_numpy()
nm = cent["NAME"].to_numpy()
n = len(cent)

oi, di = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
mask = oi != di  # drop self-pairs
dm = pd.DataFrame({
    "GEO_ID_origin": geo[oi[mask]],
    "NAME_origin": nm[oi[mask]],
    "GEO_ID_dest": geo[di[mask]],
    "NAME_dest": nm[di[mask]],
    "distance_miles": np.round(dist_mi[mask], 2),
})

# ---------------------------------------------------------------- back up + write
for name, frame in [("centroids.csv", cent), ("distance_matrix.csv", dm)]:
    path = FEATURES / name
    backup = FEATURES / name.replace(".csv", ".orig.csv")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    frame.to_csv(path, index=False)

zeros = int((dm["distance_miles"] == 0).sum())
print(f"centroids rebuilt      : {len(cent)} areas, {n_unique} distinct coords")
print(f"distance matrix rebuilt: {len(dm):,} directed corridors")
print(f"zero-distance corridors: {zeros}  (was 118)")
print(f"min / median / max mi  : {dm.distance_miles[dm.distance_miles>0].min():.1f} / "
      f"{dm.distance_miles.median():.1f} / {dm.distance_miles.max():.1f}")
print("backups                : centroids.orig.csv, distance_matrix.orig.csv")

"""
Regenerate centroid-derived columns in features_master.csv and port_proximity.csv.

These columns were computed from the broken centroids.csv (66 areas on wrong
shared coordinates), so they are wrong wherever an area was mislocated. After
rebuild_centroids.py corrected the coordinates, this script recomputes every
column that depends on geography, using haversine distance from each area's
corrected internal point:

    INTPTLAT, INTPTLON               <- corrected centroids
    nearest_port / _type / _miles    <- nearest of all 26 NTAD ports
    nearest_seaport / _miles         <- nearest seaport (type == 'seaport')
    min_distance_to_neighbor_miles   <- min over corrected distance_matrix

All non-geographic columns (VAL, TON, norms, vulnerability_score, rank,
is_metro, commodity fields) are left exactly as-is. Originals are backed up to
*.orig.csv before overwriting.
"""

from pathlib import Path
import shutil
import numpy as np
import pandas as pd

FEATURES = Path(__file__).resolve().parents[1] / "supply_chain_data" / "05_features"
R_MILES = 3958.7613


def haversine(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in miles. lat/lon in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_MILES * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- load corrected geography
cent = pd.read_csv(FEATURES / "centroids.csv").set_index("GEO_ID")
ports = pd.read_csv(FEATURES / "ntad_ports.csv")
dm = pd.read_csv(FEATURES / "distance_matrix.csv")

# nearest neighbor distance per origin (from corrected distance matrix)
min_neigh = dm.groupby("GEO_ID_origin")["distance_miles"].min()

# port arrays
p_lat = ports["lat"].to_numpy()
p_lon = ports["lon"].to_numpy()
p_name = ports["port_name"].to_numpy()
p_type = ports["type"].to_numpy()
sea = p_type == "seaport"


def port_features(geo_id):
    lat, lon = cent.loc[geo_id, ["INTPTLAT", "INTPTLON"]]
    d = haversine(lat, lon, p_lat, p_lon)
    i = int(np.argmin(d))                       # nearest of all ports
    j = int(np.argmin(np.where(sea, d, np.inf)))  # nearest seaport
    return pd.Series({
        "nearest_port": p_name[i],
        "nearest_port_type": p_type[i],
        "nearest_port_miles": round(float(d[i]), 2),
        "nearest_seaport": p_name[j],
        "nearest_seaport_miles": round(float(d[j]), 2),
    })


def regenerate(df):
    """Update only the geography-derived columns that already exist in df,
    preserving each file's original schema and column order."""
    df = df.copy()
    updates = df["GEO_ID"].apply(port_features)
    updates["INTPTLAT"] = df["GEO_ID"].map(cent["INTPTLAT"]).values
    updates["INTPTLON"] = df["GEO_ID"].map(cent["INTPTLON"]).values
    updates["min_distance_to_neighbor_miles"] = df["GEO_ID"].map(min_neigh).round(2).values
    for col in updates.columns:
        if col in df.columns:
            df[col] = updates[col].values
    return df


# ---------------------------------------------------------------- rewrite both files in place
for name in ["features_master.csv", "port_proximity.csv"]:
    path = FEATURES / name
    df = pd.read_csv(path)
    backup = FEATURES / name.replace(".csv", ".orig.csv")
    if not backup.exists():
        shutil.copy2(path, backup)
    out = regenerate(df)
    out.to_csv(path, index=False)
    changed = name == "features_master.csv"
    print(f"{name:22} rewritten ({len(out)} rows, columns preserved: {list(df.columns) == list(out.columns)})")

# ---------------------------------------------------------------- quick before/after sanity
prev = pd.read_csv(FEATURES / "features_master.orig.csv").set_index("GEO_ID")
now = pd.read_csv(FEATURES / "features_master.csv").set_index("GEO_ID")
print("\nSpot-check corrected port proximity (previously mislocated areas):")
for kw in ["Buffalo", "Baltimore", "San Diego"]:
    gid = now.index[now.NAME.str.contains(kw)][0]
    print(f"  {kw:10}: seaport {prev.loc[gid,'nearest_seaport_miles']:.0f} mi -> "
          f"{now.loc[gid,'nearest_seaport_miles']:.0f} mi "
          f"({now.loc[gid,'nearest_seaport']})")

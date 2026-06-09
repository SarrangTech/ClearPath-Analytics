# ClearPath Analytics — Gravity Model pipeline (Option C)

Scores all **17,822 directed CFS-area corridors** by disruption impact using a
spatial gravity model, and corrects the upstream geography it depends on.

```
gravity_ij = (VAL_i × VAL_j) / distance_ij²
```

High-gravity corridors move the most freight value over the shortest distance,
so a disruption there propagates the largest economic shock. The ranked output
feeds the **Network Graph** and **Rerouting** workstreams.

## Run order

```bash
pip install pyshp        # only extra dependency (pandas, numpy, matplotlib assumed)

python scripts/rebuild_centroids.py   # 1. fix coordinates + distance matrix
python scripts/rebuild_features.py    # 2. fix geo-derived feature columns
python scripts/gravity_model.py       # 3. score + rank corridors
```

Each script is idempotent and backs up any file it overwrites to `*.orig.csv`
(only on the first run, so backups always hold the original pre-fix data).

## Scripts

| Script | Reads | Writes |
|--------|-------|--------|
| `rebuild_centroids.py` | `2022_CFS_Areas.shp`, `centroids.csv` (names only) | `centroids.csv`, `distance_matrix.csv` |
| `rebuild_features.py` | corrected `centroids.csv`, `distance_matrix.csv`, `ntad_ports.csv` | `features_master.csv`, `port_proximity.csv` (geo columns only) |
| `gravity_model.py` | `distance_matrix.csv`, `features_master.csv` | `03_outputs/corridor_gravity_scores.csv`, `03_outputs/top_corridors_by_gravity.png` |

## The centroid bug (why steps 1–2 exist)

The original `centroids.csv` was geocoded by **area name**, which collapsed 66
of 134 CFS areas onto shared, often-wrong coordinates:

- Buffalo, Rochester, Albany → all placed on New York City's lat/lon
- Baltimore, MD → placed in Columbia, SC
- San Diego, Fresno, Sacramento → all placed on a generic California point

Result: 118 zero-distance corridors and silently-wrong distances on **13,266 of
17,822 corridors (74%)**. Before the fix, the gravity ranking was dominated by
these artifacts (top corridors all showed 0.0 mi).

**Fix:** the `2022_CFS_Areas` shapefile carries the Census-provided internal
point (`INTPTLAT`/`INTPTLON`) for every area — a guaranteed-inside,
authoritative coordinate. We read those directly (joined on GEO_ID =
`CFS22_GE_1`) and recompute all distances with haversine. All 134 areas now
have distinct, correct coordinates; 0 zero-distance corridors remain.

## Output: `corridor_gravity_scores.csv`

17,822 rows, ranked (rank 1 = highest disruption impact):

| Column | Meaning |
|--------|---------|
| `rank` | 1 = highest gravity |
| `GEO_ID_origin` / `NAME_origin` | origin CFS area |
| `GEO_ID_dest` / `NAME_dest` | destination CFS area |
| `VAL_origin` / `VAL_dest` | annual CFS shipment value ($K) per endpoint |
| `distance_miles` | corrected centroid-to-centroid great-circle distance |
| `distance_used` | distance with a min-floor applied (safety; floor now unused — 0 floored) |
| `dist_floored` | True if the floor was hit (all False after the centroid fix) |
| `gravity` | raw gravity score |
| `gravity_norm` | 0–1 normalized — **use directly as Network Graph edge weight** |

Matrix is directed, so `i→j` and `j→i` share the same gravity and appear as
adjacent rows.

## Note for the team

Steps 1–2 regenerate four **shared** feature files
(`centroids.csv`, `distance_matrix.csv`, `features_master.csv`,
`port_proximity.csv`). Any workstream that already consumed them must re-pull —
the previous distances were wrong for ~74% of corridors. **Non-geographic
columns are untouched** (VAL, TON, norms, `vulnerability_score`, rank,
commodity fields are byte-identical to the originals).

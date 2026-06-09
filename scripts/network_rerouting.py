"""
=============================================================================
Network Graph + Failure Simulation + Rerouting
Geospatial Supply Chain Vulnerability Analysis · MISM 6214 · Team 1
Shreya Pandey

DEPENDS ON (run these first):
  scripts/rebuild_centroids.py   → corrected centroids.csv, distance_matrix.csv
  scripts/rebuild_features.py    → features_master.csv
  scripts/gravity_model.py       → corridor_gravity_scores.csv
  supply_chain_data/03_outputs/Risk_Tier_Classifer_RF.ipynb
                                 → risk_tier_output.csv

NETWORK MODEL:
  - 134 CFS areas = nodes
  - K=8 nearest-neighbour graph (sparse / realistic)
  - Edge weight = distance_miles (Dijkstra shortest path)
  - Edge attribute gravity_norm pulled from corridor_gravity_scores.csv

FAILURE SIMULATION:
  - Simulate removal of each HIGH-tier area (top 10 % by vulnerability)
  - Risk tier sourced from RF classifier (risk_tier_output.csv)
  - For each removal: find all OD pairs whose shortest path used that node,
    measure added miles, compute rerouting cost

REROUTING COST FORMULA:
  cost ($) = extra_miles × (node_TON × 1000 short-tons) × $0.08 / ton-mile
  Result column rerouting_cost_B reported in billions USD.

OUTPUTS  →  supply_chain_data/06_network_outputs/
  01_freight_network_map.png        Static CONUS network map (dark theme)
  02_gravity_corridor_map.png       Gravity-weighted corridor overlay
  03_failure_simulation_table.png   Results table image
  04_rerouting_cost_chart.png       Bar charts (cost + OD disruption)
  05_interactive_network_map.html   Folium click-and-zoom map
  failure_simulation_results.csv    Summary — one row per HIGH area
  rerouting_detail.csv              Per-corridor rerouting detail
=============================================================================
"""

import os
import warnings

import folium
import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0.  PATHS
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEAT_FILE     = os.path.join(BASE, "supply_chain_data", "05_features", "features_master.csv")
CENT_FILE     = os.path.join(BASE, "supply_chain_data", "05_features", "centroids.csv")
DIST_FILE     = os.path.join(BASE, "supply_chain_data", "05_features", "distance_matrix.csv")
GRAVITY_FILE  = os.path.join(BASE, "supply_chain_data", "03_outputs",  "corridor_gravity_scores.csv")
RISK_FILE     = os.path.join(BASE, "supply_chain_data", "03_outputs",  "risk_tier_output.csv")
OUT_DIR       = os.path.join(BASE, "supply_chain_data", "06_network_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

COST_PER_TON_MILE = 0.08   # $/ton-mile  (BTS industry standard)
K_NEIGHBORS       = 8      # edges per node in the K-NN graph

print("=" * 70)
print("  Network Graph + Failure Simulation + Rerouting")
print("  Supply Chain Vulnerability Analysis · MISM 6214 · Team 1")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1.  LOAD DATA
# ---------------------------------------------------------------------------
print("\n[1/7] Loading pipeline outputs …")

feat_df    = pd.read_csv(FEAT_FILE)
cent_df    = pd.read_csv(CENT_FILE)
dist_df    = pd.read_csv(DIST_FILE)
gravity_df = pd.read_csv(GRAVITY_FILE)
risk_df    = pd.read_csv(RISK_FILE)

print(f"      features_master    : {len(feat_df)} CFS areas")
print(f"      centroids          : {len(cent_df)} areas  (corrected via rebuild_centroids.py)")
print(f"      distance_matrix    : {len(dist_df):,} corridors  (corrected haversine)")
print(f"      corridor_gravity   : {len(gravity_df):,} directed edges")
print(f"      risk_tier_output   : {len(risk_df)} areas  (RF classifier, LOOCV 90.3 %)")

def short_name(n):
    n = n.split(";")[0].strip()
    n = n.replace(" CFS Area", "").replace(" (part)", "")
    return (n[:32] + "…") if len(n) > 32 else n

feat_df["short_name"] = feat_df["NAME"].apply(short_name)
risk_df["short_name"] = risk_df["NAME"].apply(short_name)

# ---------------------------------------------------------------------------
# 2.  RISK TIERS  (from RF classifier output)
# ---------------------------------------------------------------------------
print("\n[2/7] Reading risk tiers from RF classifier …")

# risk_tier_output.csv columns: GEO_ID, NAME, VAL, TON, vulnerability_score,
#                                risk_tier, predicted_tier
# We use risk_tier (ground-truth percentile label validated by the RF model).
TIER_MAP = {"High": "HIGH", "Medium": "MEDIUM", "Low": "LOW"}
risk_df["tier"] = risk_df["risk_tier"].map(TIER_MAP)

high_df = risk_df[risk_df["tier"] == "HIGH"].copy()
N_HIGH  = len(high_df)

tier_counts = risk_df["tier"].value_counts()
print(f"      HIGH   : {tier_counts.get('HIGH',   0)}")
print(f"      MEDIUM : {tier_counts.get('MEDIUM', 0)}")
print(f"      LOW    : {tier_counts.get('LOW',    0)}")

print("\n      ── HIGH-TIER AREAS (RF-classified) ──")
for _, r in risk_df[risk_df["tier"] == "HIGH"].sort_values("vulnerability_score", ascending=False).iterrows():
    print(f"        {r['short_name']:<40}  score={r['vulnerability_score']:.4f}")

# 3.  ASSEMBLE NODE METADATA

node_meta = (
    feat_df
    .merge(cent_df[["GEO_ID", "INTPTLAT", "INTPTLON"]], on="GEO_ID", how="left", suffixes=("", "_cent"))
    .merge(risk_df[["GEO_ID", "tier"]], on="GEO_ID", how="left")
)

# Prefer corrected centroid coordinates; fall back to feat_df columns
if "INTPTLAT_cent" in node_meta.columns:
    node_meta["lat"] = node_meta["INTPTLAT_cent"].fillna(node_meta["INTPTLAT"])
    node_meta["lon"] = node_meta["INTPTLON_cent"].fillna(node_meta["INTPTLON"])
else:
    node_meta["lat"] = node_meta["INTPTLAT"]
    node_meta["lon"] = node_meta["INTPTLON"]

# Fill any missing tiers using vulnerability percentile
p90 = node_meta["vulnerability_score"].quantile(0.90)
p75 = node_meta["vulnerability_score"].quantile(0.75)
node_meta["tier"] = node_meta["tier"].fillna(
    node_meta["vulnerability_score"].apply(
        lambda s: "HIGH" if s >= p90 else ("MEDIUM" if s >= p75 else "LOW")
    )
)

TIER_COLOR = {
    "HIGH":   "#d62728",
    "MEDIUM": "#ff7f0e",
    "LOW":    "#aec7e8",
}

# 4.  BUILD K-NN GRAPH

print(f"\n[3/7] Building K={K_NEIGHBORS} nearest-neighbour graph …")

# Build gravity lookup: (origin_id, dest_id) → gravity_norm
gravity_lookup = (
    gravity_df.set_index(["GEO_ID_origin", "GEO_ID_dest"])["gravity_norm"]
    .to_dict()
)

G = nx.Graph()

for _, r in node_meta.iterrows():
    G.add_node(
        r["GEO_ID"],
        name       = r["short_name"],
        full_name  = r["NAME"],
        lat        = r["lat"],
        lon        = r["lon"],
        vuln_score = r["vulnerability_score"],
        tier       = r["tier"],
        color      = TIER_COLOR.get(r["tier"], "#aec7e8"),
        VAL        = r["VAL"],
        TON        = r["TON"],
        seaport_mi = float(r["nearest_seaport_miles"]) if pd.notna(r.get("nearest_seaport_miles")) else 999.0,
    )

edges_added = set()
dist_positive = dist_df[dist_df["distance_miles"] > 0].copy()

for origin_id, grp in dist_positive.groupby("GEO_ID_origin"):
    for _, row in grp.nsmallest(K_NEIGHBORS, "distance_miles").iterrows():
        u, v = row["GEO_ID_origin"], row["GEO_ID_dest"]
        key  = tuple(sorted([u, v]))
        if key in edges_added:
            continue
        grav = gravity_lookup.get((u, v), gravity_lookup.get((v, u), 0.0))
        G.add_edge(
            u, v,
            distance    = row["distance_miles"],
            weight      = row["distance_miles"],
            gravity_norm= grav,
        )
        edges_added.add(key)

print(f"      Nodes          : {G.number_of_nodes()}")
print(f"      Edges          : {G.number_of_edges():,}")
print(f"      Graph connected: {nx.is_connected(G)}")

# Ensure connectivity by bridging isolated components
if not nx.is_connected(G):
    comps     = list(nx.connected_components(G))
    main_comp = max(comps, key=len)
    print(f"      ⚠  {len(comps)} components — adding bridge edges …")
    for comp in comps:
        if comp == main_comp:
            continue
        for node in comp:
            best = dist_positive[
                (dist_positive["GEO_ID_origin"] == node) &
                (dist_positive["GEO_ID_dest"].isin(main_comp))
            ].nsmallest(1, "distance_miles")
            if not best.empty:
                row  = best.iloc[0]
                u, v = row["GEO_ID_origin"], row["GEO_ID_dest"]
                grav = gravity_lookup.get((u, v), gravity_lookup.get((v, u), 0.0))
                G.add_edge(u, v, distance=row["distance_miles"],
                           weight=row["distance_miles"], gravity_norm=grav)
    print(f"      Graph connected after bridging: {nx.is_connected(G)}")

# 5.  FAILURE SIMULATION

print(f"\n[4/7] Failure simulation — {N_HIGH} HIGH-tier areas …")

print("      Pre-computing all-pairs shortest paths (~20 s) …")
all_sp_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
all_sp_paths   = dict(nx.all_pairs_dijkstra_path(G,        weight="weight"))
print("      ✓  All-pairs shortest paths ready")

high_ids    = list(high_df["GEO_ID"])
results     = []
detail_rows = []

for idx, failed_id in enumerate(high_ids):
    if failed_id not in G.nodes:
        print(f"  [{idx+1:>2}/{N_HIGH}] {failed_id} — not in graph, skipping")
        continue

    nd      = G.nodes[failed_id]
    fname   = nd["name"]
    f_ton   = nd["TON"]
    f_val   = nd["VAL"]
    f_score = nd["vuln_score"]

    G_fail  = G.copy()
    G_fail.remove_node(failed_id)

    all_other   = [n for n in G.nodes() if n != failed_id]
    affected    = []
    extra_total = 0.0
    disconnected= 0

    for orig in all_other:
        for dest in all_other:
            if orig >= dest:
                continue
            orig_path = all_sp_paths.get(orig, {}).get(dest, [])
            if failed_id not in orig_path:
                continue

            orig_len = all_sp_lengths[orig][dest]
            try:
                new_len  = nx.dijkstra_path_length(G_fail, orig, dest, weight="weight")
                new_path = nx.dijkstra_path(G_fail, orig, dest, weight="weight")
                extra    = max(0.0, new_len - orig_len)
            except nx.NetworkXNoPath:
                new_len  = None
                new_path = []
                extra    = 0.0
                disconnected += 1

            extra_total += extra

            if extra > 0 or new_len is None:
                row = {
                    "failed_area"   : fname,
                    "origin"        : G.nodes[orig]["name"],
                    "destination"   : G.nodes[dest]["name"],
                    "orig_miles"    : round(orig_len, 1),
                    "new_miles"     : round(new_len, 1) if new_len is not None else None,
                    "extra_miles"   : round(extra, 1),
                    "disconnected"  : new_len is None,
                    "orig_path_hops": len(orig_path),
                    "new_path_hops" : len(new_path),
                }
                affected.append(row)
                detail_rows.append(row)

    # cost in billions USD
    # TON field is in thousands of short-tons → ×1000 = actual tons
    cost_B = (extra_total * f_ton * 1000 * COST_PER_TON_MILE) / 1e9

    results.append({
        "failed_area"       : fname,
        "geo_id"            : failed_id,
        "vuln_score"        : round(f_score, 4),
        "freight_value_B"   : round(f_val / 1000, 1),
        "freight_ton_M"     : round(f_ton / 1000, 1),
        "affected_od_pairs" : len(affected),
        "extra_miles_total" : round(extra_total, 1),
        "disconnected_pairs": disconnected,
        "rerouting_cost_B"  : round(cost_B, 4),
    })

    tag = f"${cost_B:.3f}B" if cost_B > 0 else "—"
    print(f"  [{idx+1:>2}/{N_HIGH}] {fname:<38}  "
          f"affected={len(affected):>4}  extra_mi={extra_total:>8,.0f}  cost={tag}")

results_df = pd.DataFrame(results).sort_values("rerouting_cost_B", ascending=False)
detail_df  = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame()

results_df.to_csv(os.path.join(OUT_DIR, "failure_simulation_results.csv"), index=False)
if not detail_df.empty:
    detail_df.to_csv(os.path.join(OUT_DIR, "rerouting_detail.csv"), index=False)

print(f"\n  Simulation results saved")

# 6.  VISUALISATIONS

print("\n[5/7] Generating visualisations …")

# Node positions for matplotlib (lon, lat)
pos_all   = {n: (d["lon"], d["lat"]) for n, d in G.nodes(data=True)
             if d["lon"] is not None and d["lat"] is not None}
pos_conus = {n: (lon, lat) for n, (lon, lat) in pos_all.items()
             if -130 < lon < -60 and 22 < lat < 52}
conus     = list(pos_conus.keys())
conus_set = set(conus)
high_set  = set(high_df["GEO_ID"])

# ── OUTPUT 1: Static K-NN Network Map ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 13))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

conus_edges      = [(u, v) for u, v in G.edges() if u in conus_set and v in conus_set]
high_adj_edges   = [(u, v) for u, v in conus_edges
                    if G.nodes[u]["tier"] == "HIGH" or G.nodes[v]["tier"] == "HIGH"]
other_edges      = [(u, v) for u, v in conus_edges if (u, v) not in high_adj_edges]

nx.draw_networkx_edges(G, pos_conus, edgelist=other_edges,
                       alpha=0.12, width=0.5, edge_color="#4a90d9", ax=ax)
nx.draw_networkx_edges(G, pos_conus, edgelist=high_adj_edges,
                       alpha=0.50, width=1.4, edge_color="#ff4444", ax=ax)

node_colors = [G.nodes[n]["color"] for n in conus]
node_sizes  = [max(40, G.nodes[n]["vuln_score"] * 900) for n in conus]
nx.draw_networkx_nodes(G, pos_conus, nodelist=conus,
                       node_color=node_colors, node_size=node_sizes,
                       alpha=0.93, ax=ax)

high_labels = {n: G.nodes[n]["name"] for n in conus if n in high_set}
nx.draw_networkx_labels(G, pos_conus, labels=high_labels,
                        font_size=7, font_color="white", font_weight="bold", ax=ax,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#d62728",
                                  edgecolor="white", alpha=0.88, linewidth=0.6))

legend_els = [
    mpatches.Patch(color="#d62728", label=f"HIGH  (n={N_HIGH}) — RF top 10 %"),
    mpatches.Patch(color="#ff7f0e", label="MEDIUM — RF 75th–90th pct"),
    mpatches.Patch(color="#aec7e8", label="LOW — RF below 75th pct"),
    mlines.Line2D([], [], color="#ff4444", lw=1.5, alpha=0.7,
                  label="Corridor linked to HIGH node"),
    mlines.Line2D([], [], color="#4a90d9", lw=0.7, alpha=0.4,
                  label=f"K={K_NEIGHBORS} nearest-neighbour edges"),
]
ax.legend(handles=legend_els, loc="lower left", fontsize=9,
          framealpha=0.88, facecolor="#1a1a2e",
          edgecolor="#4a90d9", labelcolor="white")

ax.set_title(
    f"U.S. Freight Network — 134 CFS Areas · K={K_NEIGHBORS} Nearest-Neighbour Graph\n"
    "Node size ∝ vulnerability score · Risk tier from RF classifier (90.3 % LOOCV accuracy)",
    color="white", fontsize=13, fontweight="bold", pad=14)
ax.axis("off")
ax.set_xlim(-128, -65)
ax.set_ylim(23, 50)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "01_freight_network_map.png")
plt.savefig(out1, dpi=180, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"  OUTPUT 1 → {os.path.basename(out1)}")

# ── OUTPUT 2: Gravity-Weighted Corridor Map ──────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 13))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

edges_with_gravity = [
    (u, v, d["gravity_norm"])
    for u, v, d in G.edges(data=True)
    if u in conus_set and v in conus_set and d.get("gravity_norm", 0) > 0
]
edges_with_gravity.sort(key=lambda x: x[2])

edge_cmap = plt.cm.YlOrRd
for u, v, gnorm in edges_with_gravity:
    col   = edge_cmap(0.3 + 0.7 * gnorm)
    width = 0.4 + 3.5 * gnorm
    alpha = 0.15 + 0.55 * gnorm
    ax.plot(
        [pos_conus[u][0], pos_conus[v][0]],
        [pos_conus[u][1], pos_conus[v][1]],
        color=col, linewidth=width, alpha=alpha, solid_capstyle="round",
    )

# Edges with no gravity data
no_gravity = [(u, v) for u, v, d in G.edges(data=True)
              if u in conus_set and v in conus_set and d.get("gravity_norm", 0) == 0]
nx.draw_networkx_edges(G, pos_conus, edgelist=no_gravity,
                       alpha=0.08, width=0.4, edge_color="#4a90d9", ax=ax)

nx.draw_networkx_nodes(G, pos_conus, nodelist=conus,
                       node_color=node_colors, node_size=node_sizes,
                       alpha=0.93, ax=ax)
nx.draw_networkx_labels(G, pos_conus, labels=high_labels,
                        font_size=7, font_color="white", font_weight="bold", ax=ax,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#d62728",
                                  edgecolor="white", alpha=0.88, linewidth=0.6))

sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.025, pad=0.02,
                    location="bottom", shrink=0.4)
cbar.set_label("Gravity Score (normalised)", color="white", fontsize=9)
cbar.ax.xaxis.set_tick_params(color="white")
plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white", fontsize=8)

ax.set_title(
    "U.S. Freight Corridors — Gravity-Weighted Network\n"
    "Edge colour & width ∝ gravity_norm = (VAL_i × VAL_j) / distance²  "
    "(from gravity_model.py)",
    color="white", fontsize=13, fontweight="bold", pad=14)
ax.axis("off")
ax.set_xlim(-128, -65)
ax.set_ylim(23, 50)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "02_gravity_corridor_map.png")
plt.savefig(out2, dpi=180, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"     OUTPUT 2 → {os.path.basename(out2)}")

# ── OUTPUT 3: Failure Simulation Table ──────────────────────────────────────
plot_order = results_df.reset_index(drop=True)

display = plot_order[[
    "failed_area", "vuln_score", "freight_value_B",
    "affected_od_pairs", "extra_miles_total", "rerouting_cost_B",
]].copy()
display.columns = [
    "CFS Area", "Vuln Score", "Value ($B)",
    "Affected OD\nPairs", "Extra Miles\nAdded", "Rerouting\nCost ($B)",
]
display["Value ($B)"]         = display["Value ($B)"].apply(lambda x: f"${x:,.0f}B")
display["Rerouting\nCost ($B)"] = display["Rerouting\nCost ($B)"].apply(
    lambda x: f"${x:.2f}B" if x > 0 else "—")
display["Extra Miles\nAdded"] = display["Extra Miles\nAdded"].apply(
    lambda x: f"{x:,.0f}" if x > 0 else "—")

fig, ax = plt.subplots(figsize=(20, 7))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
ax.axis("off")

tbl = ax.table(cellText=display.values, colLabels=display.columns,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 2.3)

for j in range(len(display.columns)):
    tbl[0, j].set_facecolor("#1f4e79")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

for i in range(1, len(display) + 1):
    cost_val = plot_order.iloc[i - 1]["rerouting_cost_B"]
    for j in range(len(display.columns)):
        if cost_val > 0:
            tbl[i, j].set_facecolor("#4d0000")
            tbl[i, j].set_text_props(color="white")
        elif i % 2 == 0:
            tbl[i, j].set_facecolor("#1a1a2e")
            tbl[i, j].set_text_props(color="#cccccc")
        else:
            tbl[i, j].set_facecolor("#0d1117")
            tbl[i, j].set_text_props(color="#cccccc")

ax.set_title(
    "Failure Simulation — HIGH-Tier Areas Ranked by Rerouting Cost\n"
    f"K={K_NEIGHBORS} NN network · Cost = extra miles × TON × $0.08/ton-mile"
    "  |  Risk tier: RF classifier (LOOCV 90.3 %)",
    color="white", fontsize=11, fontweight="bold", pad=18)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "03_failure_simulation_table.png")
plt.savefig(out3, dpi=180, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"    OUTPUT 3 → {os.path.basename(out3)}")

# ── OUTPUT 4: Rerouting Cost Bar Charts ─────────────────────────────────────
def _bar_colors(series, n_red=3, n_orange=3):
    ranked = series.rank(ascending=False)
    out = []
    for v in ranked:
        if v <= n_red:                out.append("#d62728")
        elif v <= n_red + n_orange:   out.append("#ff7f0e")
        else:                         out.append("#4a90d9")
    return out

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.patch.set_facecolor("#0d1117")

ax1 = axes[0]
ax1.set_facecolor("#0d1117")
plot_a = results_df.sort_values("rerouting_cost_B", ascending=True)
bars1  = ax1.barh(plot_a["failed_area"], plot_a["rerouting_cost_B"],
                  color=_bar_colors(plot_a["rerouting_cost_B"]),
                  edgecolor="white", linewidth=0.3, alpha=0.9, height=0.7)
max_cost = plot_a["rerouting_cost_B"].max()
for bar, val in zip(bars1, plot_a["rerouting_cost_B"]):
    label = f"${val:.2f}B" if val > 0 else "—  (direct)"
    ax1.text(bar.get_width() + max_cost * 0.01, bar.get_y() + bar.get_height() / 2,
             label, va="center", ha="left", color="white", fontsize=8, fontweight="bold")
ax1.set_xlabel("Estimated Rerouting Cost (Billions USD)", color="white", fontsize=10)
ax1.set_title("Rerouting Cost if Area Fails\n(extra miles × tonnage × $0.08/ton-mile)",
              color="white", fontsize=11, fontweight="bold")
ax1.tick_params(colors="white", labelsize=8)
for sp in ax1.spines.values(): sp.set_edgecolor("#444")
ax1.set_xlim(0, max(max_cost * 1.35, 0.001))

ax2 = axes[1]
ax2.set_facecolor("#0d1117")
plot_b = results_df.sort_values("affected_od_pairs", ascending=True)
bars2  = ax2.barh(plot_b["failed_area"], plot_b["affected_od_pairs"],
                  color=_bar_colors(plot_b["affected_od_pairs"]),
                  edgecolor="white", linewidth=0.3, alpha=0.9, height=0.7)
for bar, val in zip(bars2, plot_b["affected_od_pairs"]):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             str(int(val)), va="center", ha="left",
             color="white", fontsize=8.5, fontweight="bold")
ax2.set_xlabel("OD Pairs Forced to Reroute", color="white", fontsize=10)
ax2.set_title("Freight Corridors Disrupted if Area Fails\n"
              "(OD pairs whose shortest path used that node)",
              color="white", fontsize=11, fontweight="bold")
ax2.tick_params(colors="white", labelsize=8)
for sp in ax2.spines.values(): sp.set_edgecolor("#444")
ax2.set_xlim(0, max(plot_b["affected_od_pairs"].max() * 1.2, 1))

fig.suptitle(
    f"Supply Chain Disruption Impact — All {N_HIGH} HIGH-Tier Area Failures Simulated\n"
    f"Team 1 · MISM 6214 · K={K_NEIGHBORS} NN network · 2022 US Census CFS",
    color="white", fontsize=13, fontweight="bold", y=1.01)

plt.tight_layout()
out4 = os.path.join(OUT_DIR, "04_rerouting_cost_chart.png")
plt.savefig(out4, dpi=180, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"     OUTPUT 4 → {os.path.basename(out4)}")

# ── OUTPUT 5: Interactive Folium Map ─────────────────────────────────────────
print("      Building interactive HTML map …")

m = folium.Map(location=[38.5, -96.5], zoom_start=4, tiles="CartoDB dark_matter")

TIER_FCOL   = {"HIGH": "#d62728", "MEDIUM": "#ff7f0e", "LOW": "#6baed6"}
TIER_RADIUS = {"HIGH": 14, "MEDIUM": 9, "LOW": 5}

res_lookup = results_df.set_index("geo_id")[
    ["rerouting_cost_B", "affected_od_pairs", "extra_miles_total"]
].to_dict("index")

for _, row in node_meta.iterrows():
    geo_id = row["GEO_ID"]
    lat    = row["lat"]
    lon    = row["lon"]
    if pd.isna(lat) or pd.isna(lon):
        continue

    t     = row["tier"]
    col   = TIER_FCOL.get(t, "#6baed6")
    rad   = TIER_RADIUS.get(t, 5)
    sname = row["short_name"]

    extra_html = ""
    if t == "HIGH" and geo_id in res_lookup:
        rc = res_lookup[geo_id]["rerouting_cost_B"]
        ap = res_lookup[geo_id]["affected_od_pairs"]
        em = res_lookup[geo_id]["extra_miles_total"]
        extra_html = (
            f"<br><b>Rerouting Cost:</b> "
            f"{'$' + f'{rc:.2f}' + 'B' if rc > 0 else 'Direct connections only'}"
            f"<br><b>OD Pairs Disrupted:</b> {int(ap)}"
            f"<br><b>Extra Miles Added:</b> {em:,.0f}"
        )

    popup_html = f"""
    <div style='font-family:Arial;min-width:240px;'>
      <h4 style='color:{col};margin:0 0 4px;'>{sname}</h4>
      <hr style='border-color:{col};margin:4px 0;'>
      <b>Risk Tier:</b> <span style='color:{col};font-weight:bold;'>{t}</span>
        <span style='font-size:10px;color:#aaa;'>(RF classifier)</span><br>
      <b>Vuln Score:</b> {row['vulnerability_score']:.4f}<br>
      <b>Freight Value:</b> ${row['VAL']/1e3:,.0f}B<br>
      <b>Tonnage:</b> {row['TON']/1e6:.2f}B tons<br>
      <b>Nearest Seaport:</b> {row.get('nearest_seaport','—')}
        ({row.get('nearest_seaport_miles', '?'):.0f} mi)
      {extra_html}
    </div>"""

    folium.CircleMarker(
        location=[lat, lon],
        radius=rad,
        color=col,
        fill=True,
        fill_color=col,
        fill_opacity=0.85,
        weight=2 if t == "HIGH" else 1,
        tooltip=sname,
        popup=folium.Popup(popup_html, max_width=300),
    ).add_to(m)

# Draw edges adjacent to HIGH nodes, coloured by gravity
edge_layer = folium.FeatureGroup(name="HIGH-adjacent corridors")
for u, v, d in G.edges(data=True):
    if u not in high_set and v not in high_set:
        continue
    u_lat = G.nodes[u].get("lat"); u_lon = G.nodes[u].get("lon")
    v_lat = G.nodes[v].get("lat"); v_lon = G.nodes[v].get("lon")
    if any(pd.isna(x) for x in [u_lat, u_lon, v_lat, v_lon]):
        continue
    if not (-130 < u_lon < -60 and 22 < u_lat < 52):
        continue
    if not (-130 < v_lon < -60 and 22 < v_lat < 52):
        continue

    gnorm = d.get("gravity_norm", 0)
    # Colour: low gravity → blue, high → red
    r_ch  = int(60  + 195 * gnorm)
    b_ch  = int(200 - 200 * gnorm)
    edge_col = f"#{r_ch:02x}44{b_ch:02x}"
    weight_w = 1.0 + 2.5 * gnorm

    folium.PolyLine(
        [[u_lat, u_lon], [v_lat, v_lon]],
        color=edge_col,
        weight=weight_w,
        opacity=0.35 + 0.35 * gnorm,
        tooltip=(f"{G.nodes[u]['name']} ↔ {G.nodes[v]['name']}"
                 f"  {d['distance']:.0f} mi  gravity={gnorm:.3f}"),
    ).add_to(edge_layer)
edge_layer.add_to(m)
folium.LayerControl().add_to(m)

legend_html = """
<div style='position:fixed;bottom:30px;left:30px;z-index:1000;
            background:rgba(10,10,20,0.92);padding:14px 18px;
            border-radius:8px;border:1px solid #4a90d9;
            font-family:Arial;color:white;font-size:12px;'>
  <b style='font-size:13px;'>Risk Tier (RF Classifier)</b><br>
  <span style='color:#d62728;font-size:16px;'>●</span> HIGH — top 10 %<br>
  <span style='color:#ff7f0e;font-size:16px;'>●</span> MEDIUM — 75th–90th pct<br>
  <span style='color:#6baed6;font-size:16px;'>●</span> LOW — below 75th pct<br>
  <span style='color:#ff4444;'>──</span> HIGH-adjacent corridor<br>
  <hr style='border-color:#4a90d9;margin:6px 0;'>
  <i style='font-size:10px;'>Edge colour ∝ gravity score</i><br>
  <i style='font-size:10px;'>Click any node for full details</i><br>
  <i style='font-size:10px;'>Team 1 · MISM 6214 · 2022 CFS</i>
</div>"""
m.get_root().html.add_child(folium.Element(legend_html))

out5 = os.path.join(OUT_DIR, "05_interactive_network_map.html")
m.save(out5)
print(f"      OUTPUT 5 → {os.path.basename(out5)}")

# 7.  SUMMARY

print("\n[6/7] Summary\n")
print("=" * 70)
print(f"  Graph : {G.number_of_nodes()} nodes · {G.number_of_edges()} edges  (K={K_NEIGHBORS} NN)")
print(f"  HIGH-tier areas simulated : {N_HIGH}  (RF classifier)")
print()

affected_any = results_df[results_df["affected_od_pairs"] > 0]
if not affected_any.empty:
    print(f"  Areas that force rerouting when removed : {len(affected_any)}")
    print()
    print("  TOP 5 — REROUTING COST:")
    for _, r in affected_any.sort_values("rerouting_cost_B", ascending=False).head(5).iterrows():
        print(f"    {r['failed_area']:<40}  cost=${r['rerouting_cost_B']:.2f}B  "
              f"OD={r['affected_od_pairs']}")
    print()
    print("  TOP 5 — CORRIDORS DISRUPTED:")
    for _, r in affected_any.sort_values("affected_od_pairs", ascending=False).head(5).iterrows():
        print(f"    {r['failed_area']:<40}  OD={r['affected_od_pairs']}  "
              f"extra_mi={r['extra_miles_total']:,.0f}")
else:
    print("  ℹ  No OD pairs forced to reroute.")
    print("     The network is resilient at K=8 — no HIGH node is a sole bridge.")
    print("     Try K=4 for a more fragile topology.")

total_cost = results_df["rerouting_cost_B"].sum()
total_od   = results_df["affected_od_pairs"].sum()
print(f"\n  Aggregate rerouting cost (all HIGH failures) : ${total_cost:,.2f}B")
print(f"  Aggregate OD pairs disrupted                 : {int(total_od)}")
print()

print("  OUTPUT FILES  →  supply_chain_data/06_network_outputs/")
for fname in [
    "01_freight_network_map.png",
    "02_gravity_corridor_map.png",
    "03_failure_simulation_table.png",
    "04_rerouting_cost_chart.png",
    "05_interactive_network_map.html",
    "failure_simulation_results.csv",
    "rerouting_detail.csv",
]:
    fp     = os.path.join(OUT_DIR, fname)
    status = "Done" if os.path.exists(fp) else "—"
    print(f"    {status}  {fname}")

print()
print("=" * 70)
print("  DONE — open 05_interactive_network_map.html in your browser")
print("=" * 70)

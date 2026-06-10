"""
=============================================================================
Neo4j Graph Loader — Supply Chain Vulnerability Network
Geospatial Supply Chain Vulnerability Analysis · MISM 6214 · Team 1
Shreya Pandey

WHAT THIS SCRIPT DOES:
  Reads the existing CSV outputs from the pipeline and loads them into a
  Neo4j graph database for interactive exploration and Cypher querying.

  Does NOT modify any existing files.

SETUP (one-time):
  1. Install Neo4j Desktop → https://neo4j.com/download/
     OR use Neo4j AuraDB (free cloud) → https://neo4j.com/cloud/platform/aura-graph-database/
  2. Create a new database and start it
  3. Set your connection details in the CONFIG section below
  4. pip install neo4j

RUN:
  python scripts/neo4j_graph.py

GRAPH MODEL:
  (:CFSArea)  — 134 nodes, one per CFS area
      Properties: geo_id, name, val, ton, vulnerability_score, risk_tier,
                  lat, lon, is_metro, nearest_seaport, nearest_seaport_miles,
                  num_commodities, val_ton_ratio

  [:CORRIDOR] — K=8 nearest-neighbour directed edges
      Properties: distance_miles, gravity_norm, gravity_rank

USEFUL CYPHER QUERIES (paste into Neo4j Browser):
  -- All HIGH-tier nodes
  MATCH (n:CFSArea {risk_tier: 'HIGH'}) RETURN n

  -- K-NN neighbours of LA
  MATCH (a:CFSArea)-[r:CORRIDOR]-(b:CFSArea)
  WHERE a.name CONTAINS 'Los Angeles'
  RETURN a, r, b

  -- Shortest path between two cities (by distance)
  MATCH (a:CFSArea {name: 'Los Angeles-Long Beach, CA'}),
        (b:CFSArea {name: 'Chicago-Naperville, IL-IN-WI'})
  CALL apoc.algo.dijkstra(a, b, 'CORRIDOR', 'distance_miles')
  YIELD path, weight
  RETURN path, weight

  -- Top 10 highest gravity corridors
  MATCH ()-[r:CORRIDOR]->()
  RETURN r.origin_name, r.dest_name, r.gravity_norm
  ORDER BY r.gravity_norm DESC LIMIT 10

  -- Most connected HIGH-tier node
  MATCH (a:CFSArea {risk_tier: 'HIGH'})-[r:CORRIDOR]-()
  RETURN a.name, count(r) AS connections
  ORDER BY connections DESC

  -- All corridors above gravity threshold
  MATCH (a:CFSArea)-[r:CORRIDOR]->(b:CFSArea)
  WHERE r.gravity_norm > 0.5
  RETURN a, r, b
=============================================================================
"""

import os
import warnings

import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG — credentials loaded from .env file (never committed to git)
# Create a .env file in the project root with:
#   NEO4J_URI=bolt://localhost:7687
#   NEO4J_USER=neo4j
#   NEO4J_PASSWORD=your_password
# ---------------------------------------------------------------------------
load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

K_NEIGHBORS = 8

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_FILE    = os.path.join(BASE, "supply_chain_data", "05_features", "features_master.csv")
CENT_FILE    = os.path.join(BASE, "supply_chain_data", "05_features", "centroids.csv")
DIST_FILE    = os.path.join(BASE, "supply_chain_data", "05_features", "distance_matrix.csv")
GRAVITY_FILE = os.path.join(BASE, "supply_chain_data", "03_outputs",  "corridor_gravity_scores.csv")
RISK_FILE    = os.path.join(BASE, "supply_chain_data", "03_outputs",  "risk_tier_output.csv")

print("=" * 68)
print("  Neo4j Graph Loader — Supply Chain Vulnerability Network")
print("  MISM 6214 · Team 1 · Shreya Pandey")
print("=" * 68)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("\n[1/5] Loading pipeline CSV files …")

feat_df    = pd.read_csv(FEAT_FILE)
cent_df    = pd.read_csv(CENT_FILE)
dist_df    = pd.read_csv(DIST_FILE)
gravity_df = pd.read_csv(GRAVITY_FILE)
risk_df    = pd.read_csv(RISK_FILE)

print(f"      features_master  : {len(feat_df)} areas")
print(f"      centroids        : {len(cent_df)} areas")
print(f"      distance_matrix  : {len(dist_df):,} corridors")
print(f"      gravity_scores   : {len(gravity_df):,} directed edges")
print(f"      risk_tier_output : {len(risk_df)} areas")

# ---------------------------------------------------------------------------
# 2. PREPARE NODE DATA
# ---------------------------------------------------------------------------
print("\n[2/5] Preparing node data …")

TIER_MAP = {"High": "HIGH", "Medium": "MEDIUM", "Low": "LOW"}
risk_df["tier"] = risk_df["risk_tier"].map(TIER_MAP)

node_df = (
    feat_df
    .merge(cent_df[["GEO_ID", "INTPTLAT", "INTPTLON"]], on="GEO_ID", how="left",
           suffixes=("", "_cent"))
    .merge(risk_df[["GEO_ID", "tier"]], on="GEO_ID", how="left")
)

if "INTPTLAT_cent" in node_df.columns:
    node_df["lat"] = node_df["INTPTLAT_cent"].fillna(node_df["INTPTLAT"])
    node_df["lon"] = node_df["INTPTLON_cent"].fillna(node_df["INTPTLON"])
else:
    node_df["lat"] = node_df["INTPTLAT"]
    node_df["lon"] = node_df["INTPTLON"]

# Fallback tier using percentile
p90 = node_df["vulnerability_score"].quantile(0.90)
p75 = node_df["vulnerability_score"].quantile(0.75)
node_df["tier"] = node_df["tier"].fillna(
    node_df["vulnerability_score"].apply(
        lambda s: "HIGH" if s >= p90 else ("MEDIUM" if s >= p75 else "LOW")
    )
)

def clean_name(n):
    n = n.split(";")[0].strip()
    return n.replace(" CFS Area", "").replace(" (part)", "")

node_df["clean_name"] = node_df["NAME"].apply(clean_name)

print(f"      Nodes ready      : {len(node_df)}")
print(f"        HIGH   : {(node_df['tier']=='HIGH').sum()}")
print(f"        MEDIUM : {(node_df['tier']=='MEDIUM').sum()}")
print(f"        LOW    : {(node_df['tier']=='LOW').sum()}")

# ---------------------------------------------------------------------------
# 3. PREPARE EDGE DATA (K=8 KNN)
# ---------------------------------------------------------------------------
print("\n[3/5] Preparing K=8 KNN edge data …")

gravity_lookup = (
    gravity_df.set_index(["GEO_ID_origin", "GEO_ID_dest"])[["gravity_norm", "rank"]]
    .to_dict("index")
)

edges = []
dist_positive = dist_df[dist_df["distance_miles"] > 0].copy()
edges_added   = set()

for origin_id, grp in dist_positive.groupby("GEO_ID_origin"):
    for _, row in grp.nsmallest(K_NEIGHBORS, "distance_miles").iterrows():
        u, v = row["GEO_ID_origin"], row["GEO_ID_dest"]
        key  = tuple(sorted([u, v]))
        if key in edges_added:
            continue
        grav_data = gravity_lookup.get((u, v), gravity_lookup.get((v, u), {}))
        edges.append({
            "origin_id"     : u,
            "dest_id"       : v,
            "distance_miles": round(row["distance_miles"], 2),
            "gravity_norm"  : round(grav_data.get("gravity_norm", 0.0), 6),
            "gravity_rank"  : int(grav_data.get("rank", 0)),
            "origin_name"   : row["NAME_origin"].split(";")[0].strip(),
            "dest_name"     : row["NAME_dest"].split(";")[0].strip(),
        })
        edges_added.add(key)

edge_df = pd.DataFrame(edges)
print(f"      Edges ready      : {len(edge_df):,} (undirected K-NN corridors)")

# ---------------------------------------------------------------------------
# 4. LOAD INTO NEO4J
# ---------------------------------------------------------------------------
print("\n[4/5] Connecting to Neo4j and loading graph …")
print(f"      URI  : {NEO4J_URI}")
print(f"      User : {NEO4J_USER}")

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("      ✅ Connected to Neo4j")
except Exception as e:
    print(f"\n  ❌ Could not connect to Neo4j: {e}")
    print("     Make sure Neo4j is running and your password is correct in CONFIG.")
    raise SystemExit(1)

def load_graph(tx_session, nodes, edges):

    # -- Clear existing data --
    print("      Clearing existing graph data …")
    tx_session.run("MATCH (n) DETACH DELETE n")

    # -- Create constraint (unique GEO_ID) --
    tx_session.run("""
        CREATE CONSTRAINT cfs_geo_id IF NOT EXISTS
        FOR (n:CFSArea) REQUIRE n.geo_id IS UNIQUE
    """)

    # -- Load nodes in batches --
    print(f"      Loading {len(nodes)} CFSArea nodes …")
    batch_size = 50
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i : i + batch_size]
        tx_session.run("""
            UNWIND $batch AS row
            MERGE (n:CFSArea {geo_id: row.geo_id})
            SET
                n.name                    = row.name,
                n.full_name               = row.full_name,
                n.val                     = row.val,
                n.ton                     = row.ton,
                n.val_norm                = row.val_norm,
                n.ton_norm                = row.ton_norm,
                n.vulnerability_score     = row.vulnerability_score,
                n.rank                    = row.rank,
                n.risk_tier               = row.risk_tier,
                n.lat                     = row.lat,
                n.lon                     = row.lon,
                n.is_metro                = row.is_metro,
                n.nearest_seaport         = row.nearest_seaport,
                n.nearest_seaport_miles   = row.nearest_seaport_miles,
                n.nearest_port            = row.nearest_port,
                n.nearest_port_miles      = row.nearest_port_miles,
                n.num_commodities         = row.num_commodities,
                n.val_ton_ratio           = row.val_ton_ratio,
                n.dominant_commodity      = row.dominant_commodity
        """, batch=[{
            "geo_id"               : r["GEO_ID"],
            "name"                 : clean_name(r["NAME"]),
            "full_name"            : r["NAME"],
            "val"                  : float(r["VAL"]),
            "ton"                  : float(r["TON"]),
            "val_norm"             : float(r["val_norm"]),
            "ton_norm"             : float(r["ton_norm"]),
            "vulnerability_score"  : float(r["vulnerability_score"]),
            "rank"                 : int(r["rank"]),
            "risk_tier"            : str(r["tier"]),
            "lat"                  : float(r["lat"])  if pd.notna(r["lat"])  else None,
            "lon"                  : float(r["lon"])  if pd.notna(r["lon"])  else None,
            "is_metro"             : int(r["is_metro"]) if pd.notna(r.get("is_metro")) else 0,
            "nearest_seaport"      : str(r["nearest_seaport"]) if pd.notna(r.get("nearest_seaport")) else "",
            "nearest_seaport_miles": float(r["nearest_seaport_miles"]) if pd.notna(r.get("nearest_seaport_miles")) else 9999.0,
            "nearest_port"         : str(r["nearest_port"]) if pd.notna(r.get("nearest_port")) else "",
            "nearest_port_miles"   : float(r["nearest_port_miles"]) if pd.notna(r.get("nearest_port_miles")) else 9999.0,
            "num_commodities"      : int(r["num_commodities"]) if pd.notna(r.get("num_commodities")) else 0,
            "val_ton_ratio"        : float(r["val_ton_ratio"]) if pd.notna(r.get("val_ton_ratio")) else 0.0,
            "dominant_commodity"   : str(r["dominant_commodity"]) if pd.notna(r.get("dominant_commodity")) else "",
        } for _, r in pd.DataFrame(batch).iterrows()])

    # -- Load edges in batches --
    print(f"      Loading {len(edges):,} CORRIDOR relationships …")
    edge_list = edges.to_dict("records")
    for i in range(0, len(edge_list), batch_size):
        batch = edge_list[i : i + batch_size]
        tx_session.run("""
            UNWIND $batch AS row
            MATCH (a:CFSArea {geo_id: row.origin_id})
            MATCH (b:CFSArea {geo_id: row.dest_id})
            MERGE (a)-[r:CORRIDOR]-(b)
            SET
                r.distance_miles = row.distance_miles,
                r.gravity_norm   = row.gravity_norm,
                r.gravity_rank   = row.gravity_rank,
                r.origin_name    = row.origin_name,
                r.dest_name      = row.dest_name
        """, batch=batch)

with driver.session() as session:
    load_graph(session, node_df.to_dict("records"), edge_df)

driver.close()
print("      ✅ Graph loaded successfully")

# ---------------------------------------------------------------------------
# 5. SUMMARY + USEFUL QUERIES
# ---------------------------------------------------------------------------
print("\n[5/5] Done!\n")
print("=" * 68)
print("  GRAPH LOADED INTO NEO4J")
print(f"  Nodes : {len(node_df)} CFSArea nodes")
print(f"  Edges : {len(edge_df):,} CORRIDOR relationships (K={K_NEIGHBORS} NN)")
print()
print("  OPEN NEO4J BROWSER → http://localhost:7474")
print()
print("  USEFUL CYPHER QUERIES TO TRY:")
print("""
  -- 1. View all HIGH-tier nodes
  MATCH (n:CFSArea {risk_tier: 'HIGH'}) RETURN n

  -- 2. Explore neighbours of any city
  MATCH (a:CFSArea)-[r:CORRIDOR]-(b:CFSArea)
  WHERE a.name CONTAINS 'Los Angeles'
  RETURN a, r, b

  -- 3. Top 10 highest gravity corridors
  MATCH (a:CFSArea)-[r:CORRIDOR]-(b:CFSArea)
  WHERE r.gravity_norm > 0
  RETURN a.name, b.name, r.distance_miles, r.gravity_norm
  ORDER BY r.gravity_norm DESC LIMIT 10

  -- 4. Shortest path between two cities
  MATCH p = shortestPath(
    (a:CFSArea)-[:CORRIDOR*]-(b:CFSArea)
  )
  WHERE a.name CONTAINS 'Los Angeles'
  AND   b.name CONTAINS 'New York'
  RETURN p

  -- 5. Most vulnerable metro areas
  MATCH (n:CFSArea {is_metro: 1})
  RETURN n.name, n.vulnerability_score, n.risk_tier
  ORDER BY n.vulnerability_score DESC LIMIT 10

  -- 6. Areas closest to seaports
  MATCH (n:CFSArea)
  RETURN n.name, n.nearest_seaport, n.nearest_seaport_miles
  ORDER BY n.nearest_seaport_miles ASC LIMIT 10
""")
print("=" * 68)

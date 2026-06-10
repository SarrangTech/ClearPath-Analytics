"""
ClearPath Analytics — Supply Chain Vulnerability Dashboard
MISM 6214 · Team 1 · Shreya Pandey

Run:  streamlit run dashboard.py
"""

import os
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClearPath Analytics",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #4a90d9;
    }
    .high-card  { border-left-color: #d62728; }
    .med-card   { border-left-color: #ff7f0e; }
    .low-card   { border-left-color: #2ca02c; }
    h1, h2, h3 { color: #e0e0e0; }

</style>
""", unsafe_allow_html=True)

# ── Data paths ────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
FEAT_FILE    = os.path.join(BASE, "supply_chain_data", "05_features", "features_master.csv")
CENT_FILE    = os.path.join(BASE, "supply_chain_data", "05_features", "centroids.csv")
DIST_FILE    = os.path.join(BASE, "supply_chain_data", "05_features", "distance_matrix.csv")
GRAVITY_FILE = os.path.join(BASE, "supply_chain_data", "03_outputs",  "corridor_gravity_scores.csv")
RISK_FILE    = os.path.join(BASE, "supply_chain_data", "03_outputs",  "risk_tier_output.csv")
PORT_FILE    = os.path.join(BASE, "supply_chain_data", "05_features", "ntad_ports.csv")
SIM_FILE     = os.path.join(BASE, "supply_chain_data", "06_network_outputs", "failure_simulation_results.csv")
DETAIL_FILE  = os.path.join(BASE, "supply_chain_data", "06_network_outputs", "05_rerouting_detail.csv")
FREIGHT_FILE = os.path.join(BASE, "supply_chain_data", "01_api_data", "cfs_freight_area.csv")
HAZMAT_FILE  = os.path.join(BASE, "supply_chain_data", "01_api_data", "cfs_hazmat.csv")
TEMP_FILE    = os.path.join(BASE, "supply_chain_data", "01_api_data", "cfs_temp_controlled.csv")
EXPORT_FILE  = os.path.join(BASE, "supply_chain_data", "01_api_data", "cfs_exports.csv")

K_NEIGHBORS       = 8
COST_PER_TON_MILE = 0.08
TIER_COLOR        = {"HIGH": "#d62728", "MEDIUM": "#ff7f0e", "LOW": "#6baed6"}

# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    feat_df    = pd.read_csv(FEAT_FILE)
    cent_df    = pd.read_csv(CENT_FILE)
    dist_df    = pd.read_csv(DIST_FILE)
    gravity_df = pd.read_csv(GRAVITY_FILE)
    risk_df    = pd.read_csv(RISK_FILE)
    port_df    = pd.read_csv(PORT_FILE)
    sim_df     = pd.read_csv(SIM_FILE)
    detail_df  = pd.read_csv(DETAIL_FILE) if os.path.exists(DETAIL_FILE) else pd.DataFrame()

    # New data sources
    freight_df = pd.read_csv(FREIGHT_FILE) if os.path.exists(FREIGHT_FILE) else pd.DataFrame()
    hazmat_df  = pd.read_csv(HAZMAT_FILE)  if os.path.exists(HAZMAT_FILE)  else pd.DataFrame()
    temp_df    = pd.read_csv(TEMP_FILE)    if os.path.exists(TEMP_FILE)    else pd.DataFrame()
    export_df  = pd.read_csv(EXPORT_FILE)  if os.path.exists(EXPORT_FILE)  else pd.DataFrame()

    def short_name(n):
        n = n.split(";")[0].strip()
        n = n.replace(" CFS Area", "").replace(" (part)", "")
        return (n[:40] + "…") if len(n) > 40 else n

    TIER_MAP = {"High": "HIGH", "Medium": "MEDIUM", "Low": "LOW"}
    risk_df["tier"] = risk_df["risk_tier"].map(TIER_MAP)

    node_meta = (
        feat_df
        .merge(cent_df[["GEO_ID", "INTPTLAT", "INTPTLON"]], on="GEO_ID", how="left",
               suffixes=("", "_cent"))
        .merge(risk_df[["GEO_ID", "tier"]], on="GEO_ID", how="left")
    )
    if "INTPTLAT_cent" in node_meta.columns:
        node_meta["lat"] = node_meta["INTPTLAT_cent"].fillna(node_meta["INTPTLAT"])
        node_meta["lon"] = node_meta["INTPTLON_cent"].fillna(node_meta["INTPTLON"])
    else:
        node_meta["lat"] = node_meta["INTPTLAT"]
        node_meta["lon"] = node_meta["INTPTLON"]

    p90 = node_meta["vulnerability_score"].quantile(0.90)
    p75 = node_meta["vulnerability_score"].quantile(0.75)
    node_meta["tier"] = node_meta["tier"].fillna(
        node_meta["vulnerability_score"].apply(
            lambda s: "HIGH" if s >= p90 else ("MEDIUM" if s >= p75 else "LOW")
        )
    )
    node_meta["short_name"] = node_meta["NAME"].apply(short_name)
    node_meta["color"]      = node_meta["tier"].map(TIER_COLOR)

    return feat_df, cent_df, dist_df, gravity_df, risk_df, port_df, sim_df, detail_df, node_meta, freight_df, hazmat_df, temp_df, export_df


@st.cache_data
def build_graph(dist_df, node_meta, gravity_df):
    gravity_lookup = (
        gravity_df.set_index(["GEO_ID_origin", "GEO_ID_dest"])["gravity_norm"]
        .to_dict()
    )
    G = nx.Graph()

    for _, r in node_meta.iterrows():
        G.add_node(
            r["GEO_ID"],
            name       = r["short_name"],
            lat        = r["lat"],
            lon        = r["lon"],
            vuln_score = r["vulnerability_score"],
            tier       = r["tier"],
            VAL        = r["VAL"],
            TON        = r["TON"],
        )

    dist_pos    = dist_df[dist_df["distance_miles"] > 0].copy()
    edges_added = set()
    for origin_id, grp in dist_pos.groupby("GEO_ID_origin"):
        for _, row in grp.nsmallest(K_NEIGHBORS, "distance_miles").iterrows():
            u, v = row["GEO_ID_origin"], row["GEO_ID_dest"]
            key  = tuple(sorted([u, v]))
            if key in edges_added:
                continue
            grav = gravity_lookup.get((u, v), gravity_lookup.get((v, u), 0.0))
            G.add_edge(u, v, distance=row["distance_miles"],
                       weight=row["distance_miles"], gravity_norm=grav)
            edges_added.add(key)

    if not nx.is_connected(G):
        comps     = list(nx.connected_components(G))
        main_comp = max(comps, key=len)
        for comp in comps:
            if comp == main_comp:
                continue
            for node in comp:
                best = dist_pos[
                    (dist_pos["GEO_ID_origin"] == node) &
                    (dist_pos["GEO_ID_dest"].isin(main_comp))
                ].nsmallest(1, "distance_miles")
                if not best.empty:
                    r    = best.iloc[0]
                    u, v = r["GEO_ID_origin"], r["GEO_ID_dest"]
                    grav = gravity_lookup.get((u, v), gravity_lookup.get((v, u), 0.0))
                    G.add_edge(u, v, distance=r["distance_miles"],
                               weight=r["distance_miles"], gravity_norm=grav)
    return G


# ── Load everything ───────────────────────────────────────────────────────────
feat_df, cent_df, dist_df, gravity_df, risk_df, port_df, sim_df, detail_df, node_meta, freight_df, hazmat_df, temp_df, export_df = load_data()
G = build_graph(dist_df, node_meta, gravity_df)

high_df   = node_meta[node_meta["tier"] == "HIGH"]
medium_df = node_meta[node_meta["tier"] == "MEDIUM"]
low_df    = node_meta[node_meta["tier"] == "LOW"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚚 ClearPath Analytics")
    st.markdown(
        "<span style='color:#aaa;font-size:13px;'>"
        "Supply Chain Vulnerability Analysis<br>"
        "MISM 6214 · Team 1 · Shreya Pandey"
        "</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Overview",
         "🗺️ Network Map",
         "💥 Failure Simulation",
         "🔗 Gravity Corridors",
         "🔍 Area Deep-Dive",
         "📦 Commodity Risk"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**📦 Dataset**")
    st.caption("2022 US Census Commodity Flow Survey · 134 CFS Areas")
    st.markdown("**🔬 Methodology**")
    st.caption("K=8 Nearest-Neighbour Graph · RF Classifier (90.3% LOOCV) · Dijkstra Rerouting")
    st.markdown("**💵 Cost Assumption**")
    st.caption("$0.08/ton-mile (BTS standard rate)")
    st.divider()
    st.caption("Use the menu above to explore the analysis.")


# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================
if page == "📊 Overview":
    st.title("📊 Supply Chain Vulnerability Overview")
    st.caption("134 CFS areas · 2022 US Census Commodity Flow Survey · RF Classifier risk tiers")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total CFS Areas", "134")
    with col2:
        st.metric("HIGH Risk Areas", len(high_df),
                  delta="Top 10% by vulnerability", delta_color="inverse")
    with col3:
        total_val = node_meta["VAL"].sum() / 1e6
        st.metric("Total Freight Value", f"${total_val:,.1f}T")
    with col4:
        total_ton = node_meta["TON"].sum() / 1e6
        st.metric("Total Tonnage", f"{total_ton:,.1f}B tons")

    st.divider()

    # ── Tier filter for cross-filtering ────────────────────────────────────
    selected_tiers = st.multiselect(
        "Filter by Risk Tier", ["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
    )
    filtered_df = node_meta[node_meta["tier"].isin(selected_tiers)]

    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.subheader("Vulnerability Score by Area")
        plot_df = filtered_df.sort_values("vulnerability_score", ascending=False).head(30).copy()
        fig = px.bar(
            plot_df,
            x="vulnerability_score",
            y="short_name",
            orientation="h",
            color="tier",
            color_discrete_map=TIER_COLOR,
            labels={"vulnerability_score": "Vulnerability Score", "short_name": ""},
            hover_data={"VAL": True, "TON": True, "tier": True},
            category_orders={"tier": ["HIGH", "MEDIUM", "LOW"]},
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(
            template="plotly_dark",
            yaxis={"categoryorder": "total ascending"},
            legend_title="Risk Tier",
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            bargap=0.25,
        )
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.subheader("Risk Tier Distribution")
        tier_counts = filtered_df["tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        # Horizontal bar instead of pie — more readable
        tier_order = ["HIGH", "MEDIUM", "LOW"]
        tier_counts["Tier"] = pd.Categorical(tier_counts["Tier"], categories=tier_order, ordered=True)
        tier_counts = tier_counts.sort_values("Tier")
        fig2 = go.Figure()
        for _, r in tier_counts.iterrows():
            fig2.add_trace(go.Bar(
                x=[r["Count"]],
                y=[r["Tier"]],
                orientation="h",
                name=r["Tier"],
                marker_color=TIER_COLOR.get(r["Tier"], "#aaa"),
                text=[f'{r["Count"]} areas ({r["Count"]/len(filtered_df)*100:.1f}%)'],
                textposition="inside",
                insidetextanchor="middle",
            ))
        fig2.update_layout(
            template="plotly_dark", height=160,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False, barmode="stack",
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, width="stretch")

        st.subheader("Value vs Tonnage")
        fig3 = px.scatter(
            filtered_df,
            x="VAL",
            y="TON",
            color="tier",
            color_discrete_map=TIER_COLOR,
            hover_name="short_name",
            size="vulnerability_score",
            size_max=22,
            labels={"VAL": "Freight Value ($K)", "TON": "Tonnage (K tons)"},
            category_orders={"tier": ["HIGH", "MEDIUM", "LOW"]},
        )
        fig3.update_layout(
            template="plotly_dark", height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig3, width="stretch")

    st.divider()
    st.subheader("All CFS Areas — Full Data Table")
    display_cols = ["short_name", "tier", "vulnerability_score", "VAL", "TON",
                    "nearest_seaport", "nearest_seaport_miles", "is_metro"]
    available = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(
        filtered_df[available].sort_values("vulnerability_score", ascending=False)
        .rename(columns={"short_name": "Area", "tier": "Risk Tier",
                         "vulnerability_score": "Vuln Score",
                         "VAL": "Value ($K)", "TON": "Tonnage (K tons)",
                         "nearest_seaport": "Nearest Seaport",
                         "nearest_seaport_miles": "Seaport Miles",
                         "is_metro": "Metro?"}),
        width="stretch",
        height=320,
    )


# =============================================================================
# PAGE 2 — NETWORK MAP
# =============================================================================
elif page == "🗺️ Network Map":
    st.title("🗺️ Freight Network Map")
    st.caption("K=8 nearest-neighbour graph · 134 nodes · edges weighted by distance and gravity")

    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
    with col_ctrl1:
        show_tiers = st.multiselect(
            "Show tiers", ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"])
    with col_ctrl2:
        show_edges = st.checkbox("Show edges", value=True)
    with col_ctrl3:
        high_only_edges = st.checkbox("HIGH-adjacent edges only", value=False)
    with col_ctrl4:
        gravity_threshold = st.slider("Min gravity for edges", 0.0, 0.5, 0.0, 0.02)

    # Filter nodes
    visible_nodes = node_meta[node_meta["tier"].isin(show_tiers)]
    visible_ids   = set(visible_nodes["GEO_ID"])

    # Build edge traces
    edge_traces = []
    if show_edges:
        high_ids = set(high_df["GEO_ID"])
        for u, v, d in G.edges(data=True):
            if u not in visible_ids or v not in visible_ids:
                continue
            if high_only_edges and u not in high_ids and v not in high_ids:
                continue
            u_d = G.nodes[u]; v_d = G.nodes[v]
            u_lat, u_lon = u_d.get("lat"), u_d.get("lon")
            v_lat, v_lon = v_d.get("lat"), v_d.get("lon")
            if any(x is None or (isinstance(x, float) and np.isnan(x))
                   for x in [u_lat, u_lon, v_lat, v_lon]):
                continue
            gnorm = d.get("gravity_norm", 0)
            if gnorm < gravity_threshold:
                continue
            # Color edges: blue (low gravity) → orange → red (high gravity)
            er = int(60 + 195 * gnorm)
            eg = int(100 - 60 * gnorm)
            eb = int(200 - 180 * gnorm)
            edge_traces.append(go.Scattergeo(
                lon=[u_lon, v_lon, None],
                lat=[u_lat, v_lat, None],
                mode="lines",
                line=dict(width=1.0 + 2.5 * gnorm, color=f"rgba({er},{eg},{eb},0.85)"),
                opacity=0.55 + 0.4 * gnorm,
                hoverinfo="none",
                showlegend=False,
            ))

    # Node traces per tier
    node_traces = []
    for tier in ["LOW", "MEDIUM", "HIGH"]:
        if tier not in show_tiers:
            continue
        sub = visible_nodes[visible_nodes["tier"] == tier]
        node_traces.append(go.Scattergeo(
            lon=sub["lon"],
            lat=sub["lat"],
            mode="markers",
            name=tier,
            marker=dict(
                size=sub["vulnerability_score"].apply(lambda s: 6 + s * 22),
                color=TIER_COLOR[tier],
                opacity=0.88,
                line=dict(width=0.5, color="white"),
            ),
            text=sub["short_name"],
            customdata=sub[["vulnerability_score", "VAL", "TON"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Tier: " + tier + "<br>"
                "Vuln Score: %{customdata[0]:.4f}<br>"
                "Value: $%{customdata[1]:,.0f}K<br>"
                "Tonnage: %{customdata[2]:,.0f}K tons"
                "<extra></extra>"
            ),
        ))

    # Port markers
    port_trace = go.Scattergeo(
        lon=port_df["lon"],
        lat=port_df["lat"],
        mode="markers",
        name="Ports / Hubs",
        marker=dict(size=8, symbol="diamond", color="#00d4aa",
                    line=dict(width=1, color="white")),
        text=port_df["port_name"],
        hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
        customdata=port_df["type"],
    )

    fig = go.Figure(data=edge_traces + node_traces + [port_trace])
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text="US Freight Network — K=8 Nearest Neighbour Graph",
            font=dict(size=15), x=0.01,
        ),
        geo=dict(
            scope="north america",
            projection_type="albers usa",
            showland=True, landcolor="#1a1a2e",
            showocean=True, oceancolor="#0d1117",
            showlakes=True, lakecolor="#0d1117",
            showcoastlines=True, coastlinecolor="#333",
            showcountries=True, countrycolor="#333",
            center=dict(lat=38, lon=-97),
        ),
        legend=dict(
            bgcolor="rgba(20,20,40,0.85)",
            bordercolor="#4a90d9",
            borderwidth=1,
            font=dict(color="white"),
            title=dict(text="Risk Tier", font=dict(color="white")),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )
    st.plotly_chart(fig, width="stretch")

    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    col_i1.info(f"**{len(visible_nodes)}** nodes visible")
    col_i2.info(f"**{G.number_of_edges()}** K-NN edges total")
    col_i3.info("**Node size** ∝ vulnerability score")
    col_i4.info("**🟢 Teal diamonds** = ports / intermodal hubs")

    st.caption(
        "Edge color and thickness ∝ gravity score — blue=low gravity, orange=medium, red=high. "
        "Use the gravity slider to reveal only the strongest corridors."
    )


# =============================================================================
# PAGE 3 — FAILURE SIMULATION
# =============================================================================
elif page == "💥 Failure Simulation":
    st.title("💥 Failure Simulation Results")
    st.caption(
        "Each HIGH-tier area is removed from the K=8 NN graph. "
        "OD pairs that lose their shortest path are rerouted via Dijkstra. "
        "Cost = extra miles × tonnage × $0.08/ton-mile."
    )

    if sim_df.empty:
        st.warning("Run `python scripts/network_rerouting.py` first to generate simulation results.")
        st.stop()

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Areas Simulated", len(sim_df))
    c2.metric("Areas Causing Rerouting", int((sim_df["affected_od_pairs"] > 0).sum()))
    c3.metric("Total OD Pairs Disrupted", f"{int(sim_df['affected_od_pairs'].sum()):,}")
    c4.metric("Aggregate Rerouting Cost", f"${sim_df['rerouting_cost_B'].sum():,.2f}B")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Rerouting Cost by Failed Area")
        plot_sim = sim_df.sort_values("rerouting_cost_B", ascending=True).copy()
        plot_sim["label"] = plot_sim["rerouting_cost_B"].apply(
            lambda x: f"${x:.2f}B" if x > 0 else "—")
        fig_cost = px.bar(
            plot_sim,
            x="rerouting_cost_B",
            y="failed_area",
            orientation="h",
            color="rerouting_cost_B",
            color_continuous_scale=["#4a90d9", "#ff7f0e", "#d62728"],
            labels={"rerouting_cost_B": "Rerouting Cost ($B)", "failed_area": "CFS Area"},
            hover_data={"affected_od_pairs": True, "extra_miles_total": True, "vuln_score": True},
            text="label",
            title="Economic Cost of Single-Node Failure (sorted by cost)",
        )
        fig_cost.update_traces(textposition="outside", cliponaxis=False)
        fig_cost.update_layout(
            template="plotly_dark",
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            height=500,
            margin=dict(l=10, r=140, t=50, b=10),
        )
        st.plotly_chart(fig_cost, width="stretch")

    with col_b:
        st.subheader("Number of OD Pairs Disrupted per Failure")
        plot_od = sim_df.sort_values("affected_od_pairs", ascending=True).copy()
        fig_od = px.bar(
            plot_od,
            x="affected_od_pairs",
            y="failed_area",
            orientation="h",
            color="affected_od_pairs",
            color_continuous_scale=["#4a90d9", "#ff7f0e", "#d62728"],
            labels={"affected_od_pairs": "OD Pairs Disrupted", "failed_area": "CFS Area"},
            text="affected_od_pairs",
            title="Origin-Destination Pairs Forced to Reroute",
        )
        fig_od.update_traces(textposition="outside")
        fig_od.update_layout(
            template="plotly_dark",
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            height=500,
            margin=dict(l=10, r=60, t=50, b=10),
        )
        st.plotly_chart(fig_od, width="stretch")

    st.divider()

    st.subheader("Vulnerability Score vs Rerouting Cost")
    st.caption(
        "Key insight: network position (structural centrality) matters more than vulnerability score alone. "
        "Illinois Remainder has a low score but the highest rerouting cost."
    )
    fig_scatter = px.scatter(
        sim_df,
        x="vuln_score",
        y="rerouting_cost_B",
        size="affected_od_pairs",
        size_max=40,
        text="failed_area",
        color="rerouting_cost_B",
        color_continuous_scale=["#4a90d9", "#ff7f0e", "#d62728"],
        labels={
            "vuln_score": "Vulnerability Score (RF Classifier)",
            "rerouting_cost_B": "Rerouting Cost ($B)",
            "affected_od_pairs": "OD Pairs Disrupted",
        },
        title="Vulnerability Score vs Economic Disruption Cost — bubble size = OD pairs disrupted",
    )
    fig_scatter.update_traces(textposition="top center", textfont_size=9)
    fig_scatter.update_layout(
        template="plotly_dark", height=440,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_scatter, width="stretch")

    # ── Rerouting detail map ─────────────────────────────────────────────────
    if not detail_df.empty:
        st.divider()
        st.subheader("🗺️ Rerouting Impact Map — Top Disrupted Routes")
        st.caption(
            "Select a failed area to see which OD pairs were rerouted and by how many extra miles."
        )

        area_choice = st.selectbox(
            "Select a failed area to inspect",
            options=sim_df.sort_values("rerouting_cost_B", ascending=False)["failed_area"].tolist(),
        )

        sub_detail = detail_df[detail_df["failed_area"] == area_choice].copy()
        sub_detail = sub_detail[sub_detail["extra_miles"] > 0].sort_values("extra_miles", ascending=False)

        if not sub_detail.empty:
            col_map, col_tbl = st.columns([1.4, 1])

            with col_map:
                # Build origin/destination lat-lon from node_meta
                name_to_coords = dict(zip(node_meta["short_name"],
                                          zip(node_meta["lat"], node_meta["lon"])))

                top_routes = sub_detail.head(30)
                route_traces = []

                for _, row in top_routes.iterrows():
                    o_name = str(row["origin"])[:40]
                    d_name = str(row["destination"])[:40]

                    # fuzzy match against short_name
                    o_match = [k for k in name_to_coords if o_name[:15] in k]
                    d_match = [k for k in name_to_coords if d_name[:15] in k]

                    if not o_match or not d_match:
                        continue

                    o_lat, o_lon = name_to_coords[o_match[0]]
                    d_lat, d_lon = name_to_coords[d_match[0]]

                    if any(pd.isna([o_lat, o_lon, d_lat, d_lon])):
                        continue

                    intensity = min(row["extra_miles"] / (sub_detail["extra_miles"].max() + 1), 1.0)
                    route_traces.append(go.Scattergeo(
                        lon=[o_lon, d_lon, None],
                        lat=[o_lat, d_lat, None],
                        mode="lines",
                        line=dict(
                            width=0.8 + 2.5 * intensity,
                            color=f"rgba({int(60 + 195*intensity)},68,{int(200-180*intensity)},0.6)",
                        ),
                        hoverinfo="none",
                        showlegend=False,
                    ))

                # All nodes
                node_tr = go.Scattergeo(
                    lon=node_meta["lon"], lat=node_meta["lat"],
                    mode="markers", name="CFS Areas",
                    marker=dict(
                        size=node_meta["vulnerability_score"].apply(lambda s: 5 + s * 14),
                        color=node_meta["color"], opacity=0.75,
                        line=dict(width=0.4, color="white"),
                    ),
                    text=node_meta["short_name"],
                    hovertemplate="<b>%{text}</b><extra></extra>",
                )

                fig_rmap = go.Figure(data=route_traces + [node_tr])
                fig_rmap.update_layout(
                    template="plotly_dark",
                    title=dict(text=f"Top 30 Rerouted OD Pairs — Failed: {area_choice[:35]}", font=dict(size=13), x=0.01),
                    geo=dict(
                        scope="north america",
                        projection_type="albers usa",
                        showland=True, landcolor="#1a1a2e",
                        showocean=True, oceancolor="#0d1117",
                        showcoastlines=True, coastlinecolor="#333",
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=440,
                )
                st.plotly_chart(fig_rmap, width="stretch")
                st.caption("Line thickness and color intensity ∝ extra miles added by the failure.")

            with col_tbl:
                st.markdown(f"**Top rerouted routes — {area_choice[:30]}**")
                show_cols = [c for c in ["origin", "destination", "extra_miles", "orig_miles", "new_miles"] if c in sub_detail.columns]
                st.dataframe(
                    sub_detail[show_cols].head(25).rename(columns={
                        "origin": "Origin", "destination": "Destination",
                        "extra_miles": "Extra Miles", "orig_miles": "Original Miles",
                        "new_miles": "New Miles",
                    }),
                    width="stretch", height=400,
                )

    st.divider()
    st.subheader("Full Simulation Results Table")
    st.dataframe(
        sim_df.sort_values("rerouting_cost_B", ascending=False)
        .rename(columns={
            "failed_area": "CFS Area", "vuln_score": "Vuln Score",
            "freight_value_B": "Value ($B)", "freight_ton_M": "Tonnage (M)",
            "affected_od_pairs": "OD Pairs", "extra_miles_total": "Extra Miles",
            "disconnected_pairs": "Disconnected", "rerouting_cost_B": "Cost ($B)",
        }),
        width="stretch",
        height=380,
    )


# =============================================================================
# PAGE 4 — GRAVITY CORRIDORS
# =============================================================================
elif page == "🔗 Gravity Corridors":
    st.title("🔗 Gravity-Weighted Corridor Analysis")
    st.caption(
        "Gravity score = (VAL_origin × VAL_dest) / distance² — "
        "measures the economic energy of each freight corridor. "
        "Higher gravity = more freight value exchanged per unit distance."
    )

    grav_df = gravity_df[gravity_df["GEO_ID_origin"] != gravity_df["GEO_ID_dest"]].copy()
    g_max = grav_df["gravity"].max()
    grav_df["gravity_norm"] = grav_df["gravity"] / g_max if g_max > 0 else grav_df["gravity_norm"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Directed Corridors", f"{len(grav_df):,}")
    c2.metric("Top Corridor Gravity", f"{grav_df['gravity'].max():,.0f}")
    c3.metric("Median Gravity", f"{grav_df['gravity'].median():,.0f}")

    st.divider()

    n_top = st.slider("Number of top corridors to display", 10, 50, 20)

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.subheader(f"Top {n_top} Corridors by Gravity Score")
        top_grav = grav_df.nlargest(n_top, "gravity").copy()
        top_grav["corridor"] = (
            top_grav["NAME_origin"].str.split(";").str[0].str[:24] + " ↔ " +
            top_grav["NAME_dest"].str.split(";").str[0].str[:24]
        )
        fig_grav = px.bar(
            top_grav.sort_values("gravity_norm"),
            x="gravity_norm",
            y="corridor",
            orientation="h",
            color="gravity_norm",
            color_continuous_scale=["#4a90d9", "#ff7f0e", "#d62728"],
            labels={"gravity_norm": "Gravity (normalised 0–1)", "corridor": "Corridor"},
            title=f"Top {n_top} Freight Corridors Ranked by Normalised Gravity Score",
        )
        fig_grav.update_layout(
            template="plotly_dark",
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            height=540,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_grav, width="stretch")

    with col_right:
        st.subheader("Gravity Score Distribution")
        fig_hist = px.histogram(
            grav_df[grav_df["gravity"] > 0],
            x="gravity_norm",
            nbins=50,
            color_discrete_sequence=["#4a90d9"],
            labels={"gravity_norm": "Normalised Gravity Score"},
            title="Most corridors have low gravity — a few dominate",
        )
        fig_hist.update_layout(
            template="plotly_dark", height=280,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_hist, width="stretch")

        st.subheader("Gravity vs Distance")
        sample = grav_df.sample(min(2000, len(grav_df)), random_state=42)
        fig_gd = px.scatter(
            sample,
            x="distance_miles",
            y="gravity_norm",
            opacity=0.4,
            color_discrete_sequence=["#4a90d9"],
            labels={"distance_miles": "Distance (miles)", "gravity_norm": "Gravity (normalised)"},
            title="Gravity decays with distance — short, high-value corridors dominate",
        )
        fig_gd.update_layout(
            template="plotly_dark", height=280,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_gd, width="stretch")

    st.divider()

    st.subheader("Corridor Map — Top Gravity Corridors")
    n_map = st.slider("Corridors to draw on map", 20, 200, 100)

    # Filter to corridors >= 200 miles so lines are visible on a national map
    map_df = grav_df[grav_df["distance_miles"] >= 200]
    top_map = map_df.nlargest(n_map, "gravity").copy()
    top_map = top_map.merge(
        cent_df.rename(columns={"GEO_ID": "GEO_ID_origin",
                                "INTPTLAT": "lat_o", "INTPTLON": "lon_o"}),
        on="GEO_ID_origin", how="left",
    ).merge(
        cent_df.rename(columns={"GEO_ID": "GEO_ID_dest",
                                "INTPTLAT": "lat_d", "INTPTLON": "lon_d"}),
        on="GEO_ID_dest", how="left",
    )

    edge_traces_g = []
    for _, row in top_map.iterrows():
        if any(pd.isna([row.lat_o, row.lon_o, row.lat_d, row.lon_d])):
            continue
        gnorm = row["gravity_norm"]
        edge_traces_g.append(go.Scattergeo(
            lon=[row.lon_o, row.lon_d, None],
            lat=[row.lat_o, row.lat_d, None],
            mode="lines",
            line=dict(width=1.0 + 3 * gnorm,
                      color=f"rgba({int(60+195*gnorm)},68,{int(200-200*gnorm)},0.85)"),
            opacity=0.55 + 0.4 * gnorm,
            hoverinfo="none",
            showlegend=False,
        ))

    node_trace_g = go.Scattergeo(
        lon=node_meta["lon"],
        lat=node_meta["lat"],
        mode="markers",
        name="CFS Areas",
        marker=dict(
            size=node_meta["vulnerability_score"].apply(lambda s: 5 + s * 18),
            color=node_meta["color"],
            opacity=0.85,
            line=dict(width=0.5, color="white"),
        ),
        text=node_meta["short_name"],
        hovertemplate="<b>%{text}</b><extra></extra>",
    )

    fig_map_g = go.Figure(data=edge_traces_g + [node_trace_g])
    fig_map_g.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"Top {n_map} Gravity Corridors — line thickness ∝ gravity score",
            font=dict(size=13), x=0.01,
        ),
        geo=dict(
            scope="north america",
            projection_type="albers usa",
            showland=True, landcolor="#1a1a2e",
            showocean=True, oceancolor="#0d1117",
            showcoastlines=True, coastlinecolor="#333",
            showcountries=True, countrycolor="#333",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=540,
    )
    st.plotly_chart(fig_map_g, width="stretch")
    st.caption("Node color = risk tier (red=HIGH, orange=MEDIUM, blue=LOW). Node size ∝ vulnerability score.")


# =============================================================================
# PAGE 5 — AREA DEEP-DIVE
# =============================================================================
elif page == "🔍 Area Deep-Dive":
    st.title("🔍 CFS Area Deep-Dive")
    st.caption("Select any CFS area to explore its network position, corridors, commodity profile, and risk.")

    area_list = sorted(node_meta["short_name"].tolist())
    selected  = st.selectbox("Select a CFS area", area_list)

    row    = node_meta[node_meta["short_name"] == selected].iloc[0]
    geo_id = row["GEO_ID"]

    tier_col = TIER_COLOR.get(row["tier"], "#6baed6")

    st.markdown(f"""
    <div style='background:#1e2130;border-radius:10px;padding:16px 24px;
                border-left:5px solid {tier_col};margin-bottom:16px;'>
      <h3 style='color:{tier_col};margin:0;'>{row['short_name']}</h3>
      <span style='color:#aaa;font-size:13px;'>{row['NAME']}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    </style>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Tier", row["tier"])
    c2.metric("Vulnerability Score", f"{row['vulnerability_score']:.4f}")
    c3.metric("Freight Value", f"${row['VAL']/1e3:,.1f}B")
    c4.metric("Tonnage", f"{row['TON']/1e6:.2f}B tons")

    # Extra commodity metrics — compute dominant commodity live to avoid 'All Commodities' artifact
    if "num_commodities" in row.index:
        cx1, cx2, cx3, _ = st.columns(4)
        cx1.metric("Commodities Handled", int(row["num_commodities"]) if pd.notna(row.get("num_commodities")) else "—")
        if not freight_df.empty:
            _area_comm = freight_df[
                (freight_df["GEO_ID"] == geo_id) &
                (freight_df["COMM"] != "0000") &
                (freight_df["COMM_LABEL"] != "All Commodities") &
                (freight_df["VAL"] > 0)
            ].groupby("COMM_LABEL")["VAL"].sum()
            _dominant = _area_comm.idxmax() if not _area_comm.empty else "—"
        else:
            _dominant = "—"
        cx2.metric("Dominant Commodity", str(_dominant)[:30])
        cx3.metric("Value/Ton Ratio", f"{row['val_ton_ratio']:.2f}" if pd.notna(row.get("val_ton_ratio")) else "—")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Network Neighbours (K=8)")
        neighbours = []
        if geo_id in G:
            for nbr in G.neighbors(geo_id):
                nd = G.nodes[nbr]
                ed = G.edges[geo_id, nbr]
                neighbours.append({
                    "Neighbour"    : nd["name"],
                    "Distance (mi)": round(ed["distance"], 1),
                    "Gravity Norm" : round(ed.get("gravity_norm", 0), 4),
                    "Tier"         : nd["tier"],
                })
        if neighbours:
            nbr_df = pd.DataFrame(neighbours).sort_values("Distance (mi)")
            fig_nbr = px.bar(
                nbr_df,
                x="Distance (mi)",
                y="Neighbour",
                orientation="h",
                color="Tier",
                color_discrete_map=TIER_COLOR,
                labels={"Distance (mi)": "Distance (miles)", "Neighbour": ""},
                title=f"K-NN Neighbours of {selected[:30]}",
            )
            fig_nbr.update_layout(
                template="plotly_dark", height=320,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_nbr, width="stretch")
            st.dataframe(nbr_df, width="stretch", height=220)
        else:
            st.info("Node not in graph.")

    with col_r:
        st.subheader("Port & Seaport Proximity")
        port_cols = ["nearest_port", "nearest_port_type", "nearest_port_miles",
                     "nearest_seaport", "nearest_seaport_miles"]
        avail = [c for c in port_cols if c in row.index and pd.notna(row[c])]
        for c in avail:
            st.metric(c.replace("_", " ").title(), row[c])

        if not sim_df.empty:
            sim_row = sim_df[sim_df["geo_id"] == geo_id]
            if not sim_row.empty:
                st.divider()
                st.subheader("Failure Simulation Result")
                s = sim_row.iloc[0]
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("OD Pairs Disrupted",  int(s["affected_od_pairs"]))
                sc2.metric("Extra Miles Added",   f"{s['extra_miles_total']:,.0f}")
                sc3.metric("Rerouting Cost",       f"${s['rerouting_cost_B']:.2f}B")
            else:
                st.info("This area was not in the HIGH-tier simulation set.")

    st.divider()

    st.subheader("Top Gravity Corridors Involving This Area")
    area_grav = gravity_df[
        (gravity_df["GEO_ID_origin"] == geo_id) |
        (gravity_df["GEO_ID_dest"]   == geo_id)
    ].copy()
    # Deduplicate: canonical pair key so A->B and B->A count as one corridor
    area_grav["_pair_key"] = area_grav.apply(
        lambda r: tuple(sorted([r["GEO_ID_origin"], r["GEO_ID_dest"]])), axis=1
    )
    area_grav = area_grav.sort_values("gravity", ascending=False).drop_duplicates("_pair_key")
    # Put selected area first for consistent label direction
    area_grav["corridor"] = area_grav.apply(
        lambda r: (
            r["NAME_origin"].split(";")[0][:28] + " \u2194 " + r["NAME_dest"].split(";")[0][:28]
            if r["GEO_ID_origin"] == geo_id
            else r["NAME_dest"].split(";")[0][:28] + " \u2194 " + r["NAME_origin"].split(";")[0][:28]
        ), axis=1
    )
    top_corridors = area_grav.nlargest(15, "gravity")
    _ac_sorted = top_corridors.sort_values("gravity_norm").copy()
    _ac_sorted["_bar_color"] = _ac_sorted["gravity_norm"].apply(
        lambda g: f"rgb({int(60+195*g)},{int(100-60*g)},{int(200-180*g)})"
    )
    fig_ac = go.Figure(go.Bar(
        x=_ac_sorted["gravity_norm"],
        y=_ac_sorted["corridor"],
        orientation="h",
        marker_color=_ac_sorted["_bar_color"].tolist(),
        hovertemplate="<b>%{y}</b><br>Gravity (norm): %{x:.4f}<extra></extra>",
    ))
    fig_ac.update_layout(
        template="plotly_dark",
        xaxis_title="Gravity (normalised)",
        yaxis={"categoryorder": "total ascending"},
        title=f"Top 15 Gravity Corridors for {selected[:30]}",
        height=400, margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_ac, width="stretch")

    if not detail_df.empty:
        sim_row = sim_df[sim_df["geo_id"] == geo_id] if not sim_df.empty else pd.DataFrame()
        if not sim_row.empty:
            area_detail = detail_df[detail_df["failed_area"] == sim_row.iloc[0]["failed_area"]]\
                .sort_values("extra_miles", ascending=False)
            if not area_detail.empty:
                st.divider()
                st.subheader("Most Impacted Routes if This Area Fails")
                st.dataframe(area_detail.head(20), width="stretch", height=280)

    # ── Per-area commodity breakdown ─────────────────────────────────────────
    if not freight_df.empty:
        st.divider()
        st.subheader("Commodity Breakdown for This Area")
        area_comm = freight_df[
            (freight_df["GEO_ID"] == geo_id) &
            (freight_df["COMM"] != "0000") &
            (freight_df["COMM_LABEL"] != "All Commodities")
        ].copy()
        area_comm = area_comm.groupby("COMM_LABEL")[["VAL", "TON"]].sum().reset_index()
        area_comm = area_comm[area_comm["VAL"] > 0].nlargest(15, "VAL")
        if not area_comm.empty:
            col_cv, col_ct = st.columns(2)
            with col_cv:
                fig_cv = px.bar(
                    area_comm.sort_values("VAL"),
                    x="VAL", y="COMM_LABEL", orientation="h",
                    color="VAL",
                    color_continuous_scale=["#4a90d9", "#ff7f0e", "#d62728"],
                    labels={"VAL": "Value ($K)", "COMM_LABEL": ""},
                    title=f"Top Commodities by Value — {selected[:28]}",
                )
                fig_cv.update_layout(template="plotly_dark", coloraxis_showscale=False,
                                     height=380, margin=dict(l=10, r=10, t=50, b=40),
                                     yaxis={"categoryorder": "total ascending"},
                                     xaxis={"automargin": True})
                st.plotly_chart(fig_cv, width="stretch")
            with col_ct:
                fig_ct = px.bar(
                    area_comm.sort_values("TON"),
                    x="TON", y="COMM_LABEL", orientation="h",
                    color="TON",
                    color_continuous_scale=["#4a90d9", "#ff7f0e", "#d62728"],
                    labels={"TON": "Tonnage (K)", "COMM_LABEL": ""},
                    title=f"Top Commodities by Tonnage — {selected[:28]}",
                )
                fig_ct.update_layout(template="plotly_dark", coloraxis_showscale=False,
                                     height=380, margin=dict(l=10, r=10, t=50, b=40),
                                     yaxis={"categoryorder": "total ascending"},
                                     xaxis={"automargin": True})
                st.plotly_chart(fig_ct, width="stretch")
        else:
            st.info("No commodity-level data available for this area.")


# =============================================================================
# PAGE 6 — COMMODITY RISK
# =============================================================================
elif page == "📦 Commodity Risk":
    st.title("📦 Commodity Risk Analysis")
    st.caption(
        "Breakdown of freight exposure by commodity type · hazardous materials · "
        "temperature-controlled freight · state-level export dependency"
    )

    # ── KPIs ─────────────────────────────────────────────────────────────────
    if not freight_df.empty:
        cfs_comm = freight_df[
            freight_df["GEO_ID"].str.startswith("E330", na=False) &
            (freight_df["COMM_LABEL"] != "All Commodities")
        ]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Commodity Categories", cfs_comm["COMM_LABEL"].nunique())
        c2.metric("CFS Areas with Data", cfs_comm["GEO_ID"].nunique())
        if not hazmat_df.empty:
            hz_total = hazmat_df[hazmat_df["COMM_LABEL"] == "All Commodities"]["VAL"].max()
            c3.metric("Total Hazmat Value", f"${hz_total/1e6:,.1f}T")
        if not temp_df.empty:
            tc_total = temp_df[temp_df["COMM_LABEL"] == "All Commodities"]["VAL"].max()
            c4.metric("Temp-Controlled Value", f"${tc_total/1e6:,.1f}T")

    st.divider()

    # ── TAB layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏷️ Commodity Exposure",
        "☣️ Hazardous Materials",
        "🌡️ Temperature-Controlled",
        "🚢 Export Dependency",
    ])

    # ── TAB 1: Commodity Exposure ─────────────────────────────────────────────
    with tab1:
        st.subheader("Top Commodities by Freight Value Across All CFS Areas")
        if not freight_df.empty:
            cfs_only = freight_df[
                freight_df["GEO_ID"].str.startswith("E330", na=False) &
                (freight_df["COMM_LABEL"] != "All Commodities") &
                (freight_df["COMM"] != "0000")
            ].copy()
            top_comm = (
                cfs_only.groupby("COMM_LABEL")[["VAL", "TON"]]
                .sum().reset_index()
                .nlargest(20, "VAL")
            )

            col_l, col_r = st.columns(2)
            with col_l:
                fig_tv = px.bar(
                    top_comm.sort_values("VAL"),
                    x="VAL", y="COMM_LABEL", orientation="h",
                    color="VAL",
                    color_continuous_scale=["#4a90d9", "#ff7f0e", "#d62728"],
                    labels={"VAL": "Total Freight Value ($K)", "COMM_LABEL": ""},
                    title="Top 20 Commodities by Total Freight Value",
                )
                fig_tv.update_layout(
                    template="plotly_dark", coloraxis_showscale=False,
                    yaxis={"categoryorder": "total ascending", "title": ""},
                    height=560, margin=dict(l=220, r=20, t=50, b=60),
                )
                fig_tv.update_xaxes(
                    title_text="Freight Value ($K)",
                    nticks=4,
                    tickformat="~s",
                    tickangle=0,
                )
                st.plotly_chart(fig_tv, use_container_width=True)

            with col_r:
                fig_tt = px.bar(
                    top_comm.sort_values("TON"),
                    x="TON", y="COMM_LABEL", orientation="h",
                    color="TON",
                    color_continuous_scale=["#4a90d9", "#2ca02c", "#d62728"],
                    labels={"TON": "Total Tonnage (K tons)", "COMM_LABEL": ""},
                    title="Top 20 Commodities by Total Tonnage (K tons)",
                )
                fig_tt.update_layout(
                    template="plotly_dark", coloraxis_showscale=False,
                    yaxis={"categoryorder": "total ascending", "title": ""},
                    height=560, margin=dict(l=220, r=20, t=50, b=60),
                )
                fig_tt.update_xaxes(
                    title_text="Tonnage (K tons)",
                    nticks=4,
                    tickformat="~s",
                    tickangle=0,
                )
                st.plotly_chart(fig_tt, use_container_width=True)

            st.divider()
            st.subheader("Commodity Concentration — Which Areas Dominate Each Commodity?")
            selected_comm = st.selectbox(
                "Select a commodity",
                sorted(cfs_only["COMM_LABEL"].unique().tolist()),
                index=sorted(cfs_only["COMM_LABEL"].unique().tolist()).index("Pharmaceutical products")
                      if "Pharmaceutical products" in cfs_only["COMM_LABEL"].unique() else 0,
            )
            comm_areas = (
                cfs_only[cfs_only["COMM_LABEL"] == selected_comm]
                .groupby("NAME")[["VAL", "TON"]].sum().reset_index()
                .nlargest(15, "VAL")
            )
            comm_areas["NAME"] = comm_areas["NAME"].str.split(";").str[0].str.replace(" CFS Area", "").str[:40]
            fig_ca = px.bar(
                comm_areas.sort_values("VAL"),
                x="VAL", y="NAME", orientation="h",
                color="VAL",
                color_continuous_scale=["#4a90d9", "#ff7f0e", "#d62728"],
                labels={"VAL": "Freight Value ($K)", "NAME": ""},
                title=f"Top 15 CFS Areas for: {selected_comm}",
                text="VAL",
            )
            fig_ca.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig_ca.update_layout(template="plotly_dark", coloraxis_showscale=False,
                                 yaxis={"categoryorder": "total ascending"},
                                 height=460, margin=dict(l=10, r=80, t=50, b=10))
            st.plotly_chart(fig_ca, width="stretch")

            # Also merge with risk tier to show exposure by tier
            st.divider()
            st.subheader("Commodity Exposure by Risk Tier")
            cfs_tier = cfs_only.merge(
                node_meta[["GEO_ID", "tier"]].drop_duplicates(),
                on="GEO_ID", how="left"
            )
            tier_comm = (
                cfs_tier.groupby(["COMM_LABEL", "tier"])["VAL"]
                .sum().reset_index()
            )
            top20 = tier_comm.groupby("COMM_LABEL")["VAL"].sum().nlargest(20).index
            tier_comm_top = tier_comm[tier_comm["COMM_LABEL"].isin(top20)]
            fig_tc = px.bar(
                tier_comm_top,
                x="VAL", y="COMM_LABEL",
                color="tier",
                color_discrete_map=TIER_COLOR,
                orientation="h",
                barmode="stack",
                labels={"VAL": "Freight Value ($K)", "COMM_LABEL": "Commodity", "tier": "Risk Tier"},
                title="Top 20 Commodities — freight value split by risk tier of originating area",
                category_orders={"tier": ["HIGH", "MEDIUM", "LOW"]},
            )
            fig_tc.update_layout(template="plotly_dark",
                                 yaxis={"categoryorder": "total ascending"},
                                 height=520, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_tc, width="stretch")
            st.caption("RED = value originating from HIGH-risk areas — these commodities face the greatest exposure to disruption.")

    # ── TAB 2: Hazmat ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("☣️ Hazardous Materials Freight — National Breakdown")
        st.caption("Source: 2022 CFS Hazmat supplement · national-level totals by commodity type")
        if not hazmat_df.empty:
            hz = hazmat_df[
                (hazmat_df["COMM_LABEL"] != "All Commodities") &
                (hazmat_df["VAL"] > 0)
            ].drop_duplicates(subset=["COMM_LABEL"]).copy()

            col_h1, col_h2 = st.columns(2)
            with col_h1:
                fig_hv = px.bar(
                    hz.sort_values("VAL"),
                    x="VAL", y="COMM_LABEL", orientation="h",
                    color="VAL",
                    color_continuous_scale=["#ff7f0e", "#d62728"],
                    labels={"VAL": "Freight Value ($K)", "COMM_LABEL": "Hazmat Commodity"},
                    title="Hazardous Materials by Freight Value",
                )
                fig_hv.update_layout(template="plotly_dark", coloraxis_showscale=False,
                                     yaxis={"categoryorder": "total ascending"},
                                     height=440, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_hv, width="stretch")

            with col_h2:
                fig_ht = px.bar(
                    hz[hz["TON"] > 0].sort_values("TON"),
                    x="TON", y="COMM_LABEL", orientation="h",
                    color="TON",
                    color_continuous_scale=["#ff7f0e", "#d62728"],
                    labels={"TON": "Tonnage (K tons)", "COMM_LABEL": "Hazmat Commodity"},
                    title="Hazardous Materials by Tonnage",
                )
                fig_ht.update_layout(template="plotly_dark", coloraxis_showscale=False,
                                     yaxis={"categoryorder": "total ascending"},
                                     height=440, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_ht, width="stretch")

            st.divider()
            all_hz = hazmat_df[hazmat_df["COMM_LABEL"] == "All Commodities"].drop_duplicates(subset=["VAL"])
            if not all_hz.empty:
                hz_val  = all_hz["VAL"].sum()
                hz_ton  = all_hz["TON"].sum()
                st.info(
                    f"**Total hazmat freight value: ${hz_val/1e6:,.2f}T** · "
                    f"**Total hazmat tonnage: {hz_ton/1e3:,.0f}M tons** · "
                    "Gasoline and fuel oils account for the vast majority of hazmat value and tonnage."
                )
            st.dataframe(
                hz[["COMM_LABEL", "VAL", "TON"]]
                .rename(columns={"COMM_LABEL": "Commodity", "VAL": "Value ($K)", "TON": "Tonnage (K)"})
                .sort_values("Value ($K)", ascending=False),
                width="stretch", height=320,
            )

    # ── TAB 3: Temperature-Controlled ─────────────────────────────────────────
    with tab3:
        st.subheader("🌡️ Temperature-Controlled Freight — National Breakdown")
        st.caption("Source: 2022 CFS Temperature-Controlled supplement · perishable and pharmaceutical freight")
        if not temp_df.empty:
            tc = temp_df[
                (temp_df["COMM_LABEL"] != "All Commodities") &
                (temp_df["VAL"] > 0)
            ].drop_duplicates(subset=["COMM_LABEL"]).nlargest(15, "VAL")

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                fig_tcv = px.bar(
                    tc.sort_values("VAL"),
                    x="VAL", y="COMM_LABEL", orientation="h",
                    color="VAL",
                    color_continuous_scale=["#4a90d9", "#2ca02c", "#ff7f0e"],
                    labels={"VAL": "Freight Value ($K)", "COMM_LABEL": "Commodity"},
                    title="Top 15 Temp-Controlled Commodities by Value",
                )
                fig_tcv.update_layout(template="plotly_dark", coloraxis_showscale=False,
                                      yaxis={"categoryorder": "total ascending"},
                                      height=440, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_tcv, width="stretch")

            with col_t2:
                fig_tct = px.bar(
                    tc[tc["TON"] > 0].sort_values("TON"),
                    x="TON", y="COMM_LABEL", orientation="h",
                    color="TON",
                    color_continuous_scale=["#4a90d9", "#2ca02c", "#ff7f0e"],
                    labels={"TON": "Tonnage (K tons)", "COMM_LABEL": "Commodity"},
                    title="Top 15 Temp-Controlled Commodities by Tonnage",
                )
                fig_tct.update_layout(template="plotly_dark", coloraxis_showscale=False,
                                      yaxis={"categoryorder": "total ascending"},
                                      height=440, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_tct, width="stretch")

            st.divider()
            tc_all = temp_df[temp_df["COMM_LABEL"] == "All Commodities"].drop_duplicates(subset=["VAL"])
            if not tc_all.empty:
                tc_val = tc_all["VAL"].sum()
                tc_ton = tc_all["TON"].sum()
                st.info(
                    f"**Total temp-controlled freight value: ${tc_val/1e6:,.2f}T** · "
                    f"**Total tonnage: {tc_ton/1e3:,.0f}M tons** · "
                    "Pharmaceuticals and meat/poultry are the highest-value perishable categories."
                )
            st.dataframe(
                temp_df[
                    (temp_df["COMM_LABEL"] != "All Commodities") & (temp_df["VAL"] > 0)
                ].drop_duplicates(subset=["COMM_LABEL"])[["COMM_LABEL", "VAL", "TON"]]
                .rename(columns={"COMM_LABEL": "Commodity", "VAL": "Value ($K)", "TON": "Tonnage (K)"})
                .sort_values("Value ($K)", ascending=False),
                width="stretch", height=360,
            )

    # ── TAB 4: Export Dependency ───────────────────────────────────────────────
    with tab4:
        st.subheader("🚢 State-Level Export Freight Dependency")
        st.caption("Source: 2022 CFS Exports supplement · shipments destined for export by state")
        if not export_df.empty:
            ex = export_df.drop_duplicates(subset=["NAME", "COMM"]).copy()
            ex_all = ex[ex["COMM_LABEL"] == "All Commodities"].sort_values("VAL", ascending=False)

            col_e1, col_e2 = st.columns(2)
            with col_e1:
                fig_ev = px.bar(
                    ex_all.sort_values("VAL").tail(20),
                    x="VAL", y="NAME", orientation="h",
                    color="VAL",
                    color_continuous_scale=["#4a90d9", "#ff7f0e", "#d62728"],
                    labels={"VAL": "Export Freight Value ($K)", "NAME": "State"},
                    title="Top 20 States by Export Freight Value",
                    text="VAL",
                )
                fig_ev.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
                fig_ev.update_layout(template="plotly_dark", coloraxis_showscale=False,
                                     yaxis={"categoryorder": "total ascending"},
                                     height=520, margin=dict(l=10, r=80, t=50, b=10))
                st.plotly_chart(fig_ev, width="stretch")

            with col_e2:
                fig_et = px.bar(
                    ex_all.sort_values("TON").tail(20),
                    x="TON", y="NAME", orientation="h",
                    color="TON",
                    color_continuous_scale=["#4a90d9", "#ff7f0e", "#d62728"],
                    labels={"TON": "Export Tonnage (K tons)", "NAME": "State"},
                    title="Top 20 States by Export Tonnage",
                )
                fig_et.update_layout(template="plotly_dark", coloraxis_showscale=False,
                                     yaxis={"categoryorder": "total ascending"},
                                     height=520, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_et, width="stretch")

            st.divider()
            st.subheader("Export Value Map by State")
            _STATE_ABBR = {
                'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
                'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
                'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
                'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
                'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
                'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
                'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
                'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
                'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
                'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
                'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
                'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
                'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC',
            }
            ex_all_map = ex_all.copy()
            ex_all_map['abbr'] = ex_all_map['NAME'].map(_STATE_ABBR)
            ex_all_map = ex_all_map.dropna(subset=['abbr'])
            fig_map = px.choropleth(
                ex_all_map,
                locations="abbr",
                locationmode="USA-states",
                color="VAL",
                color_continuous_scale=["#1a1a2e", "#4a90d9", "#ff7f0e", "#d62728"],
                scope="usa",
                labels={"VAL": "Export Value ($K)", "NAME": "State"},
                title="Export Freight Value by State — darker = higher export dependency",
            )
            fig_map.update_layout(
                template="plotly_dark", height=420,
                margin=dict(l=0, r=0, t=50, b=0),
                coloraxis_colorbar=dict(title="Value ($K)"),
            )
            st.plotly_chart(fig_map, width="stretch")
            st.caption(
                "California, Texas, and Washington dominate export freight — "
                "these states carry the highest exposure to port disruptions."
            )

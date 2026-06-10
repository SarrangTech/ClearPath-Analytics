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

    return feat_df, cent_df, dist_df, gravity_df, risk_df, port_df, sim_df, detail_df, node_meta


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
feat_df, cent_df, dist_df, gravity_df, risk_df, port_df, sim_df, detail_df, node_meta = load_data()
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
         "🔍 Area Deep-Dive"],
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
                line=dict(width=0.4 + 2.5 * gnorm, color=f"rgba({er},{eg},{eb},0.75)"),
                opacity=0.35 + 0.5 * gnorm,
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
        fig_cost.update_traces(textposition="outside")
        fig_cost.update_layout(
            template="plotly_dark",
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            height=500,
            margin=dict(l=10, r=80, t=50, b=10),
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

    grav_df = gravity_df.copy()

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
    n_map = st.slider("Corridors to draw on map", 20, 200, 50)

    top_map = grav_df.nlargest(n_map, "gravity").copy()
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
            line=dict(width=0.5 + 3 * gnorm,
                      color=f"rgba({int(60+195*gnorm)},68,{int(200-200*gnorm)},0.6)"),
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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Tier", row["tier"])
    c2.metric("Vulnerability Score", f"{row['vulnerability_score']:.4f}")
    c3.metric("Freight Value", f"${row['VAL']/1e3:,.1f}B")
    c4.metric("Tonnage", f"{row['TON']/1e6:.2f}B tons")

    # Extra commodity metrics if available
    if "num_commodities" in row.index and "dominant_commodity" in row.index:
        cx1, cx2, cx3 = st.columns(3)
        cx1.metric("Commodities Handled", int(row["num_commodities"]) if pd.notna(row.get("num_commodities")) else "—")
        cx2.metric("Dominant Commodity", str(row["dominant_commodity"])[:30] if pd.notna(row.get("dominant_commodity")) else "—")
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
    area_grav["corridor"] = (
        area_grav["NAME_origin"].str.split(";").str[0].str[:28] + " ↔ " +
        area_grav["NAME_dest"].str.split(";").str[0].str[:28]
    )
    top_corridors = area_grav.nlargest(15, "gravity")
    fig_ac = px.bar(
        top_corridors.sort_values("gravity_norm"),
        x="gravity_norm",
        y="corridor",
        orientation="h",
        color="gravity_norm",
        color_continuous_scale=["#4a90d9", "#ff7f0e", "#d62728"],
        labels={"gravity_norm": "Gravity (normalised)", "corridor": ""},
        title=f"Top 15 Gravity Corridors for {selected[:30]}",
    )
    fig_ac.update_layout(
        template="plotly_dark", coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
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

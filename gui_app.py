import json
import math
import os
import random
import time
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from route_utils import (
    load_cities_with_roads,
    road_pairs_to_canonical_edges,
    find_data_inconsistencies,
    validate_route,
    validate_corridor_path,
)
from genetic_algorithm import GeneticAlgorithm, CorridorPathGA
from visualization import plot_route, plot_fitness


def _on_route_highlight_change():
    st.session_state.evo_frame = 0


def _init_state():
    defaults = {
        "loaded_cities": None,
        "loaded_source": None,
        "pick_src": None,
        "pick_dst": None,
        "directions_result": None,
        "classic_result": None,
        "route_pick_idx": 0,
        "evo_play": False,
        "evo_frame": 0,
        "route_map_version": 0,
        "next_pick_role": "source",
        "loaded_road_edges": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _load_uploaded_csv(uploaded_file):
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name
    try:
        cities, road_pairs = load_cities_with_roads(temp_path)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    return cities, road_pairs


def _show_error_with_fix(title, fixes):
    st.error(title)
    st.markdown("**How to fix:**")
    for fix in fixes:
        st.write(f"- {fix}")


def _fix_for_inconsistency(issue):
    text = issue.lower()
    if "duplicate city name" in text:
        return "Rename or remove duplicate city names so each city name is unique."
    if "duplicate coordinates" in text:
        return "Keep only one city per exact coordinate pair or correct one location."
    if "no city data loaded" in text:
        return "Upload a valid CSV with at least one data row."
    return "Correct the listed issue in CSV and upload again."


def _fixes_for_csv_error(error_message):
    msg = error_message.lower()
    if "invalid row at line" in msg:
        return [
            "Ensure CSV includes all required columns in each row: City, Latitude, Longitude.",
            "Remove malformed or partially empty rows.",
        ]
    if "empty city name" in msg:
        return ["Fill the city name for that row; blank names are not allowed."]
    if "invalid numeric value" in msg:
        return ["Use numeric latitude/longitude values only, e.g., 33.6844, 73.0479."]
    if "road references unknown city" in msg:
        return [
            "Every From/To name in the roads section must exactly match a City name.",
            "Fix spelling or add the missing city row.",
        ]
    if "latitude out of range" in msg:
        return ["Latitude must be between -90 and 90."]
    if "longitude out of range" in msg:
        return ["Longitude must be between -180 and 180."]
    if "non-finite coordinates" in msg:
        return ["Replace NaN/inf with valid numeric coordinates."]
    if "empty or has no valid city data" in msg:
        return ["Add valid city rows to CSV before uploading."]
    return [
        "Check CSV headers and row format.",
        "Use UTF-8 CSV file with City, Latitude, Longitude columns.",
    ]


def _validate_cities_or_stop(cities):
    issues = find_data_inconsistencies(cities)
    if issues:
        st.error("Data consistency errors detected.")
        st.markdown("**Detected issues with targeted fixes:**")
        for idx, issue in enumerate(issues, start=1):
            st.write(f"{idx}. {issue}")
            st.write(f"   - Fix: {_fix_for_inconsistency(issue)}")
        st.stop()
    if len(cities) < 4:
        _show_error_with_fix(
            "At least 4 cities are required.",
            [
                "Add more cities to your CSV.",
                "Pick a larger built-in CSV from the data folder.",
            ],
        )
        st.stop()


def _directions_fitness_line_chart(fitness_history):
    """Plot GA distance metrics; handles legacy flat list (elite-only) format."""
    fh = fitness_history or []
    if not fh:
        return
    st.subheader("Distance vs generation")
    if isinstance(fh[0], dict):
        df = pd.DataFrame(fh)
        df.insert(0, "Generation", range(1, len(df) + 1))
        chart = df.set_index("Generation")[
            ["population_best_km", "elite_best_so_far_km", "population_mean_valid_km"]
        ].rename(
            columns={
                "population_best_km": "Population best (this gen)",
                "elite_best_so_far_km": "Elite best-so-far (running)",
                "population_mean_valid_km": "Mean valid route (population)",
            }
        )
        st.line_chart(chart)
        st.caption(
            "**Population best** is the shortest feasible route in the current generation (it should move). "
            "**Elite best-so-far** only updates when the GA beats its previous record—it can stay flat if the "
            "run never beats the best random initial chromosome. **Mean valid** averages all feasible individuals "
            "each generation."
        )
    else:
        st.line_chart(
            pd.DataFrame(
                {"Generation": range(1, len(fh) + 1), "Elite best-so-far (km)": fh}
            ).set_index("Generation")
        )
        st.caption("Legacy history (elite running best only). Run **Get Directions** again for full curves.")


def _select_routes_from_archive(archive, best_dist, tol_frac, min_r, max_r):
    """
    Pick distinct routes from a saved archive using tolerance vs best_dist.
    Relaxes tolerance progressively until at least min_r routes or a cap.
    archive: list of [path_indices, distance_km].
    """
    if best_dist <= 0 or not math.isfinite(best_dist):
        best_dist = 1e-9
    by_key = {}
    for item in archive:
        if not item or len(item) != 2:
            continue
        p, d = item[0], float(item[1])
        k = tuple(p)
        if k not in by_key or d < by_key[k]:
            by_key[k] = d

    tol = float(tol_frac)
    picked = []
    for _ in range(22):
        cap = best_dist * (1.0 + tol)
        picked = []
        seen = set()
        for k, d in sorted(by_key.items(), key=lambda x: x[1]):
            if d > cap:
                continue
            if k in seen:
                continue
            seen.add(k)
            picked.append((list(k), d))
            if len(picked) >= max_r:
                break
        if len(picked) >= min_r:
            return picked[:max_r]
        tol = min(tol * 1.2 + 0.025, 5.0)
    return picked[:max_r]


def _format_travel_time(distance_km, speed_kmh):
    """time (h) = distance (km) / speed (km/h)."""
    if speed_kmh <= 0:
        return "N/A"
    total_h = distance_km / speed_kmh
    whole = int(total_h)
    minutes = int(round((total_h - whole) * 60))
    if minutes >= 60:
        whole += 1
        minutes -= 60
    return f"{whole} h {minutes} min"


def _sort_data_csv_files(filenames):
    """Prefer cities_20.csv, then cities_25.csv, then cities.csv; then remaining sorted."""
    lower_index = {f.lower(): f for f in filenames}
    preferred = ["cities_20.csv", "cities_25.csv", "cities.csv"]
    ordered = []
    used_lower = set()
    for want in preferred:
        key = want.lower()
        if key in lower_index:
            ordered.append(lower_index[key])
            used_lower.add(key)
    rest = sorted(f for f in filenames if f.lower() not in used_lower)
    return ordered + rest


def _city_idx_from_point(pt):
    cd = pt.get("customdata")
    if cd is None:
        return None
    if isinstance(cd, (list, tuple)):
        return int(cd[0])
    return int(cd)


def _fig_road_network(cities, canonical_edges, title="Road network"):
    lons = [c.lon for c in cities]
    lats = [c.lat for c in cities]
    names = [c.name for c in cities]
    fig = go.Figure()
    if canonical_edges:
        edge_x, edge_y = [], []
        for i, j in canonical_edges:
            edge_x.extend([cities[i].lon, cities[j].lon, None])
            edge_y.extend([cities[i].lat, cities[j].lat, None])
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=2, color="#78909c"),
                hoverinfo="skip",
                name="Roads",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=lons,
            y=lats,
            mode="markers+text",
            text=names,
            textposition="top center",
            textfont=dict(size=8, color="#37474f"),
            marker=dict(size=8, color="#eceff1", line=dict(width=1, color="#546e7a")),
            hovertemplate="%{text}<extra></extra>",
            name="Cities",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=520,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        dragmode="zoom",
    )
    return fig


def _fig_ga_evolution_frame(
    cities,
    snapshot,
    road_edges=None,
    src_idx=None,
    dst_idx=None,
):
    """
    Visualize one GA generation: evaluated chromosomes (bright lines, drawn beneath elite so shared edges stay readable).

    Trace order matters in Plotly: roads → ghost elite → evaluated chromosomes → crisp elite → cities on top.
    """
    palette = (
        "#FFB74D",
        "#FF4081",
        "#26C6DA",
        "#EEFF41",
        "#E040FB",
        "#69F0AE",
        "#FF9100",
        "#448AFF",
        "#FFD740",
        "#FF5252",
    )
    fig = go.Figure()

    if road_edges:
        edge_x, edge_y = [], []
        for i, j in road_edges:
            edge_x.extend([cities[i].lon, cities[j].lon, None])
            edge_y.extend([cities[i].lat, cities[j].lat, None])
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1, color="rgba(176,190,197,0.45)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    samples = snapshot.get("population_samples") or []
    bp = snapshot.get("best_so_far_path")
    bd = snapshot.get("best_so_far_dist")

    bx, by = [], []
    if bp:
        bx = [cities[i].lon for i in bp]
        by = [cities[i].lat for i in bp]
        fig.add_trace(
            go.Scatter(
                x=bx,
                y=by,
                mode="lines",
                line=dict(width=18, color="rgba(25,118,210,0.10)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bx,
                y=by,
                mode="lines",
                line=dict(width=8, color="rgba(21,101,192,0.35)"),
                showlegend=False,
                hovertemplate=(
                    f"<b>Elite best-so-far</b><br>{bd:,.2f} km<extra></extra>"
                    if bd is not None
                    else "<b>Elite best-so-far</b><extra></extra>"
                ),
            )
        )

    for idx, sample in enumerate(samples):
        path, dist = sample[0], sample[1]
        xs = [cities[i].lon for i in path]
        ys = [cities[i].lat for i in path]
        col = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=4, color=col),
                opacity=0.95,
                showlegend=False,
                hovertemplate=(
                    f"<b>Chromosome #{idx + 1}</b> (evaluated this gen)<br>"
                    f"{dist:,.2f} km<extra></extra>"
                ),
            )
        )

    if bp:
        fig.add_trace(
            go.Scatter(
                x=bx,
                y=by,
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=12,
                    color="#1565C0",
                    line=dict(width=2, color="#ffffff"),
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>Elite route stops</b><br>{bd:,.2f} km<extra></extra>"
                    if bd is not None
                    else "<b>Elite route stops</b><extra></extra>"
                ),
            )
        )

    lons = [c.lon for c in cities]
    lats = [c.lat for c in cities]
    names = [c.name for c in cities]
    colors = []
    sizes = []
    for i in range(len(cities)):
        if src_idx is not None and i == src_idx:
            colors.append("#43A047")
            sizes.append(16)
        elif dst_idx is not None and i == dst_idx:
            colors.append("#FF5252")
            sizes.append(16)
        else:
            colors.append("#78909C")
            sizes.append(9)

    fig.add_trace(
        go.Scatter(
            x=lons,
            y=lats,
            mode="markers+text",
            text=names,
            textposition="top center",
            textfont=dict(size=8, color="rgba(38,50,56,0.92)"),
            marker=dict(
                color=colors,
                size=sizes,
                line=dict(width=2, color="rgba(255,255,255,0.9)"),
            ),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=44, r=44, t=28, b=44),
        title=dict(text=""),
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=520,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        dragmode="zoom",
    )
    return fig


def _fig_city_picker(cities, src, dst, road_edges=None):
    lons = [c.lon for c in cities]
    lats = [c.lat for c in cities]
    names = [c.name for c in cities]
    colors = []
    sizes = []
    for i in range(len(cities)):
        if src is not None and i == src:
            colors.append("#1b5e20")
            sizes.append(16)
        elif dst is not None and i == dst:
            colors.append("#b71c1c")
            sizes.append(16)
        else:
            colors.append("#607d8b")
            sizes.append(10)

    customdata = [[i] for i in range(len(cities))]
    fig = go.Figure()
    if road_edges:
        edge_x, edge_y = [], []
        for i, j in road_edges:
            edge_x.extend([cities[i].lon, cities[j].lon, None])
            edge_y.extend([cities[i].lat, cities[j].lat, None])
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1.5, color="#b0bec5"),
                hoverinfo="skip",
                name="Roads",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=lons,
            y=lats,
            mode="markers+text",
            text=names,
            textposition="top center",
            textfont=dict(size=9, color="#263238"),
            marker=dict(
                color=colors,
                size=sizes,
                line=dict(width=1.5, color="#263238"),
            ),
            customdata=customdata,
            hovertemplate="<b>%{text}</b><br>Click assigns %{customdata[0]}<extra></extra>",
            name="Cities",
        )
    )
    fig.update_layout(
        title="Pick source and destination on the map (click a marker)",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=520,
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        dragmode="zoom",
    )
    return fig


def _fig_routes(cities, routes_with_dist, selected_idx):
    """routes_with_dist: list of (route_indices, distance_km), best route first."""
    fig = go.Figure()
    lons = [c.lon for c in cities]
    lats = [c.lat for c in cities]
    names = [c.name for c in cities]

    fig.add_trace(
        go.Scatter(
            x=lons,
            y=lats,
            mode="markers+text",
            text=names,
            textposition="top center",
            textfont=dict(size=8, color="#37474f"),
            marker=dict(size=8, color="#eceff1", line=dict(width=1, color="#90a4ae")),
            hovertemplate="%{text}<extra></extra>",
            name="Cities",
        )
    )

    best_color = "#0d47a1"
    alt_color = "#64b5f6"
    alt_dim = "#b3e5fc"

    mids_lon = []
    mids_lat = []
    mids_rid = []
    hover_labels = []

    for ri, (route, dist) in enumerate(routes_with_dist):
        xs = [cities[i].lon for i in route]
        ys = [cities[i].lat for i in route]

        if ri == 0:
            col = best_color
            lw = 6 if selected_idx == 0 else 5
        else:
            col = alt_color
            lw = 2
            if ri == selected_idx:
                lw = 4
                col = "#1565c0"

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line=dict(width=lw, color=col),
                marker=dict(size=6, color=col),
                name=f"Route {ri + 1} ({dist:,.1f} km)",
                hovertemplate=f"<b>Route {ri + 1}</b><br>{dist:,.2f} km<extra></extra>",
            )
        )

        r_full = routes_with_dist[ri][0]
        for seg_i in range(len(r_full) - 1):
            a = r_full[seg_i]
            b = r_full[seg_i + 1]
            mids_lon.append((cities[a].lon + cities[b].lon) / 2)
            mids_lat.append((cities[a].lat + cities[b].lat) / 2)
            mids_rid.append(ri)
            hover_labels.append(f"Route {ri + 1}: pick this path ({dist:,.1f} km)")

    fig.add_trace(
        go.Scatter(
            x=mids_lon,
            y=mids_lat,
            mode="markers",
            marker=dict(size=14, color="rgba(255,255,255,0.35)", line=dict(width=1, color="#546e7a")),
            customdata=[[rid] for rid in mids_rid],
            hovertemplate="%{text}<extra></extra>",
            text=hover_labels,
            name="Select route",
        )
    )

    fig.update_layout(
        title="Routes (darkest blue = best GA tour; lighter blues = alternatives)",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        dragmode="zoom",
    )
    return fig


def _plotly_map_selection_points(map_key):
    rm = st.session_state.get(map_key)
    if not rm:
        return []
    sel = getattr(rm, "selection", None)
    if sel is None and isinstance(rm, dict):
        sel = rm.get("selection")
    if not sel:
        return []
    if isinstance(sel, dict):
        return sel.get("points") or []
    return getattr(sel, "points", None) or []


def _extract_route_click(selection_points):
    for pt in selection_points:
        cd = pt.get("customdata")
        if cd is None:
            continue
        if isinstance(cd, (list, tuple)) and len(cd) > 0:
            try:
                return int(cd[0])
            except (TypeError, ValueError):
                continue
    return None


@st.fragment(run_every=timedelta(milliseconds=520))
def directions_evolution_fragment():
    """Animate GA snapshots so processing traces mirror actual evaluated chromosomes."""
    dr = st.session_state.get("directions_result")
    cities = st.session_state.get("loaded_cities")
    if not dr or not cities or st.session_state.get("road_network_only"):
        return

    snaps = dr.get("evolution_snapshots") or []
    if not snaps:
        st.caption("No per-generation snapshots for this session.")
        return

    max_i = len(snaps) - 1
    if st.session_state.evo_frame > max_i:
        st.session_state.evo_frame = max_i
    if st.session_state.evo_play and max_i > 0:
        if st.session_state.evo_frame >= max_i:
            st.session_state.evo_play = False
        else:
            st.session_state.evo_frame += 1

    st.slider(
        "Replay frame (snapshot index)",
        min_value=0,
        max_value=max_i,
        key="evo_frame",
        help="Frames are sampled along the run; the algorithm generation number is shown below.",
    )
    fi = int(min(max(st.session_state.evo_frame, 0), max_i))
    snap = snaps[fi]
    gen_alg = snap.get("generation", fi)
    samples_vis = snap.get("population_samples") or []
    samp_dists = [float(s[1]) for s in samples_vis]
    mean_txt = ""
    if samp_dists:
        mean_txt = (
            f" &nbsp;•&nbsp; Avg distance among drawn chromosomes: **{sum(samp_dists) / len(samp_dists):,.1f} km**"
        )
    st.markdown(
        f"**Algorithm generation:** `{gen_alg}` &nbsp;•&nbsp; **Replay frame:** {fi + 1} / {len(snaps)}"
        f" &nbsp;•&nbsp; **Chromosomes drawn:** {len(samples_vis)}{mean_txt}"
    )
    st.caption(
        "**Bright colored routes** = feasible chromosomes from this generation after fitness evaluation "
        "(spread across fitness ranks so you can see search diversity). **Soft blue band + diamonds** = elite "
        "best-so-far carried forward. Roads stay in the background."
    )
    road_e = st.session_state.get("loaded_road_edges") or None

    evo_fig = _fig_ga_evolution_frame(
        cities,
        snap,
        road_edges=road_e,
        src_idx=dr.get("src"),
        dst_idx=dr.get("dst"),
    )
    st.plotly_chart(evo_fig, width="stretch")


def main():
    st.set_page_config(page_title="Route directions", layout="wide")
    _init_state()

    st.title("Route directions")
    st.caption(
        "AI2002 - Assignment 3 | **Directions** use a genetic algorithm only for routing (random feasible "
        "initialization + evolution). Corridor filtering limits hub candidates; **# ROADS** in CSV restricts legs "
        "to those edges; without roads, legs follow **k-nearest-neighbor** connectivity."
    )

    st.sidebar.header("Input Data")
    input_type = st.sidebar.radio(
        "Input Type",
        ["Upload CSV", "Path in data folder"],
        index=1,
    )

    if input_type == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                cities, road_pairs = _load_uploaded_csv(uploaded_file)
                edges = road_pairs_to_canonical_edges(cities, road_pairs)
                st.session_state.loaded_cities = cities
                st.session_state.loaded_road_edges = edges
                st.session_state.loaded_source = f"uploaded: {uploaded_file.name}"
                st.session_state.directions_result = None
                st.session_state.classic_result = None
                st.session_state.pick_src = None
                st.session_state.pick_dst = None
                st.session_state.next_pick_role = "source"
            except Exception as e:
                _show_error_with_fix(f"Failed to parse uploaded CSV: {e}", _fixes_for_csv_error(str(e)))
                st.stop()
    else:
        data_dir = "data"
        csv_files = []
        if os.path.isdir(data_dir):
            raw_csv = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
            csv_files = _sort_data_csv_files(raw_csv)

        if not csv_files:
            _show_error_with_fix(
                "No CSV files found in data folder.",
                [
                    "Copy CSV files into this app's data folder.",
                    "Or use Upload CSV instead.",
                ],
            )
            st.stop()

        selected_file = st.sidebar.selectbox("Select CSV from data folder", csv_files, index=0)
        if st.sidebar.button("Load selected CSV"):
            csv_path = os.path.join(data_dir, selected_file)
            try:
                cities, road_pairs = load_cities_with_roads(csv_path)
                edges = road_pairs_to_canonical_edges(cities, road_pairs)
                st.session_state.loaded_cities = cities
                st.session_state.loaded_road_edges = edges
                st.session_state.loaded_source = csv_path
                st.session_state.directions_result = None
                st.session_state.classic_result = None
                st.session_state.pick_src = None
                st.session_state.pick_dst = None
                st.session_state.next_pick_role = "source"
            except Exception as e:
                _show_error_with_fix(f"Failed to load CSV from path: {e}", _fixes_for_csv_error(str(e)))
                st.stop()

    st.sidebar.header("Directions (corridor GA)")
    corridor_ratio = st.sidebar.slider(
        "Corridor width (detour ratio vs direct)",
        min_value=1.05,
        max_value=2.85,
        value=1.42,
        step=0.03,
        help="Cities allowed as hops satisfy d(src,c)+d(c,dst) ≤ ratio × direct. Larger = more optional hubs.",
    )
    alt_tolerance_pct = st.sidebar.slider(
        "Route tolerance vs GA best (%)",
        min_value=1,
        max_value=35,
        value=12,
        help="Prefer alternatives within this band of the best distance; min/max counts apply after filtering.",
    )
    min_route_suggestions = st.sidebar.slider(
        "Min routes to suggest",
        min_value=1,
        max_value=12,
        value=2,
    )
    max_route_suggestions = st.sidebar.slider(
        "Max routes to suggest",
        min_value=min_route_suggestions,
        max_value=15,
        value=max(min_route_suggestions, 5),
    )

    has_roads = bool(st.session_state.get("loaded_road_edges"))
    st.sidebar.checkbox(
        "Road network only (hide route overlays)",
        key="road_network_only",
        disabled=not has_roads,
        help="When enabled, direction maps show only cities and road edges from the CSV.",
    )

    st.sidebar.header("GA parameters (classic closed tour only)")
    pop_size = st.sidebar.slider("Population size", min_value=20, max_value=500, value=120, step=10)
    mutation_rate = st.sidebar.slider("Mutation rate", min_value=0.001, max_value=1.0, value=0.015, step=0.001)
    generations = st.sidebar.slider("Generations", min_value=50, max_value=3000, value=500, step=50)
    tournament_size = st.sidebar.slider("Tournament size", min_value=2, max_value=50, value=5, step=1)
    elite_count = st.sidebar.slider("Elite count", min_value=1, max_value=20, value=2, step=1)
    seed_input = st.sidebar.text_input("Random seed (optional)", value="")
    _speed_opts = list(range(20, 155, 5))
    travel_speed_kmh = st.sidebar.selectbox(
        "Travel speed (km/h) for time estimate",
        options=_speed_opts,
        index=_speed_opts.index(60),
    )

    cities = st.session_state.loaded_cities
    source = st.session_state.loaded_source

    if cities is None:
        st.info("Load a dataset from the sidebar to continue.")
        st.stop()

    _validate_cities_or_stop(cities)

    if tournament_size > pop_size:
        _show_error_with_fix(
            "Tournament size cannot be greater than population size.",
            ["Reduce tournament size.", "Or increase population size."],
        )
        st.stop()
    if elite_count >= pop_size:
        _show_error_with_fix(
            "Elite count must be smaller than population size.",
            ["Set elite count lower than population size.", "Typical safe value is 1 to 5."],
        )
        st.stop()

    with st.expander("Dataset preview"):
        city_table = pd.DataFrame(
            [{"City": c.name, "Latitude": c.lat, "Longitude": c.lon} for c in cities]
        )
        st.dataframe(city_table, width="stretch", hide_index=True)
        st.write(f"Total cities: **{len(cities)}**")
        re = st.session_state.get("loaded_road_edges") or set()
        st.write(f"Road edges in file: **{len(re)}**")
        if source:
            st.write(f"Loaded from: `{source}`")

    st.subheader("Map picker")
    st.write(
        "Click a city marker on the map. After you set **Source**, the next click mode switches "
        "to **Destination** automatically. Green marks source, red marks destination."
    )

    bc1, bc2, bc3 = st.columns([1, 1, 2])
    with bc1:
        if st.button("Clear source"):
            st.session_state.pick_src = None
            st.session_state.next_pick_role = "source"
    with bc2:
        if st.button("Clear destination"):
            st.session_state.pick_dst = None
            if st.session_state.pick_src is not None:
                st.session_state.next_pick_role = "destination"
            else:
                st.session_state.next_pick_role = "source"

    with bc3:
        ps = st.session_state.pick_src
        pdst = st.session_state.pick_dst
        swap_disabled = ps is None or pdst is None or ps == pdst
        if st.button("Swap source ↔ destination", disabled=swap_disabled):
            st.session_state.pick_src, st.session_state.pick_dst = pdst, ps
            st.session_state.directions_result = None
            st.session_state.pop("_route_map_blob", None)

    road_e = st.session_state.get("loaded_road_edges") or None
    city_fig = _fig_city_picker(
        cities,
        st.session_state.pick_src,
        st.session_state.pick_dst,
        road_edges=road_e,
    )
    city_event = st.plotly_chart(
        city_fig,
        key="city_pick",
        on_select="rerun",
        selection_mode="points",
        width="stretch",
    )

    sel_points = []
    if city_event and city_event.selection:
        sel_points = city_event.selection.points or []
    if not sel_points and "city_pick" in st.session_state:
        sel = st.session_state.city_pick.selection if hasattr(st.session_state.city_pick, "selection") else None
        if sel is None and isinstance(st.session_state.city_pick, dict):
            sel = st.session_state.city_pick.get("selection")
        if sel:
            sel_points = sel.get("points", []) if isinstance(sel, dict) else getattr(sel, "points", []) or []

    pick_role = st.session_state.get("next_pick_role", "source")

    if sel_points:
        idx = _city_idx_from_point(sel_points[0])
        if idx is not None:
            if pick_role == "source":
                st.session_state.pick_src = idx
                st.session_state.next_pick_role = "destination"
            else:
                if st.session_state.pick_src == idx:
                    st.warning("Destination must differ from source.")
                else:
                    st.session_state.pick_dst = idx

    role = st.session_state.get("next_pick_role", "source")
    st.caption("Next map click assigns:")
    m1, m2 = st.columns(2)
    with m1:
        if st.button(
            "Source",
            type="primary" if role == "source" else "secondary",
            use_container_width=True,
            key="pick_mode_source",
        ):
            st.session_state.next_pick_role = "source"
    with m2:
        if st.button(
            "Destination",
            type="primary" if role == "destination" else "secondary",
            use_container_width=True,
            key="pick_mode_dest",
        ):
            st.session_state.next_pick_role = "destination"

    src_i = st.session_state.pick_src
    dst_i = st.session_state.pick_dst
    if src_i is not None:
        st.success(f"Source: **{cities[src_i].name}**")
    if dst_i is not None:
        st.success(f"Destination: **{cities[dst_i].name}**")

    dirs_disabled = src_i is None or dst_i is None or src_i == dst_i
    if st.button("Get Directions", type="primary", disabled=dirs_disabled):
        seed = None
        if seed_input.strip() != "":
            try:
                seed = int(seed_input.strip())
            except ValueError:
                _show_error_with_fix(
                    "Random seed must be an integer.",
                    ["Use values like 1, 42, 123.", "Leave blank for a non-deterministic run."],
                )
                st.stop()

        try:
            if seed is not None:
                random.seed(seed)

            elite_use = max(1, min(elite_count, pop_size - 2))

            loaded_edges = st.session_state.get("loaded_road_edges") or None
            ga_roads = loaded_edges if loaded_edges else None

            ga = CorridorPathGA(
                cities=cities,
                src_idx=src_i,
                dst_idx=dst_i,
                pop_size=pop_size,
                mutation_rate=max(0.05, min(0.45, mutation_rate * 8)),
                tournament_size=min(tournament_size, pop_size),
                elite_count=elite_use,
                corridor_ratio=corridor_ratio,
                k_nn=None,
                road_edges=ga_roads,
            )

            t0 = time.time()
            best_route, best_dist, route_archive = ga.run(
                generations=generations,
                verbose=False,
            )
            elapsed = time.time() - t0

            if not validate_corridor_path(best_route, src_i, dst_i):
                st.error("GA produced an invalid path.")
                st.stop()

            st.session_state.directions_result = {
                "best_route": best_route,
                "best_dist": best_dist,
                "route_archive": route_archive,
                "evolution_snapshots": list(ga.evolution_snapshots),
                "fitness_history": list(ga.fitness_history),
                "elapsed": elapsed,
                "src": src_i,
                "dst": dst_i,
            }
            st.session_state.route_pick_idx = 0
            st.session_state.evo_frame = 0
            st.session_state.evo_play = False
            st.session_state.route_map_version = int(st.session_state.get("route_map_version", 0)) + 1
            st.session_state.pop("_route_map_blob", None)

        except Exception as e:
            _show_error_with_fix(
                f"Directions failed: {e}",
                [
                    "Confirm GA parameters are valid.",
                    "Try fewer generations or a smaller population.",
                    "Reload data and pick source and destination again.",
                ],
            )

    dr = st.session_state.directions_result
    if dr:
        st.divider()
        st.subheader("Directions output")

        archive = dr.get("route_archive")
        if not archive:
            archive = [[dr["best_route"], dr["best_dist"]]]
        tol_frac = alt_tolerance_pct / 100.0
        alts = _select_routes_from_archive(
            archive,
            dr["best_dist"],
            tol_frac,
            min_route_suggestions,
            max_route_suggestions,
        )
        if not alts:
            alts = [(dr["best_route"], dr["best_dist"])]

        if st.session_state.route_pick_idx >= len(alts):
            st.session_state.route_pick_idx = 0

        route_labels = [f"Route {i + 1}: {d:,.2f} km" for i, (_, d) in enumerate(alts)]

        def _route_label(i):
            return route_labels[i]

        route_map_key = f"route_map_{st.session_state.route_map_version}"
        rp_pts = _plotly_map_selection_points(route_map_key)
        blob = json.dumps(rp_pts, sort_keys=True, default=str)
        if blob != st.session_state.get("_route_map_blob"):
            st.session_state._route_map_blob = blob
            rid = _extract_route_click(rp_pts)
            if rid is not None and 0 <= rid < len(alts):
                st.session_state.route_pick_idx = rid
                st.session_state.evo_frame = 0

        r_only = st.session_state.get("road_network_only")
        re_plot = st.session_state.get("loaded_road_edges") or set()
        if r_only and re_plot:
            st.info("Road network only is on: optimized route overlays are hidden.")
            rmap_fig = _fig_road_network(cities, re_plot, title="Road network (from CSV)")
        else:
            rmap_fig = _fig_routes(cities, alts, st.session_state.route_pick_idx)
        st.plotly_chart(
            rmap_fig,
            key=route_map_key,
            on_select="rerun",
            selection_mode="points",
            width="stretch",
        )

        st.caption("Tip: click a pale bubble along a path segment to jump that route, or use the list below.")
        st.radio(
            "Highlight route",
            list(range(len(alts))),
            format_func=_route_label,
            horizontal=False,
            key="route_pick_idx",
            on_change=_on_route_highlight_change,
        )

        hi = min(st.session_state.route_pick_idx, len(alts) - 1)
        hi_dist = alts[hi][1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Shortest distance", f"{dr['best_dist']:,.2f} km")
        m2.metric("Highlighted route distance", f"{hi_dist:,.2f} km")
        m3.metric("Suggested routes", f"{len(alts)}")
        m4.metric(
            "Est. travel time",
            _format_travel_time(hi_dist, travel_speed_kmh),
        )

        st.subheader("GA evolution replay")
        st.caption(
            "**Play** or scrub frames below — this mirrors backend snapshots: bright routes are evaluated chromosomes, "
            "blue band + diamonds are the elite best-so-far at that algorithm generation."
        )

        ec = st.columns([1, 1, 1, 2])
        with ec[0]:
            if st.button("Play", key="evo_btn_play"):
                st.session_state.evo_play = True
        with ec[1]:
            if st.button("Pause", key="evo_btn_pause"):
                st.session_state.evo_play = False
        with ec[2]:
            if st.button("Reset", key="evo_btn_reset"):
                st.session_state.evo_frame = 0
                st.session_state.evo_play = False

        directions_evolution_fragment()

        _directions_fitness_line_chart(dr.get("fitness_history"))

        hi_route = alts[hi][0]
        order_df = pd.DataFrame(
            {
                "Stop": list(range(1, len(hi_route) + 1)),
                "City": [cities[i].name for i in hi_route],
            }
        )
        st.subheader("Stops on highlighted route")
        st.dataframe(order_df, width="stretch", hide_index=True)

    st.divider()
    with st.expander("Classic closed tour (TSP round trip, optional)"):
        st.write(
            "Runs the original closed-loop genetic tour through every city and returning to the start. "
            "Separate from the directions workflow."
        )
        if st.button("Optimize closed tour"):
            seed = None
            if seed_input.strip() != "":
                try:
                    seed = int(seed_input.strip())
                except ValueError:
                    _show_error_with_fix(
                        "Random seed must be an integer.",
                        ["Use values like 1, 42, 123.", "Leave blank for random behavior each run."],
                    )
                    st.stop()
            try:
                if seed is not None:
                    random.seed(seed)
                ga = GeneticAlgorithm(
                    cities=cities,
                    pop_size=pop_size,
                    mutation_rate=mutation_rate,
                    tournament_size=tournament_size,
                    elite_count=elite_count,
                )
                t0 = time.time()
                best_route, best_dist = ga.run(generations=generations, verbose=False)
                elapsed = time.time() - t0
                if not validate_route(best_route, len(cities)):
                    st.error("Generated route is invalid.")
                    st.stop()
                os.makedirs("results", exist_ok=True)
                route_fig = os.path.join("results", "best_route.png")
                fit_fig = os.path.join("results", "fitness_plot.png")
                plot_route(best_route, cities, save_path=route_fig)
                plot_fitness(ga.fitness_history, save_path=fit_fig)
                st.session_state.classic_result = {
                    "best_dist": best_dist,
                    "elapsed": elapsed,
                    "route_fig": route_fig,
                    "fit_fig": fit_fig,
                }
            except Exception as e:
                _show_error_with_fix(f"Classic run failed: {e}", ["Adjust GA parameters.", "Reload data and retry."])

        cr = st.session_state.classic_result
        if cr:
            c1, c2, c3 = st.columns(3)
            c1.metric("Best closed distance", f"{cr['best_dist']:,.2f} km")
            c2.metric("Runtime", f"{cr['elapsed']:.2f} s")
            c3.metric("Cities", len(cities))
            st.image(cr["route_fig"])
            st.image(cr["fit_fig"])


if __name__ == "__main__":
    main()

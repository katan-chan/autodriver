from __future__ import annotations

from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import osmnx as ox
import plotly.graph_objects as go

GRAPHML_PATH = Path("data/hanoi_roads.graphml")
BANDWIDTH_ATTR = "bandwidth_capacity"
TRAVEL_TIME_ATTR = "travel_time_seconds"
WIDTH_ATTR = "width_m"

LANE_WIDTH_BY_HIGHWAY = {
    "motorway": 3.75,
    "trunk": 3.5,
    "primary": 3.4,
    "secondary": 3.3,
    "tertiary": 3.2,
    "residential": 3.0,
    "service": 2.8,
}

DEFAULT_LANE_WIDTH_M = 3.2
DEFAULT_WIDTH_BY_HIGHWAY = {
    "motorway": 25.0,
    "trunk": 18.0,
    "primary": 14.0,
    "secondary": 12.0,
    "tertiary": 10.0,
    "residential": 8.0,
    "service": 6.0,
}
DEFAULT_WIDTH_M = 8.0


def load_hanoi_graph(graphml_path: Path = GRAPHML_PATH) -> nx.Graph:
    """Load the Hanoi road graph from disk and convert to an undirected simple graph."""
    if not graphml_path.exists():
        raise FileNotFoundError(
            f"GraphML file not found at {graphml_path}. Run crawl_hanoi_osm.py first."
        )
    multi_graph = ox.load_graphml(graphml_path)
    undirected = multi_graph.to_undirected()
    simple_graph = nx.Graph()
    for u, data in undirected.nodes(data=True):
        simple_graph.add_node(u, **data)
    for u, v, data in undirected.edges(data=True):
        if simple_graph.has_edge(u, v):
            # Keep the shortest edge between the pair to avoid duplicates
            if data.get("length", np.inf) < simple_graph[u][v].get("length", np.inf):
                simple_graph[u][v].update(data)
        else:
            simple_graph.add_edge(u, v, **data)
    return simple_graph


def assign_edge_metrics(
    graph: nx.Graph,
    bandwidth_attr: str = BANDWIDTH_ATTR,
    travel_time_attr: str = TRAVEL_TIME_ATTR,
    width_attr: str = WIDTH_ATTR,
    rng_seed: int = 42,
) -> None:
    """Ensure each edge has bandwidth & travel time attributes (generate if missing)."""
    rng = np.random.default_rng(rng_seed)

    highway_bandwidth = {
        "motorway": (40, 60),
        "trunk": (30, 45),
        "primary": (20, 35),
        "secondary": (15, 25),
        "tertiary": (10, 18),
        "residential": (5, 12),
        "service": (3, 8),
    }

    default_speed_by_highway = {
        "motorway": 80,
        "trunk": 65,
        "primary": 55,
        "secondary": 45,
        "tertiary": 40,
        "residential": 30,
        "service": 25,
    }

    for _, _, data in graph.edges(data=True):
        highway = _first_value(data.get("highway"))

        if bandwidth_attr not in data:
            low, high = highway_bandwidth.get(highway, (5, 15))
            data[bandwidth_attr] = float(rng.integers(low, high + 1))

        if travel_time_attr not in data:
            length_m = float(data.get("length") or 0.0)
            speed_kph = _parse_speed_kph(data.get("maxspeed"))
            if speed_kph is None:
                speed_kph = default_speed_by_highway.get(highway, 35)
            speed_mps = max(speed_kph * (1000 / 3600), 1e-3)
            travel_time = length_m / speed_mps if length_m > 0 else rng.uniform(5, 30)
            data[travel_time_attr] = float(travel_time)

        if width_attr not in data:
            data[width_attr] = _estimate_edge_width(
                highway=highway,
                width_value=_first_value(data.get("width")),
                lanes_value=_first_value(data.get("lanes")),
            )


def visualize_hanoi_graph(
    graph: nx.Graph,
    color_attr: str = BANDWIDTH_ATTR,
    title: str = "Hanoi road network",
    node_size: int = 4,
    show: bool = True,
):
    """Render the graph with Plotly, coloring edge midpoints by the chosen attribute."""
    node_x, node_y, node_hover = [], [], []
    node_degree = dict(graph.degree())

    for node, attrs in graph.nodes(data=True):
        node_x.append(attrs.get("x"))
        node_y.append(attrs.get("y"))
        node_hover.append(f"node {node}<br>degree={node_degree.get(node, 0)}")

    edge_x, edge_y = _collect_edge_lines(graph)

    edge_mid_x, edge_mid_y, metric_values, edge_hover = [], [], [], []
    for u, v, attrs in graph.edges(data=True):
        mid_lon, mid_lat = _edge_midpoint(u, v, attrs, graph)
        edge_mid_x.append(mid_lon)
        edge_mid_y.append(mid_lat)
        metric = float(attrs.get(color_attr, 0.0))
        metric_values.append(metric)
        travel_time = attrs.get(TRAVEL_TIME_ATTR)
        bandwidth = attrs.get(BANDWIDTH_ATTR)
        edge_hover.append(
            f"{u} ↔ {v}<br>" f"{color_attr}={metric:.2f}" +
            (f"<br>{TRAVEL_TIME_ATTR}={travel_time:.2f}s" if travel_time is not None else "") +
            (f"<br>{BANDWIDTH_ATTR}={bandwidth:.2f}" if bandwidth is not None else "")
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#B0B0B0", width=1),
            hoverinfo="none",
            showlegend=False,
            name="roads",
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=edge_mid_x,
            y=edge_mid_y,
            mode="markers",
            marker=dict(
                size=6,
                color=metric_values,
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title=color_attr),
            ),
            text=edge_hover,
            hoverinfo="text",
            name=color_attr,
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=node_size,
                color=list(node_degree.values()),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="degree"),
            ),
            text=node_hover,
            hoverinfo="text",
            name="nodes",
        )
    )

    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(title="longitude", showgrid=False, zeroline=False),
        yaxis=dict(title="latitude", showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        height=700,
    )

    if show:
        fig.show()
    return fig


def _collect_edge_lines(graph: nx.Graph) -> tuple[list[float], list[float]]:
    edge_x: list[float] = []
    edge_y: list[float] = []
    for u, v, data in graph.edges(data=True):
        coords = _edge_coords(u, v, data, graph)
        for lon, lat in coords:
            edge_x.append(lon)
            edge_y.append(lat)
        edge_x.append(None)
        edge_y.append(None)
    return edge_x, edge_y


def _edge_coords(u: int, v: int, data: dict, graph: nx.Graph) -> Iterable[tuple[float, float]]:
    geometry = data.get("geometry")
    if geometry is not None:
        return list(geometry.coords)
    return [
        (graph.nodes[u].get("x"), graph.nodes[u].get("y")),
        (graph.nodes[v].get("x"), graph.nodes[v].get("y")),
    ]


def _edge_midpoint(u: int, v: int, data: dict, graph: nx.Graph) -> tuple[float, float]:
    coords = list(_edge_coords(u, v, data, graph))
    if not coords:
        return 0.0, 0.0
    lon = (coords[0][0] + coords[-1][0]) / 2.0
    lat = (coords[0][1] + coords[-1][1]) / 2.0
    return lon, lat


def _first_value(value):
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def _parse_speed_kph(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        for item in value:
            parsed = _parse_speed_kph(item)
            if parsed is not None:
                return parsed
        return None
    text = str(value).lower().strip()
    if not text:
        return None
    for sep in [";", ","]:
        if sep in text:
            text = text.split(sep)[0]
            break
    text = text.replace("km/h", "").replace("kph", "").strip()
    factor = 1.0
    if "mph" in text:
        text = text.replace("mph", "").strip()
        factor = 1.60934
    try:
        return float(text) * factor
    except ValueError:
        return None


if __name__ == "__main__":
    graph = load_hanoi_graph(GRAPHML_PATH)
    assign_edge_metrics(graph)
    visualize_hanoi_graph(graph)

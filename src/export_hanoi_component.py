from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Iterable

import networkx as nx

from visualize_hanoi_graph import load_hanoi_graph

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

GRAPHML_PATH = DATA_DIR / "hanoi_roads.graphml"
TARGET_NODES = 10000
TARGET_LAT = 20.995872180564596
TARGET_LON = 105.80795573888336
EDGE_LIST_PATH = DATA_DIR / "hanoi_component_edges.csv"
NODE_CSV_PATH = DATA_DIR / "hanoi_component_nodes.csv"
TARGET_SEED_NODE: int | None = 6689057484  # set node id to fix component; None => auto by location
RADIUS_METERS: float | None = 5000.0  # include nodes within this radius from seed; None => use TARGET_NODES


def main():
    graph = load_hanoi_graph(GRAPHML_PATH)
    seed_node = TARGET_SEED_NODE if TARGET_SEED_NODE is not None else find_closest_node(
        graph, TARGET_LAT, TARGET_LON
    )
    component = extract_component_from_seed(
        graph,
        seed_node,
        node_limit=TARGET_NODES,
        radius_m=RADIUS_METERS,
    )
    print(
        f"Selected subgraph with seed {seed_node}, {component.number_of_nodes()} nodes, "
        f"{component.number_of_edges()} edges."
    )
    write_edge_list(component, EDGE_LIST_PATH)
    write_node_table(component, NODE_CSV_PATH)
    print(f"Edge list saved to {EDGE_LIST_PATH}")
    print(f"Node table saved to {NODE_CSV_PATH}")
    
    # Auto-refresh backend cache
    try:
        import urllib.request
        print("Triggering backend graph refresh...")
        req = urllib.request.Request("http://localhost:8000/graph/refresh", method="POST")
        with urllib.request.urlopen(req, timeout=2) as response:
            print(f"Backend refreshed: {response.status} {response.msg}")
    except Exception as e:
        print(f"Could not refresh backend (server might be down): {e}")


def extract_component_from_seed(
    graph: nx.Graph,
    start_node: int,
    node_limit: int,
    radius_m: float | None = None,
) -> nx.Graph:
    if start_node is None:
        raise ValueError("Could not determine a valid starting node.")

    if radius_m is not None:
        nodes = nodes_within_radius(graph, start_node, radius_m)
        if not nodes:
            raise ValueError(
                f"No nodes found within {radius_m} m of seed {start_node}. "
                "Try increasing RADIUS_METERS or clearing TARGET_SEED_NODE."
            )
        print(f"Collected {len(nodes)} nodes within {radius_m} m radius.")
    else:
        nodes = bfs_nodes(graph, start_node, node_limit)
        if len(nodes) < node_limit:
            print(
                f"Warning: reachable component from seed has only {len(nodes)} nodes (< {node_limit})."
            )

    return graph.subgraph(nodes).copy()


def bfs_nodes(graph: nx.Graph, start_node: int, limit: int) -> list[int]:
    queue = deque([start_node])
    visited: list[int] = []
    seen = {start_node}

    while queue and len(visited) < limit:
        node = queue.popleft()
        visited.append(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)

    return visited


def nodes_within_radius(graph: nx.Graph, seed_node: int, radius_m: float) -> list[int]:
    seed_lat = graph.nodes[seed_node].get("y")
    seed_lon = graph.nodes[seed_node].get("x")
    if seed_lat is None or seed_lon is None:
        raise ValueError(f"Seed node {seed_node} lacks coordinates.")

    selected_set: set[int] = set()
    for node, attrs in graph.nodes(data=True):
        lat = attrs.get("y")
        lon = attrs.get("x")
        if lat is None or lon is None:
            continue
        dist = haversine_m(seed_lat, seed_lon, lat, lon)
        if dist <= radius_m:
            selected_set.add(node)

    # include immediate neighbors to avoid dangling edges at boundary
    expanded = set(selected_set)
    for node in list(selected_set):
        for neighbor in graph.neighbors(node):
            expanded.add(neighbor)

    return sorted(expanded)


def write_edge_list(graph: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "source",
                "target",
                "length_m",
                "bandwidth_capacity",
                "travel_time_seconds",
                "width_m",
            ]
        )
        for u, v, data in sorted(graph.edges(data=True)):
            width_m = _estimate_width(data)
            highway = _first_str(data.get("highway"))
            bandwidth_capacity = _estimate_bandwidth_capacity(width_m, highway)

            length_m = _safe_float(data.get("length"))
            travel_time = _safe_float(data.get("travel_time_seconds"))
            if travel_time is None:
                travel_time = _estimate_travel_time_seconds(length_m)

            writer.writerow(
                [
                    u,
                    v,
                    length_m,
                    bandwidth_capacity,
                    travel_time,
                    width_m,
                ]
            )


def write_node_table(graph: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["node_id", "lon", "lat", "degree"])
        for node in sorted(graph.nodes()):
            data = graph.nodes[node]
            lon = data.get("x")
            lat = data.get("y")
            degree = graph.degree(node)
            writer.writerow([node, lon, lat, degree])


def find_closest_node(graph: nx.Graph, target_lat: float, target_lon: float) -> int | None:
    best_node = None
    best_dist = float("inf")
    for node, attrs in graph.nodes(data=True):
        lat = attrs.get("y")
        lon = attrs.get("x")
        if lat is None or lon is None:
            continue
        dist = haversine_m(lat, lon, target_lat, target_lon)
        if dist < best_dist:
            best_dist = dist
            best_node = node
    return best_node


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import asin, cos, radians, sin, sqrt

    R = 6371000  # meters
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1_r = radians(lat1)
    lat2_r = radians(lat2)
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _estimate_width(data: dict) -> float | None:
    width = _safe_float(_first_str(data.get("width")))
    if width is not None:
        return width

    lanes = _safe_float(_first_str(data.get("lanes")))
    highway = _first_str(data.get("highway"))
    lane_width = LANE_WIDTH_BY_HIGHWAY.get(highway, DEFAULT_LANE_WIDTH_M)
    if lanes is not None:
        return lanes * lane_width

    return DEFAULT_WIDTH_BY_HIGHWAY.get(highway, DEFAULT_WIDTH_M)


def _first_str(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            if item:
                return item
        return None
    return value


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

STANDARD_LANE_WIDTH_M = 3.5
MIN_EFFECTIVE_WIDTH_M = 2.0
MIN_EFFECTIVE_LANES = 0.6
BASE_CAPACITY_PER_LANE = 600.0
HIGHWAY_CAPACITY_FACTOR = {
    "motorway": 1.2,
    "trunk": 1.1,
    "primary": 1.0,
    "secondary": 0.8,
    "tertiary": 0.7,
    "residential": 0.6,
    "service": 0.45,
    "living_street": 0.4,
}
DEFAULT_CAPACITY_FACTOR = 0.6
DEFAULT_SPEED_KMH = 40.0


def _estimate_bandwidth_capacity(width_m: float | None, highway: str | None) -> float | None:
    if width_m is None and highway is None:
        return None

    width = width_m
    if width is None:
        width = DEFAULT_WIDTH_BY_HIGHWAY.get(highway, DEFAULT_WIDTH_M)

    width = max(width, MIN_EFFECTIVE_WIDTH_M)
    effective_lanes = max(width / STANDARD_LANE_WIDTH_M, MIN_EFFECTIVE_LANES)
    raw_capacity = effective_lanes * BASE_CAPACITY_PER_LANE

    factor = HIGHWAY_CAPACITY_FACTOR.get(highway, DEFAULT_CAPACITY_FACTOR)
    return raw_capacity * factor


def _estimate_travel_time_seconds(length_m: float | None, speed_kmh: float = DEFAULT_SPEED_KMH) -> float | None:
    if length_m is None or speed_kmh <= 0:
        return None

    speed_m_per_s = speed_kmh / 3.6
    return length_m / speed_m_per_s


if __name__ == "__main__":
    main()

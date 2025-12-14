"""
Routing service module.

Re-exports algorithms from the algorithms package for backward compatibility
with existing imports in main.py.
"""

from __future__ import annotations

# Re-export all algorithms from the new modular package
from ..algorithms import (
    dijkstra_shortest_path,
    compute_k_shortest_paths_for_vehicle,
    compute_all_k_shortest_paths,
    build_edge_list_and_index,
    build_routes_from_choice,
    compute_edge_loads_from_routes,
    solve_routing_with_penalty,
)

__all__ = [
    "dijkstra_shortest_path",
    "compute_k_shortest_paths_for_vehicle",
    "compute_all_k_shortest_paths",
    "build_edge_list_and_index",
    "build_routes_from_choice",
    "compute_edge_loads_from_routes",
    "solve_routing_with_penalty",
]

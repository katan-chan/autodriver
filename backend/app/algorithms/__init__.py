"""Routing algorithms implemented as standalone modules."""

from .dijkstra import dijkstra_shortest_path
from .k_shortest import (
    compute_k_shortest_paths_for_vehicle,
    compute_all_k_shortest_paths,
)
from .k_shortest_parallel import compute_all_k_shortest_paths_parallel
from .edge_utils import (
    build_edge_list_and_index,
    build_routes_from_choice,
    compute_edge_loads_from_routes,
)
from .penalty_routing import solve_routing_with_penalty
from .penalty_routing_lazy import solve_routing_with_penalty_lazy
from .config import AlgorithmConfig, get_config, update_config, reset_config

from .cost_evaluator import (
    evaluate_path_congestion,
    compute_edge_time_usage,
)

# Registry system
from .registry import register, get, list_all

# Import algorithm modules to trigger @register decorators
from . import algo_shortest
from . import algo_penalty

__all__ = [
    "dijkstra_shortest_path",
    "compute_k_shortest_paths_for_vehicle",
    "compute_all_k_shortest_paths",
    "compute_all_k_shortest_paths_parallel",
    "build_edge_list_and_index",
    "build_routes_from_choice",
    "compute_edge_loads_from_routes",
    "solve_routing_with_penalty",
    "solve_routing_with_penalty_lazy",
    "evaluate_path_congestion",
    "compute_edge_time_usage",
    "AlgorithmConfig",
    "get_config",
    "update_config",
    "reset_config",
    # Registry
    "register",
    "get",
    "list_all",
]


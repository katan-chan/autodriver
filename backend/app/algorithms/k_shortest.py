"""
K-shortest paths computation using iterative Dijkstra with edge penalties.

Supports scipy sparse matrices.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .dijkstra import dijkstra_shortest_path


def compute_k_shortest_paths_for_vehicle(
    adjacency_travel_time,
    source: int,
    target: int,
    k_paths: int,
    large_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k shortest paths for a single origin-destination pair.

    After finding each path, edges on that path receive a large penalty
    to encourage subsequent iterations to find alternative routes.

    Parameters
    ----------
    adjacency_travel_time : sparse matrix or np.ndarray
        (n_nodes, n_nodes) travel time matrix.
    source : int
        Origin node index.
    target : int
        Destination node index.
    k_paths : int
        Number of alternative paths to compute.
    large_penalty : float
        Penalty added to edges after each path is found.

    Returns
    -------
    costs : np.ndarray
        Shape (k_paths,), cost of each path.
    paths : np.ndarray
        Shape (k_paths, n_nodes), each row is a path padded with -1.
    """
    n_nodes = adjacency_travel_time.shape[0]
    costs = np.full(k_paths, np.inf)
    paths = np.full((k_paths, n_nodes), -1, dtype=np.int64)

    # Convert to lil_matrix for efficient modification
    if sparse.issparse(adjacency_travel_time):
        work_travel = adjacency_travel_time.tolil()
    else:
        work_travel = sparse.lil_matrix(adjacency_travel_time)

    for k in range(k_paths):
        cost, path = dijkstra_shortest_path(work_travel, source, target)
        costs[k] = cost
        paths[k, :] = path

        if np.isinf(cost):
            break

        # Penalize edges on this path to find different routes next iteration
        for i in range(n_nodes - 1):
            u = path[i]
            v = path[i + 1]
            if v == -1:
                break
            current = work_travel[u, v]
            if current > 0:
                work_travel[u, v] = current + large_penalty
                work_travel[v, u] = current + large_penalty

    return costs, paths


def compute_all_k_shortest_paths(
    adjacency_travel_time,
    vehicle_origin: np.ndarray,
    vehicle_destination: np.ndarray,
    k_paths: int,
    large_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k shortest paths for all vehicles.

    Parameters
    ----------
    adjacency_travel_time : sparse matrix or np.ndarray
        (n_nodes, n_nodes) travel time matrix.
    vehicle_origin : np.ndarray
        (n_vehicles,) origin node indices.
    vehicle_destination : np.ndarray
        (n_vehicles,) destination node indices.
    k_paths : int
        Number of alternative paths per vehicle.
    large_penalty : float
        Penalty for edge reuse in k-shortest computation.

    Returns
    -------
    base_costs : np.ndarray
        Shape (n_vehicles, k_paths), base cost of each path.
    all_paths : np.ndarray
        Shape (n_vehicles, k_paths, n_nodes), all candidate paths.
    """
    n_vehicles = vehicle_origin.shape[0]
    n_nodes = adjacency_travel_time.shape[0]

    base_costs = np.full((n_vehicles, k_paths), np.inf)
    all_paths = np.full((n_vehicles, k_paths, n_nodes), -1, dtype=np.int64)

    for v in range(n_vehicles):
        s = int(vehicle_origin[v])
        t = int(vehicle_destination[v])
        costs_v, paths_v = compute_k_shortest_paths_for_vehicle(
            adjacency_travel_time, s, t, k_paths, large_penalty
        )
        base_costs[v, :] = costs_v
        all_paths[v, :, :] = paths_v

    return base_costs, all_paths

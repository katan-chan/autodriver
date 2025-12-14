"""
Lazy Regret Penalty Routing with Parallel Computation.

OPTIMIZATIONS:
1. Edge→Vehicle mapping for affected vehicle lookup
2. Cached penalized costs; only recalculate for affected vehicles
3. ThreadPoolExecutor for parallel cost computation
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from .k_shortest_parallel import compute_all_k_shortest_paths_parallel
from .edge_utils import (
    build_edge_list_and_index,
    build_routes_from_choice,
)


def _precompute_path_edges(
    all_paths: np.ndarray,
    csr_indptr: np.ndarray,
    csr_indices: np.ndarray,
    csr_data: np.ndarray,
    adjacency_travel_time,
) -> tuple[list, np.ndarray, set, dict]:
    """
    Pre-extract edge indices and travel times for all paths.
    Also build edge→vehicle mapping for lazy updates.
    
    Returns:
        path_edges: list of list of (edge_idx, travel_time) tuples
        base_costs: cached base costs (n_vehicles, k_paths)
        all_edge_set: set of all unique edge indices
        edge_to_vehicles: {edge_idx: set of vehicle indices using this edge}
    """
    n_vehicles, k_paths, n_nodes = all_paths.shape
    path_edges = []
    base_costs = np.full((n_vehicles, k_paths), np.inf, dtype=np.float64)
    all_edge_set = set()
    edge_to_vehicles = defaultdict(set)
    
    for v in range(n_vehicles):
        vehicle_paths = []
        for kk in range(k_paths):
            path = all_paths[v, kk]
            edges_in_path = []
            total_time = 0.0
            valid = True
            
            for i in range(n_nodes - 1):
                u = path[i]
                w = path[i + 1]
                if w < 0:
                    break
                
                e_idx = _get_edge_index_fast(u, w, csr_indptr, csr_indices, csr_data)
                t_val = adjacency_travel_time[u, w]
                travel_t = float(t_val) if t_val > 0 else 0.0
                
                if e_idx >= 0:
                    edges_in_path.append((e_idx, travel_t))
                    all_edge_set.add(e_idx)
                    edge_to_vehicles[e_idx].add(v)
                    total_time += travel_t
                else:
                    valid = False
                    break
            
            vehicle_paths.append(edges_in_path)
            if valid and len(edges_in_path) > 0:
                base_costs[v, kk] = total_time
        
        path_edges.append(vehicle_paths)
    
    return path_edges, base_costs, all_edge_set, dict(edge_to_vehicles)


def _get_edge_index_fast(u: int, v: int, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray) -> int:
    """Fast edge index lookup from CSR structure."""
    start = indptr[u]
    end = indptr[u+1]
    for i in range(start, end):
        if indices[i] == v:
            return int(data[i]) - 1
    return -1


def _compute_cost_for_vehicle(
    v: int,
    path_edges: list,
    base_costs: np.ndarray,
    start_times: np.ndarray,
    time_bucket: float,
    max_buckets: int,
    edge_time_loads: np.ndarray,
    edge_bandwidth: np.ndarray,
    penalty_beta: float,
) -> tuple[int, np.ndarray]:
    """Compute penalized costs for a single vehicle (parallelizable)."""
    k_paths = len(path_edges[v])
    costs = np.full(k_paths, np.inf, dtype=np.float64)
    t_start = start_times[v]
    
    for kk in range(k_paths):
        base = base_costs[v, kk]
        if np.isinf(base):
            continue
        
        total_penalty = 0.0
        current_time = t_start
        
        for (e_idx, travel_t) in path_edges[v][kk]:
            bucket = int(current_time // time_bucket)
            if bucket < max_buckets:
                B = edge_bandwidth[e_idx]
                if B > 0:
                    load = edge_time_loads[e_idx, bucket]
                    ratio = load / B
                    if ratio > 1.0:
                        total_penalty += np.exp(penalty_beta * (min(ratio, 12.0) - 1.0))
            current_time += travel_t
        
        costs[kk] = base + total_penalty
    
    return v, costs


def _compute_costs_parallel(
    vehicles_to_update: set,
    path_edges: list,
    base_costs: np.ndarray,
    start_times: np.ndarray,
    time_bucket: float,
    max_buckets: int,
    edge_time_loads: np.ndarray,
    edge_bandwidth: np.ndarray,
    penalty_beta: float,
    cached_costs: np.ndarray,
    max_workers: int = 4,
) -> np.ndarray:
    """Parallel computation of penalized costs for affected vehicles only."""
    if not vehicles_to_update:
        return cached_costs
    
    vehicles_list = list(vehicles_to_update)
    
    # For small batches, don't use parallelism (overhead not worth it)
    if len(vehicles_list) < 50:
        for v in vehicles_list:
            _, costs = _compute_cost_for_vehicle(
                v, path_edges, base_costs, start_times,
                time_bucket, max_buckets, edge_time_loads,
                edge_bandwidth, penalty_beta
            )
            cached_costs[v] = costs
        return cached_costs
    
    # Parallel computation for larger batches
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _compute_cost_for_vehicle,
                v, path_edges, base_costs, start_times,
                time_bucket, max_buckets, edge_time_loads,
                edge_bandwidth, penalty_beta
            ): v for v in vehicles_list
        }
        
        for future in as_completed(futures):
            v, costs = future.result()
            cached_costs[v] = costs
    
    return cached_costs


def _update_loads_for_path(
    edge_time_loads: np.ndarray,
    path_edges_list: list,
    start_time: float,
    time_bucket: float,
    max_buckets: int,
) -> set:
    """Update edge loads for a chosen path. Returns affected edge indices."""
    affected_edges = set()
    current_time = start_time
    
    for (e_idx, travel_t) in path_edges_list:
        end_time = current_time + travel_t
        start_bucket = int(current_time // time_bucket)
        end_bucket = int(end_time // time_bucket)
        
        if start_bucket < max_buckets and e_idx >= 0:
            cap_end = min(end_bucket, max_buckets - 1)
            edge_time_loads[e_idx, start_bucket:(cap_end + 1)] += 1
            affected_edges.add(e_idx)
        
        current_time = end_time
    
    return affected_edges


def solve_routing_with_penalty_lazy(
    adjacency_travel_time,
    adjacency_bandwidth,
    origins: np.ndarray,
    destinations: np.ndarray,
    start_times: np.ndarray | None = None,
    k_paths: int = 3,
    large_penalty_for_alt: float = 1000.0,
    penalty_beta: float = 1.0,
    max_workers: int = 4,
) -> tuple:
    """
    Solve route assignment with LAZY REGRET algorithm.
    
    Key Optimizations:
    1. Edge→Vehicle mapping for O(1) affected vehicle lookup
    2. Cached costs; only recalculate for affected vehicles
    3. Parallel cost computation with ThreadPoolExecutor
    
    Complexity: O(n × m × k × e̅) where m = affected vehicles per iteration (~5-10% of n)
    vs Original: O(n² × k × e̅)
    """
    n_nodes = adjacency_travel_time.shape[0]
    n_vehicles = origins.shape[0]
    
    if start_times is None:
        start_times = np.zeros(n_vehicles, dtype=np.float64)

    # 1) Compute top-k shortest paths (parallel with OD caching)
    print("[DEBUG] Computing k-shortest paths (k={})".format(k_paths), flush=True)
    _, all_paths = compute_all_k_shortest_paths_parallel(
        adjacency_travel_time,
        origins,
        destinations,
        k_paths,
        large_penalty_for_alt,
        max_workers=max_workers,
    )
    print("[DEBUG] K-shortest paths computed.", flush=True)

    # 2) Build edge list and CSR arrays
    edge_u, edge_v, edge_bandwidth, csr_indptr, csr_indices, csr_data = build_edge_list_and_index(
        adjacency_bandwidth
    )
    n_edges = len(edge_u)
    
    # 3) Pre-compute path edges WITH edge→vehicle mapping
    print(f"[DEBUG] Pre-computing path edges with edge→vehicle mapping...", flush=True)
    path_edges, base_costs, all_edge_set, edge_to_vehicles = _precompute_path_edges(
        all_paths, csr_indptr, csr_indices, csr_data, adjacency_travel_time
    )
    print(f"[DEBUG] Unique edges: {len(all_edge_set)}", flush=True)
    
    # 4) Time bucket config
    TIME_BUCKET = 60.0
    MAX_BUCKETS = 120
    
    # 5) Initialize
    chosen_k = np.full(n_vehicles, -1, dtype=np.int64)
    assigned = np.zeros(n_vehicles, dtype=bool)
    edge_time_loads = np.zeros((n_edges, MAX_BUCKETS), dtype=np.float32)
    
    # Cached costs (updated lazily)
    cached_costs = np.full((n_vehicles, k_paths), np.inf, dtype=np.float64)
    
    # Initial full computation (parallel)
    print(f"[DEBUG] Initial cost computation (parallel)...", flush=True)
    all_vehicles = set(range(n_vehicles))
    cached_costs = _compute_costs_parallel(
        all_vehicles, path_edges, base_costs, start_times,
        TIME_BUCKET, MAX_BUCKETS, edge_time_loads, edge_bandwidth,
        penalty_beta, cached_costs, max_workers
    )
    
    print(f"[DEBUG] Starting LAZY greedy assignment for {n_vehicles} vehicles...", flush=True)
    
    # 6) Lazy Greedy assignment by regret
    for v_iter in range(n_vehicles):
        if v_iter % 100 == 0:
            print(f"[PROGRESS] Assigned {v_iter}/{n_vehicles} vehicles...", flush=True)
        
        # Find unassigned vehicle with max regret (using CACHED costs)
        best_vehicle = -1
        best_regret = -np.inf
        best_k_for_vehicle = 0

        for v in range(n_vehicles):
            if assigned[v]:
                continue
            
            costs_v = cached_costs[v]
            valid_c = costs_v[~np.isinf(costs_v)]
            
            if len(valid_c) == 0:
                continue
            
            sorted_costs = np.sort(valid_c)
            best_k = int(np.argmin(costs_v))
            
            if len(sorted_costs) >= 2:
                regret = sorted_costs[1] - sorted_costs[0]
            else:
                regret = 0.0

            if regret > best_regret or best_vehicle == -1:
                best_regret = regret
                best_vehicle = v
                best_k_for_vehicle = best_k

        if best_vehicle == -1:
            break

        # Assign the chosen vehicle
        assigned[best_vehicle] = True
        chosen_k[best_vehicle] = best_k_for_vehicle

        # Update edge loads and get affected edges
        affected_edges = _update_loads_for_path(
            edge_time_loads,
            path_edges[best_vehicle][best_k_for_vehicle],
            start_times[best_vehicle],
            TIME_BUCKET,
            MAX_BUCKETS,
        )
        
        # Find affected vehicles (LAZY: only vehicles using these edges)
        affected_vehicles = set()
        for e_idx in affected_edges:
            affected_vehicles.update(edge_to_vehicles.get(e_idx, set()))
        
        # Remove already assigned vehicles
        affected_vehicles -= set(np.where(assigned)[0])
        
        # Update costs only for affected vehicles (parallel if large enough)
        if affected_vehicles:
            cached_costs = _compute_costs_parallel(
                affected_vehicles, path_edges, base_costs, start_times,
                TIME_BUCKET, MAX_BUCKETS, edge_time_loads, edge_bandwidth,
                penalty_beta, cached_costs, max_workers
            )

    print(f"[DEBUG] Assignment complete. Assigned {assigned.sum()}/{n_vehicles} vehicles.", flush=True)

    # 7) Build final routes
    for v in range(n_vehicles):
        if chosen_k[v] < 0:
            chosen_k[v] = 0

    routes_final = build_routes_from_choice(all_paths, chosen_k)
    
    # 8) Compute detailed metadata
    routes_metadata = []
    
    for v in range(n_vehicles):
        k = chosen_k[v]
        path_edges_list = path_edges[v][k]
        base_time = base_costs[v, k]
        if np.isinf(base_time):
            base_time = 0.0
            
        current_time = start_times[v]
        accumulated_delay = 0.0
        congestion_events = []
        
        for (e_idx, travel_t) in path_edges_list:
            bucket = int(current_time // TIME_BUCKET)
            if bucket < MAX_BUCKETS and e_idx >= 0:
                B = edge_bandwidth[e_idx]
                if B > 0:
                     load = edge_time_loads[e_idx, bucket]
                     ratio = load / B
                     if ratio > 1.0:
                         penalty_val = np.exp(penalty_beta * (min(ratio, 12.0) - 1.0))
                         accumulated_delay += penalty_val
                         congestion_events.append({
                             "time": float(current_time),
                             "edge_idx": int(e_idx),
                             "ratio": float(ratio),
                             "delay": float(penalty_val)
                         })
            current_time += travel_t
            
        routes_metadata.append({
            "base_travel_time": float(base_time),
            "penalty_delay": float(accumulated_delay),
            "total_travel_time": float(base_time + accumulated_delay),
            "congestion_times": congestion_events
        })

    return routes_final, edge_time_loads, edge_u, edge_v, TIME_BUCKET, MAX_BUCKETS, routes_metadata

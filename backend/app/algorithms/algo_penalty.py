"""
Penalty-based routing algorithm with lazy regret optimization.
"""

from __future__ import annotations
import numpy as np

from .registry import register
from .penalty_routing_lazy import solve_routing_with_penalty_lazy


@register("penalty")
def solve_penalty_routing(graph_data, requests, config):
    """
    Penalty-based routing with lazy regret and parallel K-shortest optimization.
    
    Args:
        graph_data: GraphData instance
        requests: list of dicts with keys: origin_node, destination_node, request_id, start_time
        config: dict with keys: bandwidth_scale, k_paths, penalty_beta, k_shortest_penalty
    
    Returns:
        dict with keys: paths, edge_stats, timeline, edge_time_usage, peak_hours, failed_requests
    """
    # Extract parameters
    origins = np.array([graph_data.node_id_to_index(r["origin_node"]) for r in requests], dtype=np.int64)
    dests = np.array([graph_data.node_id_to_index(r["destination_node"]) for r in requests], dtype=np.int64)
    start_times = np.array([r.get("start_time", 0) for r in requests], dtype=np.float64)
    request_ids = [r["request_id"] for r in requests]
    
    # Config
    bandwidth_scale = config.get("bandwidth_scale", 1.0)
    k_paths = config.get("k_paths", 3)
    penalty_beta = config.get("penalty_beta", 1.0)
    k_shortest_penalty = config.get("k_shortest_penalty", 1000.0)
    
    # Scale bandwidth
    scaled_bandwidth = graph_data.adjacency_bandwidth * bandwidth_scale
    
    # Run penalty routing
    routes, edge_time_loads, edge_u, edge_v, time_bucket, max_buckets, routes_metadata = \
        solve_routing_with_penalty_lazy(
            graph_data.adjacency_travel_time,
            scaled_bandwidth,
            origins=origins,
            destinations=dests,
            start_times=start_times,
            k_paths=k_paths,
            large_penalty_for_alt=k_shortest_penalty,
            penalty_beta=penalty_beta,
            max_workers=4,
        )
    
    # Convert to path models
    node_paths = graph_data.routes_indices_to_node_ids(routes)
    path_models = []
    
    for i, node_path in enumerate(node_paths):
        meta = routes_metadata[i]
        path_models.append({
            "node_ids": node_path,
            "travel_time": meta["total_travel_time"],
            "base_travel_time": meta["base_travel_time"],
            "penalty_delay": meta["penalty_delay"],
            "congestion_times": meta.get("congestion_times", []),
            "request_id": request_ids[i],
            "start_time": start_times[i],
        })
    
    # Build response
    from ..main import _build_route_response
    
    result = _build_route_response(
        graph_data, path_models, [],
        bandwidth_scale,
        edge_time_loads=edge_time_loads,
        edge_u=edge_u,
        edge_v=edge_v,
        time_bucket=time_bucket,
        max_buckets=max_buckets,
    )
    
    # Extract peak hours from loads
    peak_hours = _extract_peak_hours_from_loads(
        edge_time_loads, edge_u, edge_v, bandwidth_scale, graph_data
    )
    result["peak_hours"] = peak_hours
    
    return result


def _extract_peak_hours_from_loads(edge_time_loads, edge_u, edge_v, bandwidth_scale, graph_data):
    """Extract peak hours from edge_time_loads array."""
    if edge_time_loads is None:
        return []
    
    num_edges, num_buckets = edge_time_loads.shape
    scaled_bw = graph_data.adjacency_bandwidth * bandwidth_scale
    
    peak_hours = []
    
    for bucket_idx in range(num_buckets):
        total_overload_penalty = 0.0
        overload_count = 0
        
        for edge_idx in range(num_edges):
            load = edge_time_loads[edge_idx, bucket_idx]
            if load <= 0:
                continue
            
            u_idx = int(edge_u[edge_idx])
            v_idx = int(edge_v[edge_idx])
            bw = scaled_bw[u_idx, v_idx]
            if bw <= 0:
                bw = 1.0
            
            if load > bw:
                ratio = load / bw
                penalty = np.exp(1.0 * (min(ratio, 12.0) - 1.0))
                total_overload_penalty += penalty
                overload_count += 1
        
        if overload_count > 0:
            peak_hours.append({
                "minute": bucket_idx,
                "count": overload_count,
                "total_penalty": total_overload_penalty
            })
    
    peak_hours.sort(key=lambda x: x["total_penalty"], reverse=True)
    return peak_hours

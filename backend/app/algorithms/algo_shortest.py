"""
Shortest path algorithm using Dijkstra.
"""

from __future__ import annotations
import numpy as np

from .registry import register
from .dijkstra import dijkstra_shortest_path
from .cost_evaluator import compute_edge_time_usage, evaluate_path_congestion


@register("shortest")
def solve_shortest_path(graph_data, requests, config):
    """
    Dijkstra shortest path without penalties.
    
    Args:
        graph_data: GraphData instance
        requests: list of dicts with keys: origin_node, destination_node, request_id, start_time
        config: dict with keys: bandwidth_scale, k_paths (optional)
    
    Returns:
        dict with keys: paths, edge_stats, timeline, edge_time_usage, peak_hours, failed_requests
    """
    path_models = []
    failed_requests = []
    
    # Run Dijkstra for each request
    for req in requests:
        origin_idx = graph_data.node_id_to_index(req["origin_node"])
        dest_idx = graph_data.node_id_to_index(req["destination_node"])
        
        travel_time, path_indices = dijkstra_shortest_path(
            graph_data.adjacency_travel_time, origin_idx, dest_idx
        )
        
        node_path = graph_data.path_indices_to_node_ids(list(path_indices))
        if not node_path or np.isinf(travel_time):
            failed_requests.append(req["request_id"])
            continue
        
        path_models.append({
            "node_ids": node_path,
            "travel_time": float(travel_time),
            "base_travel_time": float(travel_time),
            "penalty_delay": 0.0,
            "request_id": req["request_id"],
            "start_time": req.get("start_time", 0),
        })
    
    if not path_models:
        return {
            "paths": [],
            "edge_stats": [],
            "timeline": [],
            "edge_time_usage": {},
            "peak_hours": [],
            "failed_requests": failed_requests,
        }
    
    # Evaluate congestion for cost calculation
    time_bucket = 60.0
    edge_time_usage, max_buckets = compute_edge_time_usage(
        graph_data, path_models, time_bucket
    )
    evaluate_path_congestion(
        graph_data, path_models, edge_time_usage,
        time_bucket, config.get("bandwidth_scale", 1.0)
    )
    
    # Import helpers from main
    from ..main import _build_route_response
    
    result = _build_route_response(
        graph_data, path_models, failed_requests,
        config.get("bandwidth_scale", 1.0),
        time_bucket=time_bucket
    )
    
    # Extract peak hours
    peak_hours = _extract_peak_hours_from_usage(
        edge_time_usage, config.get("bandwidth_scale", 1.0), graph_data
    )
    result["peak_hours"] = peak_hours
    
    return result


def _extract_peak_hours_from_usage(edge_time_usage, bandwidth_scale, graph_data):
    """Extract peak hours from edge_time_usage dict."""
    if not edge_time_usage:
        return []
    
    # Determine max buckets
    all_loads = list(edge_time_usage.values())
    num_buckets = max(len(loads) for loads in all_loads) if all_loads else 0
    
    peak_hours = []
    for bucket_idx in range(num_buckets):
        total_overload_penalty = 0.0
        overload_count = 0
        
        for edge_key, loads in edge_time_usage.items():
            if bucket_idx >= len(loads):
                continue
            load = loads[bucket_idx]
            if load <= 0:
                continue
            
            try:
                u, v = graph_data.parse_edge_key(edge_key)
                bw = graph_data.edge_bandwidth(u, v) * bandwidth_scale
                if bw <= 0:
                    bw = 1.0
            except:
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

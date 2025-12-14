import numpy as np
import os
from ..graph_loader import GraphData

def compute_edge_time_usage(
    graph_data: GraphData,
    path_models: list[dict],
    time_bucket: float = 60.0,
) -> tuple[dict, int]:
    """
    Compute edge_time_usage from path_models for dynamic visualization.
    Works for both shortest and penalty algorithms.
    
    Returns:
        edge_time_usage: dict {edge_key: [load_at_bucket_0, load_at_bucket_1, ...]}
        max_buckets: number of time buckets
    """
    if not path_models:
        return {}, 0
    
    # Determine max time to set bucket count
    max_end_time = 0.0
    for p in path_models:
        t = (p.get("start_time", 0) + p.get("travel_time", 0))
        if np.isfinite(t) and t < 1e9: # Sanity check
            max_end_time = max(max_end_time, t)
            
    # Cap max buckets to prevent OverflowError
    HARD_LIMIT_BUCKETS = 24 * 60
    calculated_buckets = int(np.ceil(max_end_time / time_bucket)) + 1
    
    if calculated_buckets > HARD_LIMIT_BUCKETS:
        print(f"[WARNING] max_buckets {calculated_buckets} exceeds limit {HARD_LIMIT_BUCKETS}. Capping.", flush=True)
        max_buckets = HARD_LIMIT_BUCKETS
    else:
        max_buckets = calculated_buckets
    
    print(f"[DEBUG] compute_edge_time_usage:", flush=True)
    print(f"  - time_bucket = {time_bucket} seconds", flush=True)
    print(f"  - max_end_time = {max_end_time} seconds", flush=True)
    print(f"  - max_buckets = {max_buckets}", flush=True)
    
    # Initialize edge loads
    edge_loads: dict[str, list] = {}
    
    for idx, path in enumerate(path_models):
        nodes = path.get("node_ids", [])
        start_time = path.get("start_time", 0)
        travel_time = path.get("travel_time", 0)
        
        if len(nodes) < 2 or travel_time <= 0:
            continue
        
        # Estimate time per edge (proportional to path)
        num_edges = len(nodes) - 1
        time_per_edge = travel_time / num_edges
        
        current_time = start_time
        for i in range(num_edges):
            u, v = nodes[i], nodes[i+1]
            key = f"{u}_{v}"
            
            if key not in edge_loads:
                edge_loads[key] = [0.0] * max_buckets
            
            # Mark load in all buckets this edge occupies
            start_bucket = int(current_time // time_bucket)
            end_bucket = int((current_time + time_per_edge) // time_bucket)
            
            for b in range(start_bucket, min(end_bucket + 1, max_buckets)):
                edge_loads[key][b] += 1
            
            current_time += time_per_edge

    # Write debug files
    debug_dir = os.path.join(os.path.dirname(__file__), "..", "..", "debug_output")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Write paths_info.csv
    paths_path = os.path.join(debug_dir, "paths_info.csv")
    try:
        with open(paths_path, "w") as f:
            f.write("path_idx,request_id,start_time,travel_time,num_nodes,num_edges,time_per_edge\n")
            for idx, path in enumerate(path_models):
                nodes = path.get("node_ids", [])
                start_time = path.get("start_time", 0)
                travel_time = path.get("travel_time", 0)
                num_edges = len(nodes) - 1 if len(nodes) > 1 else 0
                time_per_edge = travel_time / num_edges if num_edges > 0 else 0
                f.write(f"{idx},{path.get('request_id', idx)},{start_time},{travel_time},"
                        f"{len(nodes)},{num_edges},{time_per_edge:.2f}\n")
    except Exception as e:
        print(f"[DEBUG] Failed to write paths_info.csv: {e}", flush=True)
    
    return edge_loads, max_buckets


def evaluate_path_congestion(
    graph_data: GraphData,
    path_models: list[dict],
    edge_time_usage: dict[str, list],
    time_bucket: float,
    bandwidth_scale: float,
    penalty_beta: float = 1.0,
):
    """
    Evaluate actual congestion costs (penalty) for a set of paths based on computed loads.
    Updates path_models in-place with 'penalty_delay' and 'travel_time'.
    """
    for path in path_models:
        nodes = path.get("node_ids", [])
        if len(nodes) < 2:
            continue
            
        start_time = path.get("start_time", 0)
        base_travel_time = path.get("base_travel_time", path.get("travel_time", 0))
        current_time = start_time
        total_penalty = 0.0
        
        # Recalculate time per edge approx
        travel_time = base_travel_time
        num_edges = len(nodes) - 1
        time_per_edge = travel_time / num_edges if num_edges > 0 else 0
        
        path["congestion_times"] = []
        
        for i in range(num_edges):
            u, v = nodes[i], nodes[i+1]
            key = f"{u}_{v}"
            
            # Get bandwidth
            bw = graph_data.edge_bandwidth(u, v) * bandwidth_scale
            if bw <= 0: bw = 1.0
            
            # Simple check at moment of entry (ceil to avoid Minute 0 false positives)
            bucket = int(np.ceil(current_time / time_bucket)) if current_time > 0 else 0
            loads = edge_time_usage.get(key)
            if loads and bucket < len(loads):
                load = loads[bucket]
                ratio = load / bw
                if ratio > 1.01:
                    # User Request: Time Bucket = 1 minute fixed.
                    # Ratio definition: Load / Capacity. 
                    # If ratio > 1, it means congestion.
                    
                    # Log ALL penalty events for debugging (use minute, not seconds)
                    request_id = path.get("request_id", "?")
                    print(f"[PENALTY] Trip#{request_id} Edge={key} Minute={bucket} Load={load} Cap={bw:.2f} Ratio={ratio:.2f}", flush=True)

                    # Clamp effective ratio for exponential safety
                    effective_ratio = min(ratio, 12.0)
                    penalty = np.exp(penalty_beta * (effective_ratio - 1.0))
                    
                    # Hard cap
                    MAX_PENALTY = 86400.0   
                    if penalty > MAX_PENALTY: 
                        penalty = MAX_PENALTY
                        
                    total_penalty += penalty
                    path["congestion_times"].append({
                        "time": current_time,
                        "delay": penalty,
                        "ratio": ratio
                    })
            
            current_time += time_per_edge
            
        path["penalty_delay"] = total_penalty
        path["travel_time"] = base_travel_time + total_penalty

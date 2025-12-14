from __future__ import annotations

from collections import defaultdict
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from .config import get_settings
from .graph_loader import GraphData, get_graph_data
from .algorithms import (
    dijkstra_shortest_path, 
    solve_routing_with_penalty_lazy,
    evaluate_path_congestion,
    compute_edge_time_usage,
    get_config,
    update_config
)



def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    return app


app = create_app()


def _get_graph_data() -> GraphData:
    return get_graph_data()


@app.route("/graph/edges", methods=["GET"])
def graph_edges():
    graph_data = _get_graph_data()
    geojson = graph_data.to_geojson
    return jsonify(geojson)


@app.route("/graph/nodes", methods=["GET"])
def graph_nodes():
    graph_data = _get_graph_data()
    geojson = graph_data.nodes_geojson
    return jsonify(geojson)


@app.route("/graph/refresh", methods=["POST"])
def refresh_graph_data():
    """Clear the graph data cache (RAM & Disk) to reload from disk."""
    try:
        current_graph = get_graph_data()
        current_graph.clear_disk_cache()
    except Exception as e:
        print(f"[WARN] Could not clear disk cache: {e}")

    get_graph_data.cache_clear()
    new_graph = get_graph_data()
    _ = new_graph.nodes_geojson
    _ = new_graph.to_geojson

    return jsonify({
        "status": "refreshed",
        "message": "Graph data reloaded from CSV. Disk cache cleared and regenerated."
    })


@app.route("/graph/nearest-node", methods=["POST"])
def nearest_node():
    graph_data = _get_graph_data()
    data = request.get_json()
    lat = data.get("lat")
    lon = data.get("lon")
    if lat is None or lon is None:
        return jsonify({"error": "lat and lon are required"}), 400

    node_id = graph_data.nearest_node(lat, lon)
    coords = graph_data.node_coords(node_id)
    if coords is None:
        return jsonify({"error": "Node coordinates not found"}), 404
    lon_out, lat_out = coords
    return jsonify({"node_id": node_id, "lat": lat_out, "lon": lon_out})


@app.route("/generate-fake-trips", methods=["POST"])
def generate_fake_trips():
    """Generate random fake trips using numpy for efficiency."""
    graph_data = _get_graph_data()
    data = request.get_json() or {}
    
    count = data.get("count", 10)
    max_start = data.get("max_start_time", 300)  # Default 5 minutes (300 seconds)

    node_ids = np.array(graph_data.node_ids, dtype=np.int64)
    n_nodes = len(node_ids)
    if n_nodes < 2:
        return jsonify({"error": "Need at least 2 nodes to generate trips"}), 400

    rng = np.random.default_rng()
    origin_indices = rng.integers(0, n_nodes, size=count)
    dest_indices = rng.integers(0, n_nodes, size=count)

    same_mask = origin_indices == dest_indices
    retries = 0
    while same_mask.any() and retries < 5:
        dest_indices[same_mask] = rng.integers(0, n_nodes, size=same_mask.sum())
        same_mask = origin_indices == dest_indices
        retries += 1

    valid_mask = origin_indices != dest_indices
    origin_indices = origin_indices[valid_mask]
    dest_indices = dest_indices[valid_mask]
    # All vehicles start at time 0
    start_times = np.zeros(len(origin_indices))

    trips = []
    for i in range(len(origin_indices)):
        trips.append({
            "request_id": i + 1,
            "origin_node": int(node_ids[origin_indices[i]]),
            "destination_node": int(node_ids[dest_indices[i]]),
            "start_time": round(float(start_times[i]), 2),
        })

    return jsonify({"trips": trips})


@app.route("/config", methods=["GET"])
def get_algorithm_config():
    """Get current algorithm configuration."""
    cfg = get_config()
    return jsonify(cfg.model_dump())


@app.route("/config", methods=["POST"])
def update_algorithm_config():
    """Update algorithm configuration."""
    data = request.get_json() or {}
    updated = update_config(**data)
    return jsonify(updated.model_dump())


@app.route("/route", methods=["POST"])
def compute_route():
    """Compute routes using registered algorithms."""
    graph_data = _get_graph_data()
    data = request.get_json()
    
    print(f"[DEBUG] /route called with data keys: {list(data.keys()) if data else 'None'}", flush=True)
    
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    # Extract parameters
    algorithm = data.get("algorithm", "shortest")
    bandwidth_scale = float(data.get("bandwidth_scale", 1.0))
    k_paths = int(data.get("k_paths", 3))
    requests_list = data.get("requests")
    
    print(f"[DEBUG] algorithm={algorithm}, num_requests={len(requests_list) if requests_list else 0}", flush=True)
    
    # Parse requests into standardized format
    vehicle_requests = []
    if requests_list and len(requests_list) > 0:
        for req in requests_list:
            # Validate nodes exist
            if not graph_data.node_exists(req["origin_node"]):
                return jsonify({"error": f"Node {req['origin_node']} not found"}), 400
            if not graph_data.node_exists(req["destination_node"]):
                return jsonify({"error": f"Node {req['destination_node']} not found"}), 400
            
            vehicle_requests.append({
                "request_id": req["request_id"],
                "origin_node": req["origin_node"],
                "destination_node": req["destination_node"],
                "start_time": req.get("start_time", data.get("start_time", 0.0)),
            })
    else:
        # Single request format
        origin_node = data.get("origin_node")
        destination_node = data.get("destination_node")
        if origin_node is None or destination_node is None:
            return jsonify({"error": "origin_node and destination_node are required"}), 400
        
        if not graph_data.node_exists(origin_node):
            return jsonify({"error": f"Node {origin_node} not found"}), 400
        if not graph_data.node_exists(destination_node):
            return jsonify({"error": f"Node {destination_node} not found"}), 400
        
        vehicle_requests.append({
            "request_id": 1,
            "origin_node": origin_node,
            "destination_node": destination_node,
            "start_time": data.get("start_time", 0.0),
        })
    
    print(f"[DEBUG] Validated {len(vehicle_requests)} requests", flush=True)
    
    # Prepare config
    cfg = get_config()
    config = {
        "bandwidth_scale": bandwidth_scale,
        "k_paths": k_paths,
        "penalty_beta": cfg.penalty_beta,
        "k_shortest_penalty": cfg.k_shortest_penalty,
    }
    
    # Get algorithm from registry and execute
    try:
        from .algorithms import get as get_algorithm
        solve_func = get_algorithm(algorithm)
        print(f"[DEBUG] Executing algorithm: {algorithm}", flush=True)
        result = solve_func(graph_data, vehicle_requests, config)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"[ERROR] Algorithm execution failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Algorithm execution failed: {str(e)}"}), 500








    algorithm = data.get("algorithm", "shortest")
    bandwidth_scale = float(data.get("bandwidth_scale", 1.0))
    k_paths = int(data.get("k_paths", 3))
    requests_list = data.get("requests")
    
    print(f"[DEBUG] algorithm={algorithm}, num_requests={len(requests_list) if requests_list else 0}", flush=True)
    print(f"[DEBUG] bandwidth_scale={bandwidth_scale} (Should be 1.0 ideally)", flush=True)
    if requests_list and len(requests_list) > 0:
        print(f"[DEBUG] First request: {requests_list[0]}", flush=True)
    
    # Normalize requests
    vehicle_requests = []
    if requests_list:
        for idx, req in enumerate(requests_list, start=1):
            vehicle_requests.append({
                "request_id": req.get("request_id", idx),
                "origin_node": req["origin_node"],
                "destination_node": req["destination_node"],
                "start_time": req.get("start_time", data.get("start_time", 0.0)),
            })
    else:
        origin_node = data.get("origin_node")
        destination_node = data.get("destination_node")
        if origin_node is None or destination_node is None:
            return jsonify({"error": "origin_node and destination_node are required"}), 400
        vehicle_requests.append({
            "request_id": 1,
            "origin_node": origin_node,
            "destination_node": destination_node,
            "start_time": data.get("start_time", 0.0),
        })

    origin_indices = []
    dest_indices = []
    request_ids = []
    start_times = []

    for vr in vehicle_requests:
        origin_node = vr["origin_node"]
        dest_node = vr["destination_node"]
        
        if not graph_data.node_exists(origin_node):
            print(f"[DEBUG] Origin node {origin_node} not found!", flush=True)
            return jsonify({"error": f"Node {origin_node} not found"}), 400
        if not graph_data.node_exists(dest_node):
            print(f"[DEBUG] Dest node {dest_node} not found!", flush=True)
            return jsonify({"error": f"Node {dest_node} not found"}), 400
            
        origin_indices.append(graph_data.node_id_to_index(origin_node))
        dest_indices.append(graph_data.node_id_to_index(dest_node))
        request_ids.append(vr["request_id"])
        start_times.append(float(vr["start_time"]))

    print(f"[DEBUG] Validated {len(origin_indices)} requests, first pair: {origin_indices[0] if origin_indices else 'N/A'} -> {dest_indices[0] if dest_indices else 'N/A'}", flush=True)

    path_models = []
    failed_requests = []

    if algorithm == "shortest":
        # Debug: check adjacency matrix
        adj = graph_data.adjacency_travel_time
        print(f"[DEBUG] Adjacency matrix shape: {adj.shape}, nnz (non-zero entries): {adj.nnz}", flush=True)
        if adj.nnz > 0:
            # Check a few sample values
            sample_row = adj[origin_indices[0], :].toarray().flatten()
            non_zero_count = (sample_row > 0).sum()
            print(f"[DEBUG] Sample row {origin_indices[0]}: {non_zero_count} neighbors", flush=True)
        
        print(f"[DEBUG] Running Dijkstra for {len(origin_indices)} pairs...", flush=True)
        for i, (origin_idx, dest_idx) in enumerate(zip(origin_indices, dest_indices)):
            travel_time, path_indices = dijkstra_shortest_path(
                graph_data.adjacency_travel_time, origin_idx, dest_idx
            )
            if i == 0:
                print(f"[DEBUG] First route: travel_time={travel_time}, path_len={len(path_indices)}", flush=True)
            node_path = graph_data.path_indices_to_node_ids(list(path_indices))
            if not node_path or np.isinf(travel_time):
                failed_requests.append(request_ids[i])
                continue
            path_models.append({
                "node_ids": node_path,
                "travel_time": float(travel_time),
                "base_travel_time": float(travel_time),
                "penalty_delay": 0.0,
                "request_id": request_ids[i],
                "start_time": start_times[i],
            })
        print(f"[DEBUG] Shortest: path_models={len(path_models)}, failed={len(failed_requests)}", flush=True)
        if not path_models:
            return jsonify({"error": "No path found for any request"}), 404
            
        # Evaluate congestion cost (User Request: Cost depends on traffic, not just algo)
        time_bucket = 60.0
        edge_time_usage, max_buckets = compute_edge_time_usage(
            graph_data, path_models, time_bucket
        )
        evaluate_path_congestion(
            graph_data, path_models, edge_time_usage, time_bucket, bandwidth_scale
        )
            
        # Convert edge_time_usage generic dict to numpy-like arrays for response helper
        # Accessing private _build_route_response expects arrays if provided, OR we can pass nothing
        # But _build_route_response calculates edge_time_usage if not provided. 
        # However, we just calculated it. Let's let the helper re-use logic or just pass path_models which now have updated travel_time.
        # Actually, `_build_route_response` calls `_compute_edge_time_usage_from_paths` if we don't pass `edge_time_loads`.
        # BUT, we want the UPDATED path_models (with Total Time) to be used.
        # The frontend visualization needs edge_time_usage.
        
        # We need to construct the args for _build_route_response manually or let it recompute.
        # Recomputing is safe and cheap enough for frontend viz.
        
        # Extract peak_hours by checking edge loads at each minute
        # User feedback: "t=0 chưa ai ra khỏi nhà" - use actual edge loads, not path start times
        peak_hours = []
        if edge_time_usage:
            # Determine max buckets from the data
            all_loads = list(edge_time_usage.values())
            num_buckets = max(len(loads) for loads in all_loads) if all_loads else 0
            
            for bucket_idx in range(num_buckets):
                # Check all edges at this bucket
                total_overload_penalty = 0.0
                overload_count = 0
                
                for edge_key, loads in edge_time_usage.items():
                    if bucket_idx >= len(loads):
                        continue
                    load = loads[bucket_idx]
                    if load <= 0:
                        continue
                    
                    # Get bandwidth for this edge
                    try:
                        u, v = graph_data.parse_edge_key(edge_key)
                        bw = graph_data.edge_bandwidth(u, v) * bandwidth_scale
                        if bw <= 0:
                            bw = 1.0
                    except:
                        bw = 1.0
                    
                    # Check if overloaded
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
        
        resp = _build_route_response(
            graph_data, path_models, failed_requests, bandwidth_scale,
            time_bucket=time_bucket
        )
        resp["peak_hours"] = peak_hours
        return jsonify(resp)

    if algorithm == "penalty":
        cfg = get_config()
        
        # Determine time bucket (default 60s)
        time_bucket = 60.0
        
        # Bandwidth is already in Per-Minute units (from graph_loader)
        # So we just apply the tuning scale factor.
        scaled_bandwidth = graph_data.adjacency_bandwidth * bandwidth_scale
        
        origin_array = np.array(origin_indices, dtype=np.int64)
        dest_array = np.array(dest_indices, dtype=np.int64)
        start_times_array = np.array(start_times, dtype=np.float64)
        
        print(f"[DEBUG] Calling penalty routing: origins={origin_array[:3]}... dests={dest_array[:3]}...", flush=True)
        
        routes, edge_time_loads, edge_u, edge_v, _, max_buckets, routes_metadata = solve_routing_with_penalty_lazy(
            graph_data.adjacency_travel_time,
            scaled_bandwidth,
            origins=origin_array,
            destinations=dest_array,
            start_times=start_times_array,
            k_paths=k_paths,
            large_penalty_for_alt=cfg.k_shortest_penalty,
            penalty_beta=cfg.penalty_beta,
            max_workers=4,  # Parallel workers
        )
        
        print(f"[DEBUG] Penalty routing returned routes shape: {routes.shape}", flush=True)
        
        node_paths = graph_data.routes_indices_to_node_ids(routes)
        
        print(f"[DEBUG] Converted to node_paths, count={len(node_paths)}, first path len={len(node_paths[0]) if node_paths else 0}", flush=True)
        
        if len(node_paths) != len(request_ids):
            return jsonify({"error": "Route solver returned unexpected results"}), 500
            
        for i, node_path in enumerate(node_paths):
            if not node_path:
                failed_requests.append(request_ids[i])
                continue
            
            # Use metadata from penalty routing if available
            meta = routes_metadata[i] if i < len(routes_metadata) else {}
            
            # Use total_travel_time (base + penalty) if available, otherwise fallback
            travel_time = meta.get("total_travel_time")
            if travel_time is None:
                travel_time = graph_data.path_travel_time(node_path)
            
            path_models.append({
                "node_ids": node_path,
                "travel_time": travel_time,
                "base_travel_time": meta.get("base_travel_time", travel_time),
                "penalty_delay": meta.get("penalty_delay", 0.0),
                "congestion_times": meta.get("congestion_times", []),
                "request_id": request_ids[i],
                "start_time": start_times[i],
            })
        
        print(f"[DEBUG] path_models count={len(path_models)}, failed_requests={len(failed_requests)}", flush=True)
        
        # Aggregate congestion stats for global "peak hours"
        # User feedback: Use actual edge loads at each bucket, not path congestion_times
        peak_hours = []
        
        if edge_time_loads is not None and edge_u is not None:
            num_edges, num_buckets = edge_time_loads.shape
            scaled_bw = graph_data.adjacency_bandwidth * bandwidth_scale
            
            for bucket_idx in range(num_buckets):
                total_overload_penalty = 0.0
                overload_count = 0
                
                for edge_idx in range(num_edges):
                    load = edge_time_loads[edge_idx, bucket_idx]
                    if load <= 0:
                        continue
                    
                    # Get bandwidth for this edge
                    u_idx = int(edge_u[edge_idx])
                    v_idx = int(edge_v[edge_idx])
                    bw = scaled_bw[u_idx, v_idx]
                    if bw <= 0:
                        bw = 1.0
                    
                    # Check if overloaded
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
        
        # Write peak_hours.csv
        import os
        debug_dir = os.path.join(os.path.dirname(__file__), "..", "..", "debug_output")
        os.makedirs(debug_dir, exist_ok=True)
        peak_path = os.path.join(debug_dir, "peak_hours.csv")
        with open(peak_path, "w") as f:
            f.write("minute,congestion_count,total_penalty_delay\n")
            for p in peak_hours:
                f.write(f"{p['minute']},{p['count']},{p['total_penalty']:.2f}\n")
        print(f"[DEBUG] Wrote peak_hours.csv to {peak_path} ({len(peak_hours)} entries)", flush=True)

        if not path_models:
            return jsonify({"error": "No route generated for any request"}), 404
        
        # Build response with edge_time_usage for dynamic visualization
        resp = _build_route_response(
            graph_data, path_models, failed_requests, bandwidth_scale,
            edge_time_loads=edge_time_loads,
            edge_u=edge_u,
            edge_v=edge_v,
            time_bucket=time_bucket,
            max_buckets=max_buckets,
        )
        resp["peak_hours"] = peak_hours
        return jsonify(resp)

    return jsonify({"error": "Unsupported algorithm"}), 400


def _build_route_response(
    graph_data: GraphData,
    path_models: list[dict],
    failed_requests: list[int],
    bandwidth_scale: float = 1.0,
    edge_time_loads: np.ndarray = None,
    edge_u: np.ndarray = None,
    edge_v: np.ndarray = None,
    time_bucket: float = 60.0,
    max_buckets: int = 120,
) -> dict:
    
    # Standardize edge_time_usage creation
    final_edge_time_usage = {}
    
    if edge_time_loads is not None and edge_u is not None and edge_v is not None:
        # From Penalty Algorithm (Numpy arrays)
        for i in range(len(edge_u)):
            loads = edge_time_loads[i].tolist()
            if any(l > 0 for l in loads):
                u_node = graph_data.index_to_node_id(int(edge_u[i]))
                v_node = graph_data.index_to_node_id(int(edge_v[i]))
                key = f"{u_node}_{v_node}"
                final_edge_time_usage[key] = loads
        final_time_bucket = time_bucket
    else:
        # From Shortest Path (Compute now)
        final_edge_time_usage, _ = compute_edge_time_usage(
            graph_data, path_models, time_bucket=60.0
        )
        final_time_bucket = 60.0

    # Build stats using Peak Load (Time-aware)
    edge_stats = _build_edge_usage_stats(graph_data, path_models, final_edge_time_usage, bandwidth_scale)
    timeline = _build_timeline(path_models)
    
    response = {
        "paths": path_models,
        "edge_stats": edge_stats,
        "timeline": timeline,
        "failed_requests": failed_requests,
        "edge_time_usage": final_edge_time_usage,
        "time_bucket_seconds": final_time_bucket,
    }
    
    # Calculate actual max time
    actual_max_seconds = 0.0
    if path_models:
        for p in path_models:
            end_time = (p.get("start_time") or 0) + (p.get("travel_time") or 0)
            if end_time > actual_max_seconds:
                actual_max_seconds = end_time
    response["max_time_minutes"] = int(np.ceil(actual_max_seconds / 60))
    
    return response


def _build_edge_usage_stats(
    graph_data: GraphData,
    path_models: list[dict],
    edge_time_usage: dict[str, list],
    bandwidth_scale: float = 1.0,
) -> list[dict]:
    # User feedback: "Old visualization was correct".
    # Reverting to counting total trips per edge.
    
    usage_counter: defaultdict[str, int] = defaultdict(int)
    for path in path_models:
        nodes = path["node_ids"]
        for u, v in zip(nodes, nodes[1:]):
            key = graph_data.edge_key(u, v)
            usage_counter[key] += 1

    stats = []
    
    # We can still use edge_time_usage to get peak load if we want extra info, 
    # but the primary "used_count" should be the Trip Count.
    
    for key, count in usage_counter.items():
        source, target = graph_data.parse_edge_key(key)
        capacity = graph_data.edge_bandwidth(source, target)
        effective_b = (capacity * bandwidth_scale) if capacity else None
        
        # Ratio used in "Old Visualize": Total Count / Bandwidth (Dimensionally weird, but requested)
        ratio = (count / effective_b) if effective_b and effective_b > 0 else None
        
        stats.append({
            "edge_key": key,
            "source": source,
            "target": target,
            "bandwidth_capacity": capacity,
            "effective_bandwidth": effective_b,
            "used_count": count,
            "usage_ratio": ratio,
        })

    stats.sort(key=lambda item: item["used_count"], reverse=True)
    return stats


def _build_timeline(path_models: list[dict]) -> list[dict]:
    timeline = []
    for idx, path in enumerate(path_models, start=1):
        request_id = path.get("request_id", idx)
        start_time = float(path.get("start_time", 0.0))
        travel_time = float(path.get("travel_time", 0.0))
        timeline.append({
            "request_id": request_id,
            "start_time": start_time,
            "travel_time": travel_time,
            "end_time": start_time + travel_time,
        })

    timeline.sort(key=lambda item: (item["start_time"], item["request_id"]))
    return timeline


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

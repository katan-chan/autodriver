from __future__ import annotations

from collections import defaultdict
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from .config import get_settings
from .graph_loader import GraphData, get_graph_data
from .algorithms import (
    get_config,
    update_config,
    compute_edge_time_usage,
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


@app.route("/algorithms", methods=["GET"])
def list_algorithms():
    """List all available routing algorithms."""
    from .algorithms import list_all
    return jsonify({"algorithms": list_all()})


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

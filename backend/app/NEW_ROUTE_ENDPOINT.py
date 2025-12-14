"""
Clean /route endpoint implementation using registry system.

INSTRUCTIONS:
1. Open d:/autodriver/backend/app/main.py
2. Find the @app.route("/route", methods=["POST"]) function
3. Replace the ENTIRE function (from @app.route to the line before the next @app.route or def)
4. Paste this code
"""

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
    from .algorithms import get_config
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

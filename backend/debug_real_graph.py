
import sys
import os
import numpy as np
import time

# Ensure current dir (backend) is in path
sys.path.append(os.getcwd())

from app.graph_loader import get_graph_data
from app.algorithms.penalty_routing import solve_routing_with_penalty
from app.algorithms.config import get_config

def test_real_graph():
    print("Loading Real Graph...", flush=True)
    t0 = time.time()
    graph_data = get_graph_data()
    print(f"Graph loaded in {time.time() - t0:.2f}s. Nodes: {graph_data.num_nodes}", flush=True)
    
    # Generate 20 random trips
    n_trips = 20
    node_ids = np.array(graph_data.node_ids, dtype=np.int64)
    n_nodes = len(node_ids)
    
    rng = np.random.default_rng(42) # Seed for reproducibility
    origin_indices_raw = rng.integers(0, n_nodes, size=n_trips)
    dest_indices_raw = rng.integers(0, n_nodes, size=n_trips)
    
    # Ensure they are different
    mask = origin_indices_raw != dest_indices_raw
    origin_indices_raw = origin_indices_raw[mask]
    dest_indices_raw = dest_indices_raw[mask]
    
    origins = origin_indices_raw
    destinations = dest_indices_raw
    start_times = np.zeros(len(origins), dtype=np.float64) # All start at 0
    
    print(f"Running solver for {len(origins)} trips...", flush=True)
    cfg = get_config()
    k_paths = 3
    
    t1 = time.time()
    try:
        routes = solve_routing_with_penalty(
            graph_data.adjacency_travel_time,
            graph_data.adjacency_bandwidth, # Use raw bandwidth for now, mimic scale=1
            origins=origins,
            destinations=destinations,
            start_times=start_times,
            k_paths=k_paths,
            large_penalty_for_alt=cfg.k_shortest_penalty,
            penalty_beta=cfg.penalty_beta
        )
        print(f"Solver finished in {time.time() - t1:.2f}s", flush=True)
        print("Routes shape:", routes.shape, flush=True)
        print("Success!", flush=True)
    except Exception as e:
        print("Solver CRASHED:", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_graph()

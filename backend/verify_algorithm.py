
import sys
import os
import numpy as np
from scipy import sparse

# Ensure current dir (backend) is in path
sys.path.append(os.getcwd())

from app.algorithms.penalty_routing import solve_routing_with_penalty

def test_algorithm():
    print("Testing solve_routing_with_penalty...")
    
    # 1. Mock Graph Data
    # 3 nodes: 0 -> 1 -> 2
    # Edge 0->1: bandwidth=1, time=10
    # Edge 1->2: bandwidth=1, time=10
    
    n_nodes = 3
    # Adjacency Travel Time (0->1: 10, 1->2: 10)
    # CSR format
    row = np.array([0, 1])
    col = np.array([1, 2])
    data_time = np.array([10.0, 10.0])
    adj_time = sparse.coo_matrix((data_time, (row, col)), shape=(n_nodes, n_nodes)).tocsr()
    
    # Adjacency Bandwidth (Capacity = 1.0)
    data_bw = np.array([1.0, 1.0])
    adj_bw = sparse.coo_matrix((data_bw, (row, col)), shape=(n_nodes, n_nodes)).tocsr()
    
    # 2. Mock Request
    # 2 Vehicles
    # V1: 0->2, start=0
    # V2: 0->2, start=0 (Should conflict if same bucket?)
    origins = np.array([0, 0])
    destinations = np.array([2, 2])
    start_times = np.array([0.0, 0.0])
    
    try:
        routes = solve_routing_with_penalty(
            adjacency_travel_time=adj_time,
            adjacency_bandwidth=adj_bw,
            origins=origins,
            destinations=destinations,
            start_times=start_times,
            k_paths=3,
            large_penalty_for_alt=100.0,
            penalty_beta=1.0
        )
        print("Algorithm executed successfully!")
        print("Routes shape:", routes.shape)
        print("Routes:\n", routes)
        
        # Verify routes are 0->1->2 (which is [0, 1, 2])
        # Note: output might be padded with -1 or similar depending on implementation
        # The implementation returns 'routes_final' from 'build_routes_from_choice'
        # which returns (n_vehicles, n_nodes) array.
        
        expected = np.array([0, 1, 2])
        if np.all(routes[0, :3] == expected):
             print("Route logic seems correct (0->1->2 found).")
        else:
             print("Warning: unexpected route path.")
             
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_algorithm()

"""
Edge utility functions for routing algorithms.

Supports scipy sparse matrices.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def build_edge_list_and_index(
    adjacency_bandwidth,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build edge list and index from adjacency bandwidth matrix.

    Assumes undirected graph; only considers unique edges.

    Parameters
    ----------
    adjacency_bandwidth : sparse matrix or np.ndarray
        (n_nodes, n_nodes) bandwidth matrix; 0 means no edge.

    Returns
    -------
    edge_u : np.ndarray
        (m,) source node of each edge.
    edge_v : np.ndarray
        (m,) target node of each edge.
    edge_bandwidth : np.ndarray
        (m,) bandwidth capacity of each edge.
    edge_index_dict : dict
        {(u, v): idx} mapping for quick lookup.
    """
    if sparse.issparse(adjacency_bandwidth):
        coo = adjacency_bandwidth.tocoo()
        rows, cols, data = coo.row, coo.col, coo.data
    else:
        rows, cols = np.nonzero(adjacency_bandwidth)
        data = adjacency_bandwidth[rows, cols]

    # Only keep edges where u < v (undirected, avoid duplicates)
    mask = rows < cols
    u_arr = rows[mask]
    v_arr = cols[mask]
    bw_arr = data[mask]

    edge_u = u_arr.astype(np.int64)
    edge_v = v_arr.astype(np.int64)
    edge_bandwidth = bw_arr.astype(np.float64)

    # Build CSR lookup instead of dict
    # We want a matrix M where M[u, v] = idx + 1 (so 0 means no edge)
    # We build it symmetrically
    
    # Concatenate (u, v, idx) and (v, u, idx)
    all_u = np.concatenate([edge_u, edge_v])
    all_v = np.concatenate([edge_v, edge_u])
    # logical index usually 0-based, but 0 in sparse matrix means empty. 
    # So we store idx+1.
    all_data = np.concatenate([np.arange(len(edge_u)), np.arange(len(edge_u))]) + 1
    
    n_nodes = adjacency_bandwidth.shape[0]
    csr = sparse.coo_matrix((all_data, (all_u, all_v)), shape=(n_nodes, n_nodes)).tocsr()
    
    return edge_u, edge_v, edge_bandwidth, csr.indptr, csr.indices, csr.data



def build_routes_from_choice(
    all_paths: np.ndarray,
    chosen_k: np.ndarray,
) -> np.ndarray:
    """
    Extract routes from all_paths based on chosen path index for each vehicle.

    Parameters
    ----------
    all_paths : np.ndarray
        Shape (n_vehicles, k_paths, n_nodes).
    chosen_k : np.ndarray
        Shape (n_vehicles,), index of chosen path [0..k_paths-1].

    Returns
    -------
    routes : np.ndarray
        Shape (n_vehicles, n_nodes), selected path for each vehicle.
    """
    n_vehicles = all_paths.shape[0]
    routes = all_paths[np.arange(n_vehicles), chosen_k, :]
    return routes.copy()


def compute_edge_loads_from_routes(
    routes: np.ndarray,
    edge_index_dict: dict,
    n_edges: int,
) -> np.ndarray:
    """
    Compute load (number of routes) on each edge.

    Parameters
    ----------
    routes : np.ndarray
        Shape (n_vehicles, n_nodes), each row is a path.
    edge_index_dict : dict
        {(u, v): idx} mapping.
    n_edges : int
        Total number of edges.

    Returns
    -------
    edge_loads : np.ndarray
        (n_edges,) count of routes using each edge.
    """
    edge_loads = np.zeros(n_edges, dtype=np.int64)

    for route in routes:
        for i in range(len(route) - 1):
            u = route[i]
            v = route[i + 1]
            if v < 0:
                break
            e_idx = edge_index_dict.get((u, v), -1)
            if e_idx >= 0:
                edge_loads[e_idx] += 1

    return edge_loads

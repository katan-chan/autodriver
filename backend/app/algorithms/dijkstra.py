"""
Dijkstra's shortest path algorithm using scipy sparse.

Supports both dense numpy arrays and scipy sparse matrices.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra


def dijkstra_shortest_path(
    adjacency_travel_time,
    source: int,
    target: int,
) -> tuple[float, np.ndarray]:
    """
    Dijkstra shortest path using scipy.sparse.csgraph.

    Parameters
    ----------
    adjacency_travel_time : np.ndarray or scipy.sparse matrix
        (n_nodes, n_nodes) matrix; entry is travel time, 0 or missing if no edge.
    source : int
        Source node index.
    target : int
        Target node index.

    Returns
    -------
    total_cost : float
        Shortest path cost (np.inf if unreachable).
    path : np.ndarray
        Shape (n_nodes,), sequence of node indices from source to target, padded with -1.
    """
    # Convert to csr if needed
    if sparse.issparse(adjacency_travel_time):
        graph = adjacency_travel_time.tocsr()
    else:
        # Dense array - convert inf to 0 for scipy (0 = no edge)
        adj_copy = adjacency_travel_time.copy()
        adj_copy[np.isinf(adj_copy)] = 0
        graph = sparse.csr_matrix(adj_copy)
    
    n_nodes = graph.shape[0]
    
    # Run dijkstra from source to target
    dist, predecessors = scipy_dijkstra(
        graph, 
        directed=False, 
        indices=source, 
        return_predecessors=True,
        limit=np.inf,
    )
    
    # Build path from predecessors
    path = np.full(n_nodes, -1, dtype=np.int64)
    
    if np.isinf(dist[target]):
        return float(dist[target]), path
    
    # Reconstruct path by backtracking
    tmp = []
    cur = target
    while cur != -9999 and cur != source:  # -9999 is scipy's "no predecessor"
        tmp.append(cur)
        cur = predecessors[cur]
    tmp.append(source)
    tmp.reverse()
    
    for i, node in enumerate(tmp):
        if i < n_nodes:
            path[i] = node
    
    return float(dist[target]), path

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from .config import get_settings


class GraphData:
    def __init__(self):
        self.settings = get_settings()
        self._loaded = False
        
        # Data storage
        self._graph: nx.Graph | None = None
        self._edge_df: pd.DataFrame | None = None
        self._node_df: pd.DataFrame | None = None
        self._node_ids: List[int] = []
        self._node_coords: Dict[int, Tuple[float, float]] = {}
        
        # Mappings
        self._node_index: Dict[int, int] = {}
        self._index_node: Dict[int, int] = {}
        self._edge_capacities: Dict[str, float] = {}
        
        # Matrices (CSR)
        self._adjacency_travel_time: sparse.csr_matrix | None = None
        self._adjacency_bandwidth: sparse.csr_matrix | None = None

    def ensure_loaded(self) -> None:
        if self._loaded:
            return

        print("[DEBUG] Loading heavy graph data (Lazy Load)...", flush=True)
        # 1. Load CSVs
        self._edge_df = pd.read_csv(self.settings.edge_csv_path)
        nodes_path = self.settings.edge_csv_path.with_name("hanoi_component_nodes.csv")
        if not nodes_path.exists():
            raise FileNotFoundError(f"Node CSV not found at {nodes_path}")
        self._node_df = pd.read_csv(nodes_path)

        # 2. Build NetworkX Graph (Keep this for structural checks/utils if needed)
        self._graph = nx.Graph()
        # Vectorized node addition is tricky with nx, but we can do bulk add
        # Actually for 10k nodes, iteration is okay-ish if simple, but let's be cleaner
        # We'll stick to iteration for NX as it's inevitable if using the library, 
        # but simpler loop.
        
        # Node processing - use int64 to prevent overflow on Windows (C int = 32-bit)
        node_ids = self._node_df["node_id"].astype(np.int64).tolist()
        lons = self._node_df["lon"].values
        lats = self._node_df["lat"].values
        
        self._node_ids = sorted(list(set(node_ids))) # Ensure sorted for consistency
        self._node_index = {node_id: idx for idx, node_id in enumerate(self._node_ids)}
        self._index_node = {idx: node_id for node_id, idx in self._node_index.items()}
        self._node_coords = {
            nid: (float(lon), float(lat)) 
            for nid, lon, lat in zip(node_ids, lons, lats) 
            if pd.notna(lon) and pd.notna(lat)
        }
        
        # Add to NX (Optional optimization: use add_nodes_from with attrs)
        # self._graph.add_nodes_from((n, {"x": x, "y": y}) for n, (x, y) in self._node_coords.items())
        # For compatibility with existing code that checks graph:
        for nid, (x, y) in self._node_coords.items():
            self._graph.add_node(nid, x=x, y=y)

        # 3. Vectorized Matrix Construction
        print("[DEBUG] Building adjacency matrices (Vectorized)...", flush=True)
        
        # Cast source/target to int64 to match _node_index keys (prevent int32 overflow on Windows)
        self._edge_df["source"] = self._edge_df["source"].astype(np.int64)
        self._edge_df["target"] = self._edge_df["target"].astype(np.int64)
        
        # Filter valid edges (both nodes exist in our component)
        valid_mask = (self._edge_df["source"].isin(self._node_index)) & \
                     (self._edge_df["target"].isin(self._node_index))
        df_valid = self._edge_df[valid_mask].copy()
        
        print(f"[DEBUG] Edge filtering: total={len(self._edge_df)}, valid={len(df_valid)}", flush=True)

        # Map to indices
        u_indices = df_valid["source"].map(self._node_index).astype(np.int32).values
        v_indices = df_valid["target"].map(self._node_index).astype(np.int32).values

        # Prepare weights
        # Travel Time
        travel_time = df_valid["travel_time_seconds"].fillna(0.0).values
        # Fallback for missing travel time: length / 10.0 (dummy speed)
        mask_missing_tt = np.isclose(travel_time, 0.0)
        if mask_missing_tt.any():
            length_m = df_valid.get("length_m", df_valid.get("length", pd.Series(0, index=df_valid.index)))
            lengths = length_m.fillna(0.0).values
            travel_time[mask_missing_tt] = lengths[mask_missing_tt] / 10.0
            travel_time = np.maximum(travel_time, 0.01) # Avoid zero cost loops

        # Bandwidth
        bandwidth_raw = df_valid["bandwidth_capacity"].fillna(1.0).values
        
        # Convert to Per-Minute Capacity (User Request: / 60 and ceil)
        # e.g. 3400 veh/h -> 57 veh/min
        bandwidth = np.ceil(bandwidth_raw / 60.0)
        bandwidth = np.maximum(bandwidth, 1.0) # Ensure at least 1
        
        num_nodes = len(self._node_ids)
        
        # Construct CSR Matrices directly (Undirected -> Add transpose)
        # Travel Time
        # COO format: (data, (row, col))
        print(f"[DEBUG] Building adjacency: {len(u_indices)} edges, num_nodes={num_nodes}", flush=True)
        print(f"[DEBUG] Edge indices range: u=[{u_indices.min()}-{u_indices.max()}], v=[{v_indices.min()}-{v_indices.max()}]", flush=True)
        print(f"[DEBUG] Travel time range: [{travel_time.min():.2f} - {travel_time.max():.2f}]", flush=True)
        
        tt_coo = sparse.coo_matrix(
            (travel_time, (u_indices, v_indices)), 
            shape=(num_nodes, num_nodes),
            dtype=np.float32
        )
        self._adjacency_travel_time = (tt_coo + tt_coo.T).tocsr()
        
        print(f"[DEBUG] Adjacency matrix built: shape={self._adjacency_travel_time.shape}, nnz={self._adjacency_travel_time.nnz}", flush=True)

        # Bandwidth
        bw_coo = sparse.coo_matrix(
            (bandwidth, (u_indices, v_indices)),
            shape=(num_nodes, num_nodes),
            dtype=np.float32
        )
        self._adjacency_bandwidth = (bw_coo + bw_coo.T).tocsr()

        # 4. Build Edge Capacity Map & NX Edges
        # This part requires iteration if we want string keys, but it's fast enough 
        # or we could optimize later. For now, just filling data.
        # Efficient NX edge adding
        edge_data_iter = zip(
            df_valid["source"], df_valid["target"], 
            df_valid["length_m"].fillna(0),
            df_valid["bandwidth_capacity"].fillna(0),
            df_valid["travel_time_seconds"].fillna(0),
            df_valid["width_m"].fillna(0)
        )
        for s, t, l, bw_orig, tt, w in edge_data_iter:
            # Apply same scaling to graph attributes and capacity map
            # User request: Convert hourly capacity to minute capacity (ceil(bw/60))
            bw = max(1.0, np.ceil(bw_orig / 60.0))
            self._graph.add_edge(s, t, length=l, bandwidth_capacity=bw, travel_time_seconds=tt, width_m=w)
            # Edge key map
            k = self._edge_key(s, t)
            self._edge_capacities[k] = bw

        self._loaded = True
        print("[DEBUG] Graph data loaded successfully.", flush=True)

    @property
    def graph(self) -> nx.Graph:
        self.ensure_loaded()
        return self._graph # type: ignore

    @property
    def edge_df(self) -> pd.DataFrame:
        self.ensure_loaded()
        return self._edge_df # type: ignore

    @property
    def node_df(self) -> pd.DataFrame:
        self.ensure_loaded()
        return self._node_df # type: ignore
        
    @property
    def node_ids(self) -> List[int]:
        self.ensure_loaded()
        return self._node_ids

    @property
    def node_index(self) -> Dict[int, int]:
        self.ensure_loaded()
        return self._node_index

    @property
    def index_node(self) -> Dict[int, int]:
        self.ensure_loaded()
        return self._index_node

    @property
    def num_nodes(self) -> int:
        self.ensure_loaded()
        return len(self._node_ids)

    @property
    def adjacency_travel_time(self) -> sparse.csr_matrix:
        self.ensure_loaded()
        return self._adjacency_travel_time # type: ignore

    @property
    def adjacency_bandwidth(self) -> sparse.csr_matrix:
        self.ensure_loaded()
        return self._adjacency_bandwidth # type: ignore
    
    # ... Helper methods that depend on loaded data ...
    def nearest_node(self, lat: float, lon: float) -> int:
        self.ensure_loaded()
        if not self._node_coords:
             raise ValueError("Graph has no coordinates")
        # Optimization: Use KDTree or simple distance calculation
        # Current impl is numpy broadcast, which involves full array.
        # Since we have coords property, we can just use it.
        points = np.array(list(self._node_coords.values()))
        dists = np.linalg.norm(points - np.array([lon, lat]), axis=1)
        idx = int(np.argmin(dists))
        return list(self._node_coords.keys())[idx]

    def node_exists(self, node_id: int) -> bool:
        self.ensure_loaded()
        return node_id in self._node_index

    def ensure_node(self, node_id: int) -> None:
        if not self.node_exists(node_id):
            raise ValueError(f"Node {node_id} not present in the component graph")

    def node_id_to_index(self, node_id: int) -> int:
        self.ensure_loaded()
        return int(self._node_index[node_id])

    def index_to_node_id(self, index: int) -> int:
        self.ensure_loaded()
        return int(self._index_node[index])

    def node_coords(self, node_id: int) -> Tuple[float, float] | None:
        # Note: nodes_geojson uses this. If we force ensure_loaded here, 
        # we break the "Visualize First" promise if nodes_geojson calls this.
        # BUT nodes_geojson calls this inside _generate_nodes_geojson.
        # _generate_nodes_geojson is only called if cache MISSES.
        # So it IS correct to ensure loaded here.
        self.ensure_loaded()
        return self._node_coords.get(node_id)
        
    def path_indices_to_node_ids(self, path_indices: List[int]) -> List[int]:
        self.ensure_loaded()
        node_path: List[int] = []
        for idx in path_indices:
            if idx < 0: break
            node_id = self._index_node.get(int(idx))
            if node_id is not None:
                node_path.append(node_id)
        return node_path

    def routes_indices_to_node_ids(self, routes: np.ndarray) -> List[List[int]]:
        self.ensure_loaded()
        node_paths: List[List[int]] = []
        for row in routes:
            node_paths.append(self.path_indices_to_node_ids([int(x) for x in row]))
        return node_paths

    # GeoJSON properties (Lazy + Cached)
    @property
    def to_geojson(self) -> Dict:
        # Check disk cache first WITHOUT ensure_loaded
        # Check disk cache first WITHOUT ensure_loaded
        return self._get_or_create_geojson(
            "hanoi_component_edges_min_v1.geojson", self._generate_edges_geojson
        )

    @property
    def nodes_geojson(self) -> Dict:
        return self._get_or_create_geojson(
            "hanoi_component_nodes.geojson", self._generate_nodes_geojson
        )

    def _get_or_create_geojson(self, filename: str, generator_func) -> Dict:
        import json
        from pathlib import Path

        settings = get_settings()
        cache_path = settings.edge_csv_path.parent / filename

        if cache_path.exists():
            print(f"[DEBUG] Loading {filename} from disk...", flush=True)
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load cache {filename}: {e}", flush=True)

        print(f"[DEBUG] Generating {filename} (Disk Cache Miss)...", flush=True)
        # Only NOW do we need the heavy data
        self.ensure_loaded() 
        data = generator_func()
        
        try:
            print(f"[DEBUG] Saving {filename} to disk...", flush=True)
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[WARN] Failed to save cache {filename}: {e}", flush=True)
            
        return data

    def clear_disk_cache(self) -> None:
        from pathlib import Path
        settings = get_settings()
        for filename in ["hanoi_component_edges.geojson", "hanoi_component_nodes.geojson"]:
            cache_path = settings.edge_csv_path.parent / filename
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    print(f"[DEBUG] Deleted cache file {filename}", flush=True)
                except Exception as e:
                    print(f"[WARN] Failed to delete cache file {filename}: {e}", flush=True)

    def _generate_edges_geojson(self) -> Dict:
        features: List[Dict] = []
        # Use graph edges to ensure we get scaled attributes (bandwidth_capacity)
        for u, v, data in self._graph.edges(data=True):
            geometry = self._edge_geometry(u, v)
            if geometry is None:
                continue
            
            edge_key = self._edge_key(u, v)
            features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {
                        **{
                            key: val
                            for key, val in data.items()
                            if key not in {"geometry"}
                        },
                        "source": u,
                        "target": v,
                        "edge_key": edge_key,
                    }
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def _generate_nodes_geojson(self) -> Dict:
        features: List[Dict] = []
        for node_id in self._node_ids:
            coords = self.node_coords(node_id)
            if coords is None:
                continue
            lon, lat = coords
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "node_id": node_id,
                        "degree": int(self._graph.degree(node_id)),
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def _edge_geometry(self, u: int, v: int):
        # We can look up in edge_df but easier to rely on graph if loaded
        if not self._graph.has_edge(u, v):
            return None
        data = self._graph[u][v]
        geom = data.get("geometry")
        if geom is not None:
            coords = list(geom.coords)
            return {"type": "LineString", "coordinates": coords}
        u_coords = self.node_coords(u)
        v_coords = self.node_coords(v)
        if u_coords and v_coords:
            return {"type": "LineString", "coordinates": [u_coords, v_coords]}
        return None

    @staticmethod
    def _edge_key(u: int, v: int) -> str:
        a, b = sorted((int(u), int(v)))
        return f"{a}_{b}"

    def edge_key(self, u: int, v: int) -> str:
        return self._edge_key(u, v)

    def edge_bandwidth(self, u: int, v: int) -> float | None:
        self.ensure_loaded()
        return self._edge_capacities.get(self._edge_key(u, v))

    def parse_edge_key(self, edge_key: str) -> Tuple[int, int]:
        left, right = edge_key.split("_", 1)
        return int(left), int(right)
        
    def path_travel_time(self, node_ids: List[int]) -> float | None:
        self.ensure_loaded()
        if len(node_ids) < 2: return 0.0
        total = 0.0
        for u, v in zip(node_ids, node_ids[1:]):
            if v is None: continue
            u_idx = self._node_index.get(int(u))
            v_idx = self._node_index.get(int(v))
            if u_idx is None or v_idx is None: return None
            
            weight = self._adjacency_travel_time[u_idx, v_idx]
            if np.isinf(weight) or weight == 0: # Check 0 for sparse
                # Actually sparse 0 means no edge, but if we handle 0 correctly in build
                # we should check explicit existence or value
                # logic: if u,v not in graph, should return None
                # sparse matrix indexing returns 0.0 if not found
                # For travel time, 0.0 is suspicious unless same node.
                # Let's assume > 0 for valid edges.
                if u != v and weight == 0.0: return None 
                
            total += float(weight)
        return total


@lru_cache(maxsize=1)
def get_graph_data() -> GraphData:
    return GraphData()

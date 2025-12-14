import type { Feature } from "geojson";

export interface NodeCoordinate {
  node_id: number;
  lat: number;
  lon: number;
}

export interface VehicleRouteRequest {
  request_id?: number;
  origin_node: number;
  destination_node: number;
  start_time?: number;
}

export interface RoutePath {
  node_ids: number[];
  travel_time?: number | null;
  base_travel_time?: number | null;
  penalty_delay?: number | null;
  congestion_times?: number[];
  request_id?: number;
  start_time?: number;
  coordinates?: NodeCoordinate[];
}

export interface EdgeUsageStat {
  edge_key: string;
  source: number;
  target: number;
  bandwidth_capacity?: number;
  effective_bandwidth?: number;  // B after scaling
  used_count: number;
  usage_ratio?: number;
}

export interface TimelineItem {
  request_id: number;
  start_time: number;
  travel_time: number;
  end_time: number;
}

export interface PeakHour {
  minute: number;
  count: number;
  total_penalty: number;
}

export interface RouteResponse {
  paths: RoutePath[];
  edge_stats: EdgeUsageStat[];
  timeline: TimelineItem[];
  peak_hours?: PeakHour[];
  failed_requests?: number[];
  // Dynamic visualization (penalty algorithm only)
  edge_time_usage?: Record<string, number[]>;
  time_bucket_seconds?: number;
  max_time_minutes?: number;
}

export interface FeatureCollection {
  type: "FeatureCollection";
  features: Feature[];
}

export interface NearestNodeResponse {
  node_id: number;
  lat: number;
  lon: number;
}

// keep legacy node coordinate type for UI overlays

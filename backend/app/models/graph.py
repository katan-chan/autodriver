from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class NodeCoordinate(BaseModel):
    node_id: int
    lat: float
    lon: float


class GeoJSONFeature(BaseModel):
    type: str
    geometry: dict
    properties: dict


class FeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature] = Field(default_factory=list)


class NearestNodeRequest(BaseModel):
    lat: float
    lon: float


class NearestNodeResponse(BaseModel):
    node_id: int
    lat: float
    lon: float


class VehicleRouteRequest(BaseModel):
    request_id: Optional[int] = None
    origin_node: int
    destination_node: int
    start_time: Optional[float] = None


class RouteRequest(BaseModel):
    origin_node: Optional[int] = Field(default=None, description="Fallback single-route origin")
    destination_node: Optional[int] = Field(default=None, description="Fallback single-route destination")
    start_time: Optional[float] = Field(default=None, description="Single-route start time (seconds)")
    algorithm: str = Field(default="shortest", description="shortest|penalty")
    k_paths: int = 3
    bandwidth_scale: float = Field(default=0.01, description="Scale factor for bandwidth. B_effective = capacity * scale")
    requests: Optional[List[VehicleRouteRequest]] = Field(
        default=None,
        description="Batch vehicle requests. If provided, origin/destination fields are ignored.",
    )


class RoutePath(BaseModel):
    node_ids: List[int]
    travel_time: Optional[float] = None
    request_id: Optional[int] = None
    start_time: Optional[float] = None


class EdgeUsageStat(BaseModel):
    edge_key: str
    source: int
    target: int
    bandwidth_capacity: Optional[float] = None
    effective_bandwidth: Optional[float] = None  # B after scaling
    used_count: int
    usage_ratio: Optional[float] = None


class TimelineItem(BaseModel):
    request_id: int
    start_time: float
    travel_time: float
    end_time: float


class RouteResponse(BaseModel):
    paths: List[RoutePath]
    edge_stats: List[EdgeUsageStat] = Field(default_factory=list)
    timeline: List[TimelineItem] = Field(default_factory=list)
    failed_requests: List[int] = Field(default_factory=list)


class GenerateFakeTripsRequest(BaseModel):
    count: int = Field(default=20, ge=1, description="Number of trips to generate")
    max_start_time: float = Field(default=300.0, ge=1.0, description="Maximum start time in seconds")


class GenerateFakeTripsResponse(BaseModel):
    trips: List[VehicleRouteRequest]

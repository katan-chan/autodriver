import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl, { type MapGeoJSONFeature } from "maplibre-gl";
import api from "./services/api";
import type { FeatureCollection, RoutePath, RouteResponse, VehicleRouteRequest } from "./types";

const MAP_CENTER: [number, number] = [105.83416, 21.027764];
const MAP_ZOOM = 11;
const EDGE_SOURCE_ID = "hanoi-edges";
const EDGE_LAYER_ID = "hanoi-edges-layer";
const EDGE_LABEL_LAYER_ID = "hanoi-edges-label-layer";
const NODE_SOURCE_ID = "hanoi-nodes";
const NODE_LAYER_ID = "hanoi-nodes-layer";
const ROUTE_SOURCE_ID = "route-paths";
const ROUTE_LAYER_ID = "route-paths-layer";
const ROUTE_COLORS = ["#fb7185", "#34d399", "#fde047", "#a78bfa", "#60a5fa"];

const getRequestColor = (requestId: number | undefined) => {
  if (!requestId) {
    return ROUTE_COLORS[0];
  }
  const base = ROUTE_COLORS[(requestId - 1) % ROUTE_COLORS.length];
  const rotations = Math.floor((requestId - 1) / ROUTE_COLORS.length);
  if (rotations === 0) {
    return base;
  }
  const hueShift = (rotations * 37) % 360;
  const color = hslFromHex(base, hueShift);
  return color;
};

const hslFromHex = (hex: string, hueShift: number) => {
  const h = hex.replace("#", "");
  const r = parseInt(h.substring(0, 2), 16) / 255;
  const g = parseInt(h.substring(2, 4), 16) / 255;
  const b = parseInt(h.substring(4, 6), 16) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let hue = 0;
  let saturation = 0;
  const lightness = (max + min) / 2;
  if (max !== min) {
    const d = max - min;
    saturation = lightness > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r:
        hue = (g - b) / d + (g < b ? 6 : 0);
        break;
      case g:
        hue = (b - r) / d + 2;
        break;
      case b:
        hue = (r - g) / d + 4;
        break;
    }
    hue /= 6;
  }
  const shiftedHue = (hue * 360 + hueShift) % 360;
  return `hsl(${shiftedHue}, ${Math.round(saturation * 100)}%, ${Math.round(lightness * 100)}%)`;
};

type NodeSelection = {
  node_id: number;
  lat: number;
  lon: number;
};

function App() {
  const mapContainer = useRef<HTMLDivElement | null>(null);
  const mapInstance = useRef<maplibregl.Map | null>(null);
  const nodeCoordRef = useRef<Map<number, [number, number]>>(new Map());
  const nodeIdListRef = useRef<number[]>([]);
  const edgesDataRef = useRef<FeatureCollection | null>(null);
  const originMarkerRef = useRef<maplibregl.Marker | null>(null);
  const destinationMarkerRef = useRef<maplibregl.Marker | null>(null);
  const fakeMarkersRef = useRef<maplibregl.Marker[]>([]);
  const vehicleMarkersRef = useRef<maplibregl.Marker[]>([]);

  const [status, setStatus] = useState<string>("Loading map...");
  const [activeTab, setActiveTab] = useState<"map" | "timeline" | "settings">("map");

  // Algorithm config state
  const [configBandwidthScale, setConfigBandwidthScale] = useState<number>(0.25);
  const [configPenaltyBeta, setConfigPenaltyBeta] = useState<number>(1.0);
  const [configKPaths, setConfigKPaths] = useState<number>(3);
  const [configKShortestPenalty, setConfigKShortestPenalty] = useState<number>(1000.0);
  const [selectionMode, setSelectionMode] = useState<"origin" | "destination">("origin");
  const [originSelection, setOriginSelection] = useState<NodeSelection | null>(null);
  const [destinationSelection, setDestinationSelection] = useState<NodeSelection | null>(null);
  const [algorithm, setAlgorithm] = useState<"shortest" | "penalty">("shortest");
  const [kPaths, setKPaths] = useState<number>(3);
  const [routeResponse, setRouteResponse] = useState<RouteResponse | null>(null);
  const [isRouting, setIsRouting] = useState<boolean>(false);
  const [fakeTripCount, setFakeTripCount] = useState<number>(20);
  const [maxStartTime, setMaxStartTime] = useState<number>(300);
  const [fakeTrips, setFakeTrips] = useState<VehicleRouteRequest[]>([]);
  const [visibleTripId, setVisibleTripId] = useState<number | null>(null);
  const [showTripMarkers, setShowTripMarkers] = useState<boolean>(false);
  const [configLoading, setConfigLoading] = useState<boolean>(false);

  // Dynamic time visualization
  const [currentMinute, setCurrentMinute] = useState<number>(0);
  const [maxTimeMinutes, setMaxTimeMinutes] = useState<number>(60);
  const [showVehicles, setShowVehicles] = useState<boolean>(true);

  const updateFakeTripMarkers = (trips: VehicleRouteRequest[], filterId: number | null, show: boolean) => {
    console.log(`[DEBUG] updateFakeTripMarkers called: trips=${trips.length}, show=${show}, nodeCoords=${nodeCoordRef.current.size}`);
    fakeMarkersRef.current.forEach((marker) => marker.remove());
    fakeMarkersRef.current = [];
    const map = mapInstance.current;
    if (!map || !show) return;
    trips.forEach((trip) => {
      if (filterId && trip.request_id !== filterId) {
        return;
      }
      const color = getRequestColor(trip.request_id);
      const pickup = nodeCoordRef.current.get(trip.origin_node);
      const dropoff = nodeCoordRef.current.get(trip.destination_node);
      console.log(`[DEBUG] Trip ${trip.request_id}: origin=${trip.origin_node} found=${!!pickup}, dest=${trip.destination_node} found=${!!dropoff}`);
      if (pickup) {
        const el = document.createElement("div");
        el.className = "fake-trip-triangle pickup";
        el.style.setProperty("--triangle-color", color);
        const marker = new maplibregl.Marker({ element: el }).setLngLat(pickup).addTo(map);
        fakeMarkersRef.current.push(marker);
      }
      if (dropoff) {
        const el = document.createElement("div");
        el.className = "fake-trip-triangle dropoff";
        el.style.setProperty("--triangle-color", color);
        const marker = new maplibregl.Marker({ element: el }).setLngLat(dropoff).addTo(map);
        fakeMarkersRef.current.push(marker);
      }
    });
    console.log(`[DEBUG] Created ${fakeMarkersRef.current.length} markers`);
  };

  const loadConfig = async () => {
    try {
      const response = await api.get("/config");
      const cfg = response.data;
      setConfigBandwidthScale(cfg.bandwidth_scale ?? 0.25);
      setConfigPenaltyBeta(cfg.penalty_beta ?? 1.0);
      setConfigKPaths(cfg.k_paths ?? 3);
      setConfigKShortestPenalty(cfg.k_shortest_penalty ?? 1000.0);
    } catch (error) {
      console.error("Failed to load config:", error);
    }
  };

  const saveConfig = async () => {
    setConfigLoading(true);
    try {
      await api.post("/config", {
        bandwidth_scale: configBandwidthScale,
        penalty_beta: configPenaltyBeta,
        k_paths: configKPaths,
        k_shortest_penalty: configKShortestPenalty,
      });
      setStatus("Đã lưu cấu hình thuật toán.");
    } catch (error) {
      console.error("Failed to save config:", error);
      setStatus("Lỗi khi lưu cấu hình.");
    } finally {
      setConfigLoading(false);
    }
  };

  useEffect(() => {
    loadConfig();
  }, []);

  useEffect(() => {
    console.log("[App] UI bundle loaded - build with custom markers");
    if (mapContainer.current && !mapInstance.current) {
      mapInstance.current = new maplibregl.Map({
        container: mapContainer.current,
        style: {
          version: 8,
          glyphs: "https://fonts.openmaptiles.org/{fontstack}/{range}.pbf",
          sources: {
            osm: {
              type: "raster",
              tiles: [
                "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
                "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
                "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
              ],
              tileSize: 256,
              attribution:
                '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            },
          },
          layers: [
            {
              id: "osm",
              type: "raster",
              source: "osm",
              minzoom: 0,
              maxzoom: 19,
            },
          ],
        },
        center: MAP_CENTER,
        zoom: MAP_ZOOM,
      });

      mapInstance.current.addControl(new maplibregl.NavigationControl());
      mapInstance.current.on("load", () => {
        setStatus("Đang tải dữ liệu đường...");
        loadRoadEdges();
        loadNodes();
        mapInstance.current?.on("click", (event: maplibregl.MapMouseEvent) => {
          handleMapClick(event.lngLat.lng, event.lngLat.lat);
        });
      });
    }
    return () => {
      mapInstance.current?.remove();
      mapInstance.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (activeTab === "map") {
      mapInstance.current?.resize();
    }
  }, [activeTab]);

  const loadRoadEdges = async () => {
    try {
      const response = await api.get<FeatureCollection>("/graph/edges");
      const geojson = response.data;
      const map = mapInstance.current;
      if (!map) return;

      if (map.getSource(EDGE_SOURCE_ID)) {
        map.removeLayer(EDGE_LAYER_ID);
        if (map.getLayer(EDGE_LABEL_LAYER_ID)) {
          map.removeLayer(EDGE_LABEL_LAYER_ID);
        }
        map.removeSource(EDGE_SOURCE_ID);
      }

      map.addSource(EDGE_SOURCE_ID, {
        type: "geojson",
        data: geojson,
      });

      map.addLayer({
        id: EDGE_LAYER_ID,
        type: "line",
        source: EDGE_SOURCE_ID,
        paint: {
          "line-color": [
            "interpolate",
            ["linear"],
            ["coalesce", ["get", "usage_ratio"], 0],
            0,
            "#22c55e",
            0.5,
            "#eab308",
            1,
            "#f97316",
            1.5,
            "#ef4444",
          ],
          "line-width": [
            "+",
            ["interpolate", ["linear"], ["coalesce", ["get", "bandwidth_capacity"], 1], 1, 2, 12, 5],
            ["*", ["coalesce", ["get", "usage_ratio"], 0], 2],
          ],
          "line-opacity": 0.6,
        },
      });

      map.addLayer({
        id: EDGE_LABEL_LAYER_ID,
        type: "symbol",
        source: EDGE_SOURCE_ID,
        minzoom: 13,
        layout: {
          "text-field": [
            "concat",
            ["to-string", ["coalesce", ["get", "used_count"], 0]],
            " / ",
            ["to-string", ["round", ["coalesce", ["get", "effective_bandwidth"], ["get", "bandwidth_capacity"], 0]]],
          ],
          "text-size": 10,
          "text-allow-overlap": false,
          "text-font": ["Open Sans Regular"],
        },
        paint: {
          "text-color": [
            "interpolate",
            ["linear"],
            ["coalesce", ["get", "usage_ratio"], 0],
            0,
            "#22c55e",
            1,
            "#f97316",
            2,
            "#dc2626",
          ],
          "text-halo-color": "#ffffff",
          "text-halo-width": 1.2,
        },
      });

      map.moveLayer(EDGE_LABEL_LAYER_ID);

      edgesDataRef.current = geojson;
      setStatus("Đã tải dữ liệu. Nhấp vào bản đồ để chọn điểm.");
    } catch (error) {
      console.error(error);
      setStatus("Không tải được dữ liệu từ backend. Đảm bảo API đang chạy.");
    }
  };

  const handleGenerateFakeTrips = async () => {
    const count = Number.isFinite(fakeTripCount) && fakeTripCount >= 1 ? Math.floor(fakeTripCount) : 20;
    const maxStart = Number.isFinite(maxStartTime) && maxStartTime >= 1 ? maxStartTime : 300;
    setStatus("Đang sinh fake trips từ server...");
    try {
      const response = await api.post<{ trips: VehicleRouteRequest[] }>("/generate-fake-trips", {
        count,
        max_start_time: maxStart,
      });
      const trips = response.data.trips;
      if (!trips.length) {
        setStatus("Server không sinh được chuyến nào.");
        return;
      }
      setFakeTrips(trips);
      updateFakeTripMarkers(trips, visibleTripId, showTripMarkers);
      setStatus(`Đã sinh ${trips.length} chuyến giả từ server. Chạy thuật toán để xem kết quả.`);
    } catch (error: any) {
      console.error("Generate fake trips error:", error.response?.data || error);
      const detail = error.response?.data?.detail;
      setStatus(`Lỗi: ${detail ? JSON.stringify(detail) : "Không sinh được fake trips"}`);
    }
  };

  const runFakeTrips = async () => {
    const tripsToRun = fakeTrips;
    if (!tripsToRun.length) {
      setStatus("Chưa có chuyến giả. Hãy sinh dữ liệu trước.");
      return;
    }
    setIsRouting(true);
    setStatus("Đang tính tuyến cho fake trips...");
    try {
      const payload = {
        algorithm,
        k_paths: algorithm === "penalty" ? kPaths : 1,
        bandwidth_scale: configBandwidthScale,
        requests: tripsToRun,
      };
      const response = await api.post<RouteResponse>("/route", payload);
      processRouteResponse(
        response.data,
        `Đã tính ${response.data.paths.length} tuyến cho fake trips. Xem tab Timeline để phân tích.`,
      );
    } catch (error) {
      console.error(error);
      setStatus("Không tính được tuyến fake trips. Kiểm tra backend.");
    } finally {
      setIsRouting(false);
    }
  };

  const handleRunFakeTrips = () => runFakeTrips();

  const loadNodes = async () => {
    try {
      const response = await api.get<FeatureCollection>("/graph/nodes");
      const geojson = response.data;
      const map = mapInstance.current;
      if (!map) return;

      if (map.getSource(NODE_SOURCE_ID)) {
        map.removeLayer(NODE_LAYER_ID);
        map.removeSource(NODE_SOURCE_ID);
      }

      map.addSource(NODE_SOURCE_ID, {
        type: "geojson",
        data: geojson,
      });

      map.addLayer({
        id: NODE_LAYER_ID,
        type: "circle",
        source: NODE_SOURCE_ID,
        paint: {
          "circle-radius": 2.2,
          "circle-color": "#1d4ed8",
          "circle-stroke-width": 0.5,
          "circle-stroke-color": "#ffffff",
        },
      });

      // Ensure layer z-order: edges at bottom, then nodes on top
      // Markers (DOM elements) are always above all layers
      if (map.getLayer(EDGE_LAYER_ID)) {
        map.moveLayer(NODE_LAYER_ID); // Move nodes to top
      }

      const coords = new Map<number, [number, number]>();
      for (const feature of geojson.features) {
        if (feature.geometry.type !== "Point") continue;
        const [lon, lat] = feature.geometry.coordinates as [number, number];
        const props = feature.properties as { node_id: number };
        coords.set(props.node_id, [lon, lat]);
      }
      nodeCoordRef.current = coords;
      nodeIdListRef.current = Array.from(coords.keys());

      map.on("mouseenter", NODE_LAYER_ID, () => {
        map.getCanvas().style.cursor = "pointer";
      });

      map.on("mouseleave", NODE_LAYER_ID, () => {
        map.getCanvas().style.cursor = "";
      });
    } catch (error) {
      console.error(error);
    }
  };

  const handleMapClick = async (lon: number, lat: number) => {
    if (activeTab !== "map") return;
    setStatus("Đang tìm node gần nhất...");
    try {
      const response = await api.post("/graph/nearest-node", { lat, lon });
      const node = response.data as NodeSelection;
      if (selectionMode === "origin") {
        setOriginSelection(node);
        placeMarker(originMarkerRef, node.lon, node.lat, "origin-marker");
        setStatus(`Origin node: ${node.node_id}`);
      } else {
        setDestinationSelection(node);
        placeMarker(destinationMarkerRef, node.lon, node.lat, "destination-marker");
        setStatus(`Destination node: ${node.node_id}`);
      }
    } catch (error) {
      console.error(error);
      setStatus("Không tìm được node gần nhất.");
    }
  };

  const placeMarker = (
    markerRef: React.MutableRefObject<maplibregl.Marker | null>,
    lon: number,
    lat: number,
    className: string,
  ) => {
    if (!mapInstance.current) return;
    markerRef.current?.remove();
    const el = document.createElement("div");
    el.className = `manual-marker ${className}`;
    markerRef.current = new maplibregl.Marker({ element: el })
      .setLngLat([lon, lat])
      .addTo(mapInstance.current);
  };

  const processRouteResponse = (response: RouteResponse, message: string) => {
    setRouteResponse(response);
    setVisibleTripId(null);
    setShowTripMarkers(false); // Default hidden to prevent crash with high trip count
    setShowVehicles(false);
    drawRoutes(response.paths, null, false);
    // Don't call applyEdgeUsage - let useEffect handle t=0 (clearEdgeLoads)
    setStatus(message);

    // Set max time for dynamic visualization (if available)
    if (response.max_time_minutes) {
      setMaxTimeMinutes(response.max_time_minutes);
      setCurrentMinute(0); // This triggers useEffect to clearEdgeLoads
    }
  };

  const handleRunRoute = async () => {
    if (!originSelection || !destinationSelection) {
      setStatus("Chọn đủ điểm origin & destination trước khi chạy.");
      return;
    }
    setIsRouting(true);
    setStatus("Đang tính tuyến...");
    try {
      const payload = {
        origin_node: originSelection.node_id,
        destination_node: destinationSelection.node_id,
        start_time: 0,
        algorithm,
        k_paths: algorithm === "penalty" ? kPaths : 1,
        bandwidth_scale: configBandwidthScale,
      };
      const response = await api.post<RouteResponse>("/route", payload);
      processRouteResponse(
        response.data,
        `Đã tính ${response.data.paths.length} tuyến (single). Chuyển tab Timeline để xem Gantt.`,
      );
    } catch (error) {
      console.error(error);
      setStatus("Không tính được tuyến. Kiểm tra log backend.");
    } finally {
      setIsRouting(false);
    }
  };

  const drawRoutes = (paths: RoutePath[], filterId: number | null, show: boolean) => {
    const map = mapInstance.current;
    if (!map) return;
    const nodeCoords = nodeCoordRef.current;
    const subset = show
      ? filterId
        ? paths.filter((path) => (path.request_id ?? null) === filterId)
        : paths
      : [];
    const features = subset
      .map((path) => {
        const coords = path.node_ids
          .map((nodeId) => nodeCoords.get(nodeId))
          .filter((coord): coord is [number, number] => Array.isArray(coord));
        if (coords.length < 2) {
          return null;
        }
        const color = getRequestColor(path.request_id);
        return {
          type: "Feature",
          geometry: { type: "LineString", coordinates: coords },
          properties: {
            request_id: path.request_id ?? 0,
            color,
          },
        };
      })
      .filter(Boolean);

    const collection = { type: "FeatureCollection", features } as FeatureCollection;

    if (map.getSource(ROUTE_SOURCE_ID)) {
      (map.getSource(ROUTE_SOURCE_ID) as maplibregl.GeoJSONSource).setData(collection as any);
      if (map.getLayer(ROUTE_LAYER_ID)) {
        map.setLayoutProperty(ROUTE_LAYER_ID, "visibility", show ? "visible" : "none");
      }
    } else {
      map.addSource(ROUTE_SOURCE_ID, {
        type: "geojson",
        data: collection,
      });
      map.addLayer({
        id: ROUTE_LAYER_ID,
        type: "line",
        source: ROUTE_SOURCE_ID,
        layout: {
          "line-join": "round",
          "line-cap": "round",
          "visibility": show ? "visible" : "none"
        },
        paint: {
          "line-color": ["coalesce", ["get", "color"], "#ef4444"],
          "line-width": 5,
          "line-opacity": 0.9,
        },
      });
    }

    if (map.getLayer(EDGE_LABEL_LAYER_ID)) {
      map.moveLayer(EDGE_LABEL_LAYER_ID);
    }
  };

  useEffect(() => {
    updateFakeTripMarkers(fakeTrips, visibleTripId, showTripMarkers);
  }, [fakeTrips, visibleTripId, showTripMarkers]);

  useEffect(() => {
    if (routeResponse) {
      const map = mapInstance.current;
      if (map && map.isStyleLoaded()) {
        drawRoutes(routeResponse.paths, visibleTripId, showTripMarkers);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visibleTripId, showTripMarkers]);

  const applyEdgeUsage = (response: RouteResponse) => {
    const map = mapInstance.current;
    const base = edgesDataRef.current;
    if (!map || !base) return;
    const usageMap = new Map(response.edge_stats.map((stat) => [stat.edge_key, stat]));
    const updated: FeatureCollection = {
      ...base,
      features: base.features.map((feature) => {
        const props = feature.properties as Record<string, unknown>;
        const key = props?.edge_key as string | undefined;
        const stat = key ? usageMap.get(key) : undefined;
        return {
          ...feature,
          properties: {
            ...props,
            used_count: stat?.used_count ?? 0,
            usage_ratio: stat?.usage_ratio ?? 0,
            effective_bandwidth: ((props?.bandwidth_capacity as number) || 1) * configBandwidthScale,
          },
        };
      }),
    };
    edgesDataRef.current = updated;
    const source = map.getSource(EDGE_SOURCE_ID) as maplibregl.GeoJSONSource | undefined;
    source?.setData(updated as any);
  };

  // Update edge colors based on load at specific minute (for dynamic visualization)
  const updateEdgeColorsAtMinute = (minute: number) => {
    const map = mapInstance.current;
    const base = edgesDataRef.current;
    if (!map || !base || !routeResponse?.edge_time_usage) return;

    const timeUsage = routeResponse.edge_time_usage;
    // Slider at minute M shows edges being traversed during that minute
    // Bucket index = minute - 1 when minute > 0 (since bucket 0 = minute 0-1)
    const timeBucket = Math.max(0, minute - 1);

    const updated: FeatureCollection = {
      ...base,
      features: base.features.map((feature) => {
        const props = feature.properties as Record<string, unknown>;
        const edgeKey = props?.edge_key as string | undefined;

        if (!edgeKey) return feature;

        // Get load at this time bucket for this edge
        const loads = timeUsage[edgeKey];
        const loadAtTime = loads ? (loads[timeBucket] || 0) : 0;

        // Bandwidth is now already in Per-Minute units from backend (graph_loader)
        const bandwidth = ((props?.bandwidth_capacity as number) || 1) * configBandwidthScale;
        const usageRatio = loadAtTime / bandwidth;

        return {
          ...feature,
          properties: {
            ...props,
            used_count: loadAtTime,
            usage_ratio: usageRatio,
          },
        };
      }),
    };

    const source = map.getSource(EDGE_SOURCE_ID) as maplibregl.GeoJSONSource | undefined;
    source?.setData(updated as any);
  };

  // Effect: update edge colors when currentMinute changes
  useEffect(() => {
    if (currentMinute === 0) {
      // At t=0, no vehicle has moved yet - show zero load on all edges
      clearEdgeLoads();
    } else if (routeResponse?.edge_time_usage) {
      updateEdgeColorsAtMinute(currentMinute);
    }
    // Update vehicle positions for all algorithms (if toggle is on)
    if (showVehicles && routeResponse?.paths && routeResponse.max_time_minutes) {
      updateVehiclePositions(currentMinute * 60);
    } else {
      // Clear vehicle markers if toggle is off
      vehicleMarkersRef.current.forEach((m) => m.remove());
      vehicleMarkersRef.current = [];
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentMinute, showVehicles]);

  // Auto-update visualization when bandwidth scale changes
  useEffect(() => {
    if (routeResponse) {
      if (currentMinute === 0) {
        clearEdgeLoads();
      } else if (routeResponse.edge_time_usage) {
        updateEdgeColorsAtMinute(currentMinute);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [configBandwidthScale]);

  // Clear all edge loads (set usage_ratio to 0)
  const clearEdgeLoads = () => {
    const map = mapInstance.current;
    const base = edgesDataRef.current;
    if (!map || !base) return;

    const cleared: FeatureCollection = {
      ...base,
      features: base.features.map((feature) => ({
        ...feature,
        properties: {
          ...feature.properties,
          used_count: 0,
          usage_ratio: 0,
        },
      })),
    };

    const source = map.getSource(EDGE_SOURCE_ID) as maplibregl.GeoJSONSource | undefined;
    source?.setData(cleared as any);
  };

  // Update vehicle position markers at a given time (seconds)
  const updateVehiclePositions = (timeSeconds: number) => {
    const map = mapInstance.current;
    if (!map || !routeResponse?.paths) return;

    // Remove old vehicle markers
    vehicleMarkersRef.current.forEach((m) => m.remove());
    vehicleMarkersRef.current = [];

    for (const path of routeResponse.paths) {
      const startTime = path.start_time || 0;
      const nodes = path.node_ids;
      const requestId = path.request_id || 0;
      const totalTravelTime = path.travel_time || 0;

      // Vehicle not started yet - skip
      const endTime = startTime + totalTravelTime;
      if (timeSeconds < startTime) continue;

      // If trip finished, show at destination
      if (timeSeconds >= endTime) {
        const destCoords = nodeCoordRef.current.get(nodes[nodes.length - 1]);
        if (destCoords) {
          const color = getRequestColor(requestId);
          const el = document.createElement("div");
          el.className = "vehicle-marker finished";
          el.style.backgroundColor = color;
          const marker = new maplibregl.Marker({ element: el })
            .setLngLat(destCoords)
            .addTo(map);
          vehicleMarkersRef.current.push(marker);
        }
        continue;
      }

      // Calculate time per edge (proportional distribution)
      const numEdges = nodes.length - 1;
      if (numEdges <= 0) continue;
      const timePerEdge = totalTravelTime / numEdges;

      // Find current node using cumsum of travel times
      // Vehicle is at node i when cumulative time to reach node i >= current time
      let cumTime = startTime;
      let currentNodeId = nodes[0];

      for (let i = 1; i <= numEdges; i++) {
        cumTime += timePerEdge;
        if (cumTime >= timeSeconds) {
          // Haven't reached node i yet, still at node i-1
          currentNodeId = nodes[i - 1];
          break;
        }
        currentNodeId = nodes[i];
      }

      // Get coordinates for current node
      const coords = nodeCoordRef.current.get(currentNodeId);
      if (coords) {
        const color = getRequestColor(requestId);
        const el = document.createElement("div");
        el.className = "vehicle-marker";
        el.style.backgroundColor = color;
        const marker = new maplibregl.Marker({ element: el })
          .setLngLat(coords)
          .addTo(map);
        vehicleMarkersRef.current.push(marker);
      }
    }
  };

  const timeline = routeResponse?.timeline ?? [];
  const maxEndTime = useMemo(() => {
    if (!timeline.length) return 0;
    return Math.max(...timeline.map((item) => item.end_time));
  }, [timeline]);

  const renderTimeline = () => {
    if (!timeline.length) {
      return <p className="empty-state">Chưa có dữ liệu tuyến. Chạy thuật toán trên tab Bản đồ.</p>;
    }
    return (
      <div className="timeline-wrapper">
        <div className="timeline-bars">
          {timeline.map((item) => {
            const leftPct = maxEndTime ? (item.start_time / maxEndTime) * 100 : 0;
            const widthPct = maxEndTime ? (item.travel_time / maxEndTime) * 100 : 0;
            return (
              <div key={`timeline-${item.request_id}`} className="timeline-row">
                <span className="timeline-label">Req {item.request_id}</span>
                <div className="timeline-bar-track">
                  <div
                    className="timeline-bar"
                    style={{
                      left: `${leftPct}%`,
                      width: `${Math.max(widthPct, 1)}%`,
                    }}
                  />
                </div>
                <span className="timeline-meta">
                  start {item.start_time.toFixed(1)}s · travel {item.travel_time.toFixed(1)}s
                </span>
              </div>
            );
          })}
        </div>
        <table className="timeline-table">
          <thead>
            <tr>
              <th>Request</th>
              <th>Start (s)</th>
              <th>Travel (s)</th>
              <th>End (s)</th>
            </tr>
          </thead>
          <tbody>
            {timeline.map((item) => (
              <tr key={`timeline-table-${item.request_id}`}>
                <td>{item.request_id}</td>
                <td>{item.start_time.toFixed(1)}</td>
                <td>{item.travel_time.toFixed(1)}</td>
                <td>{item.end_time.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="app-shell">
      <header>
        <div className="header-row">
          <div>
            <h1>Hanoi Road Network</h1>
            <p className="status-line">{status}</p>
            <p className="hint-line">
              Backend: {import.meta.env.VITE_API_BASE_URL || "http://118.70.128.4:8000"}
            </p>
          </div>
          <div className="tabs">
            <button
              type="button"
              className={`tab-button ${activeTab === "map" ? "active" : ""}`}
              onClick={() => setActiveTab("map")}
            >
              Bản đồ
            </button>
            <button
              type="button"
              className={`tab-button ${activeTab === "timeline" ? "active" : ""}`}
              onClick={() => setActiveTab("timeline")}
            >
              Timeline / Gantt
            </button>
            <button
              type="button"
              className={`tab-button ${activeTab === "settings" ? "active" : ""}`}
              onClick={() => setActiveTab("settings")}
            >
              Cài đặt
            </button>
          </div>
        </div>
        <p className="tab-helper">1) Sinh điểm giả / hoặc chọn node thủ công → 2) Chạy thuật toán → 3) Chuyển qua tab Timeline để xem biểu đồ.</p>
      </header>
      <main>
        <div className={`map-panel ${activeTab === "map" ? "" : "hidden"}`}>
          <aside className="control-panel">
            <section>
              <h2>Fake trips</h2>
              <p className="hint-line">Sinh các cặp origin/destination ngẫu nhiên trong component.</p>
              <label className="control-label">
                Số chuyến cần sinh
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={fakeTripCount}
                  onChange={(e) => setFakeTripCount(Number(e.target.value))}
                />
              </label>
              <label className="control-label">
                Thời điểm bắt đầu tối đa (s)
                <input
                  type="number"
                  min={10}
                  max={3600}
                  value={maxStartTime}
                  onChange={(e) => setMaxStartTime(Number(e.target.value))}
                />
              </label>
              <div className="fake-buttons">
                <button type="button" className="secondary-btn" onClick={handleGenerateFakeTrips}>
                  1. Sinh fake trips
                </button>
                <button
                  type="button"
                  className="run-route-btn"
                  onClick={handleRunFakeTrips}
                  disabled={isRouting || fakeTrips.length === 0}
                >
                  {isRouting ? "Đang tính..." : "2. Chạy thuật toán"}
                </button>

                {routeResponse && routeResponse.paths.length > 0 && (
                  <div className="route-summary" style={{ fontSize: "0.85rem", padding: "0.5rem", background: "#f1f5f9", borderRadius: "6px", border: "1px solid #cbd5e1" }}>
                    <div style={{ fontWeight: 600, marginBottom: "4px" }}>System Cost Summary:</div>
                    <div>Total Time: {(routeResponse.paths.reduce((acc, p) => acc + (p.travel_time || 0), 0)).toFixed(1)}s</div>
                    <div>Total Penalty: {(routeResponse.paths.reduce((acc, p) => acc + (p.penalty_delay || 0), 0)).toFixed(1)}s</div>
                    <div>Avg Time: {(routeResponse.paths.reduce((acc, p) => acc + (p.travel_time || 0), 0) / routeResponse.paths.length).toFixed(1)}s/trip</div>
                  </div>
                )}

                <button
                  type="button"
                  className={`secondary-btn toggle-btn ${showTripMarkers ? "active" : ""}`}
                  onClick={() => {
                    const newState = !showTripMarkers;
                    setShowTripMarkers(newState);
                    setShowVehicles(newState);
                  }}
                  disabled={fakeTrips.length === 0}
                >
                  {showTripMarkers ? "Ẩn xe & điểm" : "Hiện xe & điểm"}
                </button>
              </div>
              <label className="control-label">
                Thuật toán
                <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value as "shortest" | "penalty")}>
                  <option value="shortest">Shortest path</option>
                  <option value="penalty">Penalty routing (k-paths)</option>
                </select>
              </label>
              {algorithm === "penalty" && (
                <label className="control-label">
                  Số đường thay thế (k)
                  <input
                    type="number"
                    min={2}
                    max={5}
                    value={kPaths}
                    onChange={(e) => setKPaths(Number(e.target.value))}
                  />
                </label>
              )}
              {fakeTrips.length > 0 && (
                <div className="fake-summary">
                  <p>
                    Đã sinh {fakeTrips.length} chuyến. Ví dụ:
                  </p>
                  <ul className="trip-legend">
                    {fakeTrips.map((trip) => {
                      const path = routeResponse?.paths.find((p) => (p.request_id ?? -1) === trip.request_id);
                      return (
                        <li key={`fake-${trip.request_id}`}>
                          <button
                            type="button"
                            className={visibleTripId === trip.request_id ? "active" : ""}
                            onClick={() =>
                              setVisibleTripId((prev) => (prev === trip.request_id ? null : trip.request_id ?? null))
                            }
                            style={{ flexDirection: "column", alignItems: "flex-start", gap: "2px", padding: "8px" }}
                          >
                            <div style={{ display: "flex", alignItems: "center", gap: "8px", width: "100%" }}>
                              <span
                                className="trip-dot"
                                style={{ backgroundColor: getRequestColor(trip.request_id) }}
                              />
                              <span style={{ fontWeight: 500 }}>Trip #{trip.request_id}</span>
                              {typeof trip.start_time === "number" && (
                                <span className="trip-meta" style={{ marginLeft: "auto" }}>start {trip.start_time.toFixed(1)}s</span>
                              )}
                            </div>

                            {path && (
                              <div className="trip-cost-details" style={{ width: "100%", marginTop: "4px", paddingLeft: "16px", fontSize: "0.85em", color: "#475569" }}>
                                <div style={{ display: "flex", justifyContent: "space-between" }}>
                                  <span>Time: <strong>{path.travel_time?.toFixed(1) ?? "?"}s</strong></span>
                                  {path.penalty_delay != null && path.penalty_delay > 0 && (
                                    <span style={{ color: "#ef4444" }}>+Pen: {path.penalty_delay.toFixed(1)}s</span>
                                  )}
                                </div>
                                {(path.base_travel_time != null) && (
                                  <div style={{ fontSize: "0.85em", color: "#64748b" }}>
                                    (Base: {path.base_travel_time.toFixed(1)}s)
                                  </div>
                                )}
                              </div>
                            )}
                          </button>
                        </li>
                      );
                    })}
                    {!fakeTrips.length && <li>Chưa sinh chuyến nào</li>}
                  </ul>
                </div>
              )}
            </section>

            <section>
              <h2>Chọn điểm thủ công</h2>
              <div className="selection-mode">
                <span>Mode:</span>
                <button
                  type="button"
                  className={selectionMode === "origin" ? "active" : ""}
                  onClick={() => setSelectionMode("origin")}
                >
                  Origin
                </button>
                <button
                  type="button"
                  className={selectionMode === "destination" ? "active" : ""}
                  onClick={() => setSelectionMode("destination")}
                >
                  Destination
                </button>
              </div>
              <div className="selection-summary">
                <p>
                  Origin: {originSelection ? originSelection.node_id : "(chưa chọn)"}
                </p>
                <p>
                  Destination: {destinationSelection ? destinationSelection.node_id : "(chưa chọn)"}
                </p>
              </div>
              <button
                type="button"
                className="secondary-btn"
                onClick={handleRunRoute}
                disabled={isRouting || !originSelection || !destinationSelection}
              >
                Chạy định tuyến thủ công
              </button>
            </section>

            {routeResponse && (
              <section className="route-summary">
                <h2>Kết quả gần nhất</h2>
                <p>{routeResponse.paths.length} tuyến.</p>
                <p>{routeResponse.edge_stats.length} cạnh có tải.</p>
                <p>Timeline entries: {routeResponse.timeline.length}. Mở tab Timeline để xem biểu đồ.</p>
              </section>
            )}
          </aside>
          <div className="map-wrapper">
            <div className="map-container" ref={mapContainer} />
            <div className="legend-panel">
              <div>
                <h3>Usage & bandwidth</h3>
                <p>Xanh: nhàn rỗi · Đỏ: vượt tải (used/B)</p>
                <p>Độ dày = width_m + số chuyến đi qua</p>
              </div>
              <div>
                <h3>Fake trips</h3>
                <p>▲ pickup · ▼ delivery (cùng màu = cùng request)</p>
              </div>
            </div>

            {/* Time Slider for Dynamic Visualization */}
            {routeResponse?.max_time_minutes && routeResponse.max_time_minutes > 0 && (
              <div className="time-slider-container">
                <button
                  type="button"
                  className="time-btn"
                  onClick={() => setCurrentMinute(Math.max(0, currentMinute - 1))}
                  disabled={currentMinute <= 0}
                >
                  −
                </button>
                <div style={{ position: "relative", flex: 1, display: "flex", alignItems: "center" }}>
                  <label className="time-slider-label" style={{ width: "100%", margin: 0 }}>
                    <span className="time-label" style={{ minWidth: "220px" }}>Thời gian: {currentMinute} / {maxTimeMinutes} phút</span>
                    <input
                      type="range"
                      min={0}
                      max={maxTimeMinutes}
                      value={currentMinute}
                      onChange={(e) => setCurrentMinute(Number(e.target.value))}
                      className="time-slider"
                      style={{ width: "100%" }}
                    />
                  </label>

                  {/* Peak Hours Markers */}
                  {routeResponse.peak_hours && routeResponse.peak_hours.map((ph, idx) => {
                    // Position calculation: normalized to 0-100%
                    const leftPercent = (ph.minute / maxTimeMinutes) * 100;
                    if (leftPercent < 0 || leftPercent > 100) return null;

                    return (
                      <div
                        key={idx}
                        className="peak-hour-marker"
                        title={`Congestion: ${ph.count} events, penalty: ${ph.total_penalty.toFixed(1)}s`}
                        style={{ left: `calc(${leftPercent}% - 3px)` }} // Adjust for marker width
                        onClick={() => setCurrentMinute(ph.minute)}
                      />
                    );
                  })}
                </div>
                <button
                  type="button"
                  className="time-btn"
                  onClick={() => setCurrentMinute(Math.min(maxTimeMinutes, currentMinute + 1))}
                  disabled={currentMinute >= maxTimeMinutes}
                >
                  +
                </button>
                <button
                  type="button"
                  className="reset-time-btn"
                  onClick={() => {
                    setCurrentMinute(0);
                    if (routeResponse) applyEdgeUsage(routeResponse);
                    // Clear vehicle markers
                    vehicleMarkersRef.current.forEach((m) => m.remove());
                    vehicleMarkersRef.current = [];
                  }}
                >
                  Reset
                </button>
              </div>
            )}
          </div>
        </div>
        <div className={`timeline-panel ${activeTab === "timeline" ? "" : "hidden"}`}>
          {renderTimeline()}
        </div>
        <div className={`settings-panel ${activeTab === "settings" ? "" : "hidden"}`}>
          <div className="settings-container">
            <h2>Cấu hình thuật toán</h2>
            <p className="hint-line">Điều chỉnh các tham số cho penalty routing.</p>

            <div className="settings-grid">
              <label className="control-label">
                <span>Bandwidth Scale</span>
                <span className="hint-line">B = capacity × scale. VD: capacity=600, scale=0.01 → B=6 xe</span>
                <input
                  type="number"
                  step="0.001"
                  min={0.001}
                  value={configBandwidthScale}
                  onChange={(e) => setConfigBandwidthScale(Number(e.target.value))}
                />
              </label>

              <label className="control-label">
                <span>Penalty Beta (β)</span>
                <span className="hint-line">penalty = e^(β × max(load/B − 1, 0))</span>
                <input
                  type="number"
                  step="0.1"
                  min={0.1}
                  value={configPenaltyBeta}
                  onChange={(e) => setConfigPenaltyBeta(Number(e.target.value))}
                />
              </label>

              <label className="control-label">
                <span>K-Paths</span>
                <span className="hint-line">Số đường thay thế mặc định</span>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={configKPaths}
                  onChange={(e) => setConfigKPaths(Number(e.target.value))}
                />
              </label>

              <label className="control-label">
                <span>K-Shortest Penalty</span>
                <span className="hint-line">Penalty khi tìm đường thay thế</span>
                <input
                  type="number"
                  step="100"
                  min={100}
                  value={configKShortestPenalty}
                  onChange={(e) => setConfigKShortestPenalty(Number(e.target.value))}
                />
              </label>

            </div>

            <div className="settings-actions">
              <button
                type="button"
                className="run-route-btn"
                onClick={saveConfig}
                disabled={configLoading}
              >
                {configLoading ? "Đang lưu..." : "Lưu cấu hình"}
              </button>
              <button
                type="button"
                className="secondary-btn"
                onClick={loadConfig}
              >
                Tải lại từ server
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;

function getFeaturePoint(feature: MapGeoJSONFeature): [number, number] | null {
  const geometry = feature.geometry;
  if (!geometry) return null;
  if (geometry.type === "Point") {
    return geometry.coordinates as [number, number];
  }
  if (geometry.type === "LineString") {
    const coords = geometry.coordinates as [number, number][];
    return coords[0] ?? null;
  }
  if (geometry.type === "MultiPoint" || geometry.type === "MultiLineString") {
    const coords = geometry.coordinates as [number, number][] | [number, number][][];
    return Array.isArray(coords) && coords.length ? (coords[0] as [number, number]) : null;
  }
  return null;
}

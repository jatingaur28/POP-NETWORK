"""
KDN Enterprise Northbound REST & WebSocket API Server.
Zero mock data. Integrates live Redis IPC caching, asynchronous database queries,
Prometheus observability, Segment Routing SID generation, and BGP FlowSpec controls.
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Header, Query, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field
import redis.asyncio as aioredis

# Ensure root path resolution for sdn_kdn module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from sdn_kdn.core.config import settings
from sdn_kdn.infrastructure.controllers import (
    EnterpriseRoutingController,
    SegmentRoutingController,
)
from sdn_kdn.infrastructure.traffic_eng import (
    TrafficEngineeringManager,
    QoSTrafficClass,
)

logger = logging.getLogger("NorthboundAPI")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [NORTHBOUND-API] - %(message)s")

# Global Controller Singletons
pce_controller = EnterpriseRoutingController()
sr_controller = SegmentRoutingController()
te_manager = TrafficEngineeringManager(pce_controller=pce_controller)

# Global Async Redis Pool
redis_pool: Optional[aioredis.Redis] = None


# ==============================================================================
# LIFECYCLE MANAGEMENT
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages Async Redis Connection Pool Lifecycle and IPC Health."""
    global redis_pool
    try:
        redis_pool = aioredis.from_url(
            f"redis://{settings.redis_host}:{settings.redis_port}",
            password=getattr(settings, "redis_password", None),
            decode_responses=True,
            max_connections=500,
            socket_timeout=2.0
        )
        await redis_pool.ping()
        logger.info("🟢 FastAPI Async Redis Connection Pool established.")
    except Exception as e:
        logger.critical("🚨 Failed to connect to Redis IPC broker on %s:%s: %s", settings.redis_host, settings.redis_port, str(e))
        redis_pool = None

    yield

    if redis_pool:
        await redis_pool.aclose()
        logger.info("🛑 FastAPI Async Redis Connection Pool closed.")


app = FastAPI(
    title="Autonomous Knowledge-Defined Network (KDN) API",
    version="4.2.0",
    description="Enterprise REST & WebSocket Northbound Interface for Real-Time POP Telemetry and Autonomous TE Routing.",
    lifespan=lifespan
)

# Enable CORS for NOC Web Interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# AUTHENTICATION & SECURITY DEPENDENCIES
# ==============================================================================
async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Enforces enterprise API key validation if configured in environment."""
    configured_key = getattr(settings, "pop_api_key", None)
    if configured_key and x_api_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header."
        )


# ==============================================================================
# PYDANTIC SCHEMAS FOR NORTHBOUND VALIDATION
# ==============================================================================
class OSPFOverrideIntent(BaseModel):
    target_link: str = Field(..., description="Unique link identifier (e.g., 'NYC-LON-01')")
    new_cost: int = Field(..., ge=1, le=65535, description="New OSPF metric value [1 - 65535]")
    reason: Optional[str] = Field("Operator manual intervention", description="Audit trail rationale")

class BGPFlowSpecIntent(BaseModel):
    target_link: str = Field(..., description="Link identifying target edge interface")
    protocol: str = Field("tcp", description="Protocol filter ('tcp', 'udp', 'icmp')")
    source_prefix: Optional[str] = Field(None, description="Malicious source CIDR prefix")
    destination_prefix: Optional[str] = Field(None, description="Protected destination CIDR prefix")
    action_type: str = Field("rate-limit", description="Action ('drop', 'rate-limit', 'redirect')")
    rate_bytes: int = Field(0, ge=0, description="Target bandwidth limit in bytes/sec (0 = discard)")
    reason: Optional[str] = Field("Volumetric DDoS mitigation", description="Security justification")

class ExternalTelemetryPush(BaseModel):
    link: str = Field(..., description="Physical link ID")
    bandwidth_gbps: float = Field(..., ge=0.0, description="Live throughput in Gbps")
    latency_ms: float = Field(..., ge=0.0, description="Measured RTT latency in milliseconds")
    optical_temp_c: Optional[float] = Field(None, description="Optical transceiver temperature in Celsius")
    interface_errors: Optional[float] = Field(0.0, description="Interface packet drop/CRC error count")


# ==============================================================================
# 1. CORE SYSTEM HEALTH & OBSERVABILITY
# ==============================================================================
@app.get("/health", tags=["System Governance"])
async def health_check() -> Dict[str, Any]:
    """Validates real-time connectivity across the Redis IPC broker and SQL inventory database."""
    db_status, db_err = "disconnected", None
    redis_status = "disconnected"

    # 1. Check SQL Database
    try:
        db_url = getattr(settings, "database_url", None)
        if db_url:
            from sqlalchemy import create_engine, text
            engine = create_engine(db_url, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_status = "connected"
        else:
            db_status = "not_configured"
    except Exception as exc:
        db_err = str(exc)

    # 2. Check Redis IPC
    if redis_pool:
        try:
            await redis_pool.ping()
            redis_status = "connected"
        except Exception:
            redis_status = "unreachable"

    overall_healthy = (redis_status == "connected") and (db_status in ("connected", "not_configured"))
    return {
        "status": "healthy" if overall_healthy else "degraded",
        "timestamp": time.time(),
        "ipc_message_broker": redis_status,
        "database_backend": db_status,
        "database_error": db_err,
        "pce_engine": "online"
    }


@app.get("/metrics", response_class=PlainTextResponse, tags=["System Governance"])
async def prometheus_metrics() -> str:
    """Exports live telemetry and SDN metrics in standard Prometheus text format for Grafana scraping."""
    lines = [
        "# HELP sdn_controller_status Operational status of the SDN controller",
        "# TYPE sdn_controller_status gauge",
        "sdn_controller_status 1.0"
    ]
    if not redis_pool:
        return "\n".join(lines)

    try:
        # Pull live telemetry cache
        raw_tel = await redis_pool.get("latest_telemetry_cache")
        if raw_tel:
            telemetry: List[Dict[str, Any]] = json.loads(raw_tel)
            lines.extend([
                "# HELP pop_link_bandwidth_gbps Live link bandwidth in Gbps",
                "# TYPE pop_link_bandwidth_gbps gauge",
                "# HELP pop_link_latency_ms Measured link latency in ms",
                "# TYPE pop_link_latency_ms gauge",
                "# HELP pop_link_ospf_cost Active hardware OSPF metric",
                "# TYPE pop_link_ospf_cost gauge"
            ])
            for t in telemetry:
                lid = t.get("link", "unknown")
                bw = float(t.get("bandwidth_gbps", 0.0))
                lat = float(t.get("latency_ms", 0.0))
                cost = float(t.get("ospf_cost", 10))
                lines.append(f'pop_link_bandwidth_gbps{{link="{lid}"}} {bw:.4f}')
                lines.append(f'pop_link_latency_ms{{link="{lid}"}} {lat:.2f}')
                lines.append(f'pop_link_ospf_cost{{link="{lid}"}} {cost:.0f}')
    except Exception as e:
        logger.error("Prometheus metrics export error: %s", str(e))

    return "\n".join(lines) + "\n"


# ==============================================================================
# 2. TOPOLOGY & GIS EXPORTS
# ==============================================================================
@app.get("/api/v1/topology", tags=["Network Topology"])
async def get_live_topology() -> Dict[str, Any]:
    """Fetches the current physical router/server network topology from live cache."""
    if not redis_pool:
        raise HTTPException(status_code=503, detail="Redis IPC broker offline.")
    data = await redis_pool.get("kdn_latest_topology")
    if not data:
        raise HTTPException(status_code=404, detail="No active physical topology found in database cache.")
    return json.loads(data)


@app.get("/api/v1/export/topology.geojson", tags=["Network Topology"])
async def export_topology_geojson() -> Dict[str, Any]:
    """Exports physical nodes and links as an RFC 7946 GeoJSON FeatureCollection for GIS visualization."""
    if not redis_pool:
        raise HTTPException(status_code=503, detail="Redis IPC broker offline.")
    raw_topo = await redis_pool.get("kdn_latest_topology")
    if not raw_topo:
        raise HTTPException(status_code=404, detail="Topology cache empty.")

    topo = json.loads(raw_topo)
    nodes = topo.get("nodes", {})
    links = topo.get("links", {})

    features: List[Dict[str, Any]] = []

    # 1. Point Features (Router Nodes)
    for nid, ndata in nodes.items():
        lon = float(ndata.get("lon", ndata.get("longitude", 0.0)))
        lat = float(ndata.get("lat", ndata.get("latitude", 0.0)))
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "id": nid,
                "label": ndata.get("label", nid),
                "tier": ndata.get("tier", 2),
                "loopback_ip": ndata.get("loopback_ip", "N/A")
            }
        })

    # 2. LineString Features (Fiber Links)
    for lid, ldata in links.items():
        src_id = ldata.get("source") or ldata.get("src")
        dst_id = ldata.get("target") or ldata.get("dst")
        src_node = nodes.get(src_id)
        dst_node = nodes.get(dst_id)

        if src_node and dst_node:
            src_coord = [float(src_node.get("lon", 0.0)), float(src_node.get("lat", 0.0))]
            dst_coord = [float(dst_node.get("lon", 0.0)), float(dst_node.get("lat", 0.0))]
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [src_coord, dst_coord]},
                "properties": {
                    "id": lid,
                    "source": src_id,
                    "target": dst_id,
                    "max_bw_gbps": float(ldata.get("max_bw_gbps", 10.0)),
                    "ospf_cost": int(ldata.get("ospf_cost", 10))
                }
            })

    return {"type": "FeatureCollection", "features": features}


# ==============================================================================
# 3. LIVE HARDWARE TELEMETRY STREAM & PUSH
# ==============================================================================
@app.get("/api/v1/telemetry", tags=["Telemetry Ingestion"])
async def get_live_telemetry() -> List[Dict[str, Any]]:
    """Retrieves the latest batched hardware SNMP/gNMI telemetry frame."""
    if not redis_pool:
        raise HTTPException(status_code=503, detail="Redis IPC broker offline.")
    data = await redis_pool.get("latest_telemetry_cache")
    if not data:
        raise HTTPException(status_code=404, detail="No hardware telemetry available from pollers.")
    return json.loads(data)


@app.post("/api/v1/telemetry/push", status_code=status.HTTP_202_ACCEPTED, tags=["Telemetry Ingestion"])
async def push_external_telemetry(payload: ExternalTelemetryPush, _: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Allows external monitoring agents (Telegraf, Prometheus exporters) to push live metrics."""
    if not redis_pool:
        raise HTTPException(status_code=503, detail="Redis IPC broker offline.")
    
    # Broadcast pushed telemetry over IPC stream
    telemetry_item = payload.model_dump()
    telemetry_item["timestamp"] = time.time()
    await redis_pool.publish("kdn_live_telemetry", json.dumps([telemetry_item]))
    
    return {"status": "accepted", "ingested_link": payload.link}


# ==============================================================================
# 4. PATH COMPUTATION ELEMENT (PCE) & SEGMENT ROUTING
# ==============================================================================
@app.get("/api/v1/pce/optimal-path", tags=["Path Computation Element"])
async def get_optimal_path(source: str = Query(..., description="Source router node"), 
                           destination: str = Query(..., description="Destination router node")) -> Dict[str, Any]:
    """Computes Dijkstra shortest path strictly using active hardware OSPF metrics."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, pce_controller.compute_optimal_path, source, destination)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/pce/segment-routing", tags=["Path Computation Element"])
async def get_segment_routing(source: str = Query(...), destination: str = Query(...)) -> Dict[str, Any]:
    """Computes strict SR-MPLS label stacks and RFC-compliant SRv6 locators for end-to-end steering."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, sr_controller.compute_strict_path, pce_controller, source, destination)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==============================================================================
# 5. TRAFFIC ENGINEERING & QOS SLICING
# ==============================================================================
@app.get("/api/v1/traffic-eng/wcmp-split", tags=["Traffic Engineering"])
async def get_wcmp_split(source: str = Query(...), 
                         destination: str = Query(...), 
                         max_paths: int = Query(3, ge=1, le=8), 
                         temperature: float = Query(2.0, ge=0.1, le=10.0)) -> Dict[str, Any]:
    """Calculates Weighted Cost Multi-Path (WCMP) traffic allocation ratios across k-shortest paths."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, te_manager.compute_wcmp_split, source, destination, max_paths, temperature
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/traffic-eng/qos-policy", tags=["Traffic Engineering"])
async def get_qos_policy(source: str = Query(...), 
                         destination: str = Query(...), 
                         qos_tier: str = Query("VIP_LOW_LATENCY", description="'VIP_LOW_LATENCY', 'REAL_TIME_MEDIA', 'BULK_TRANSFER'")) -> Dict[str, Any]:
    """Generates an intent-based DSCP QoS policy mapped to Segment Routing or WCMP multi-pathing."""
    try:
        qos_enum = QoSTrafficClass[qos_tier]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid QoS tier '{qos_tier}'. Valid: {[e.name for e in QoSTrafficClass]}")

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, te_manager.build_qos_traffic_policy, source, destination, qos_enum)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==============================================================================
# 6. ROUTING CONTROL & BGP FLOWSPEC MITIGATION
# ==============================================================================
@app.post("/api/v1/intents/override", status_code=status.HTTP_202_ACCEPTED, tags=["Routing Control"])
async def manual_routing_override(payload: OSPFOverrideIntent, _: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Publishes a human operator OSPF metric override intent to the hardware routing engine."""
    if not redis_pool:
        raise HTTPException(status_code=503, detail="Redis IPC broker offline.")
    
    intent = {
        "action": "human_override",
        "costs": {payload.target_link: payload.new_cost},
        "reason": payload.reason,
        "timestamp": time.time()
    }
    await redis_pool.publish("kdn_routing_intents", json.dumps(intent))
    logger.info("Published operator override intent: %s ➔ Cost %d", payload.target_link, payload.new_cost)
    return {"status": "published_to_ipc", "intent": intent}


@app.post("/api/v1/intents/bgp-flowspec", status_code=status.HTTP_202_ACCEPTED, tags=["Security & Mitigation"])
async def inject_bgp_flowspec(payload: BGPFlowSpecIntent, _: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Injects a BGP FlowSpec security rule to rate-limit or drop hostile traffic at the edge."""
    if not redis_pool:
        raise HTTPException(status_code=503, detail="Redis IPC broker offline.")

    intent = {
        "action": "bgp_flowspec_mitigation",
        "target_link": payload.target_link,
        "protocol": payload.protocol,
        "source_prefix": payload.source_prefix,
        "destination_prefix": payload.destination_prefix,
        "action_type": payload.action_type,
        "rate_bytes": payload.rate_bytes,
        "reason": payload.reason,
        "timestamp": time.time()
    }
    await redis_pool.publish("kdn_routing_intents", json.dumps(intent))
    logger.warning("🚨 Injected BGP FlowSpec Mitigation on %s (Action: %s)", payload.target_link, payload.action_type)
    return {"status": "flowspec_published", "intent": intent}


# ==============================================================================
# 7. WEBSOCKET REAL-TIME TELEMETRY STREAM
# ==============================================================================
@app.websocket("/ws/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket):
    """
    Sub-second asynchronous WebSocket stream.
    Broadcasts live hardware telemetry frames directly to dashboards without polling.
    """
    await websocket.accept()
    logger.info("🔌 WebSocket Client connected to live telemetry stream: %s", websocket.client)

    if not redis_pool:
        await websocket.send_json({"error": "Redis IPC broker offline."})
        await websocket.close()
        return

    pubsub = redis_pool.pubsub()
    await pubsub.subscribe("kdn_live_telemetry")

    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message.get("type") == "message":
                data = json.loads(message["data"])
                await websocket.send_json(data)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket Client disconnected: %s", websocket.client)
    except Exception as e:
        logger.error("WebSocket stream error: %s", str(e))
    finally:
        await pubsub.unsubscribe("kdn_live_telemetry")


# ==============================================================================
# STANDALONE EXECUTION
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Launching Production Northbound REST & WebSocket API on %s:%d...", settings.api_host, settings.api_port)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info")
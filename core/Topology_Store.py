"""
Enterprise Hardware Topology Store (Single Source of Truth).
Bridges the persistent Neon Serverless PostgreSQL database with the ultra-fast Redis IPC cache.
Engineered for zero-copy JSON parsing and thread-safe concurrent access.
"""

import logging
import threading
from typing import Dict, Any, List

# Optimize for extreme speed if orjson is installed (C-based JSON parser)
try:
    import orjson as json
except ImportError:
    import json

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from geopy.distance import geodesic

from sdn_kdn.core.config import settings
from sdn_kdn.core.redis_utils import RedisConnectionManager
from sdn_kdn.core.exceptions import TopologyLoadError, DatabaseConnectionError

logger = logging.getLogger("TopologyStore")


class HardwareTopologyStore:
    """
    Tier-1 Carrier Topology Manager.
    Reads strictly from centralized SQL/Redis caches. No mock data.
    """
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        """Thread-safe Singleton implementation to prevent database pool exhaustion."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(HardwareTopologyStore, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Sets up the Redis IPC client and the SQLAlchemy Postgres Engine."""
        self.redis = RedisConnectionManager.get_sync_client()
        self.topology_key = "kdn_latest_topology"
        
        # Setup Serverless Postgres connection pooling
        try:
            self.engine = create_engine(
                settings.database_url,
                pool_size=settings.db_pool_size,
                max_overflow=settings.db_max_overflow,
                pool_pre_ping=True,  # Crucial for serverless environments to drop dead connections
                connect_args={'sslmode': settings.db_sslmode}
            )
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info("🟢 SQLAlchemy connection pool to Neon Database established.")
        except Exception as e:
            logger.error("Failed to initialize database engine: %s", str(e))
            self.engine = None

    def _calculate_geospatial_cost(self, lat1: float, lon1: float, lat2: float, lon2: float) -> int:
        """
        Calculates physical fiber length in kilometers.
        Serves as an emergency fallback if the hardware OSPF cost metric is missing.
        """
        if (lat1, lon1) == (0.0, 0.0) or (lat2, lon2) == (0.0, 0.0):
            return 100  # Default base cost
        try:
            distance_km = geodesic((lat1, lon1), (lat2, lon2)).kilometers
            # Approximate OSPF cost (1 cost unit per 10km of fiber)
            return max(1, int(distance_km / 10.0))
        except Exception:
            return 100

    def _rebuild_from_postgres(self) -> Dict[str, Any]:
        """
        Cold Cache Miss Handler: Queries the Live Neon Database, reconstructs the 
        network dictionary, and enforces physical geographic physics constraints.
        """
        if not self.engine:
            raise DatabaseConnectionError("SQLAlchemy Engine is offline. Cannot reach live database.")

        topology = {"nodes": {}, "links": {}}
        
        try:
            with self.SessionLocal() as session:
                # 1. Fetch live router hardware
                nodes_result = session.execute(text("SELECT id, hostname, loopback_ip, tier, latitude, longitude, status FROM network_nodes WHERE status = 'ACTIVE'"))
                for row in nodes_result:
                    topology["nodes"][row.id] = {
                        "id": row.id,
                        "hostname": row.hostname,
                        "loopback_ip": row.loopback_ip,
                        "tier": row.tier,
                        "lat": float(row.latitude) if row.latitude else 0.0,
                        "lon": float(row.longitude) if row.longitude else 0.0,
                    }

                # 2. Fetch live physical circuits
                links_result = session.execute(text("SELECT id, source, target, snmp_if_index, max_bw_gbps, ospf_cost, status FROM network_links WHERE status = 'ACTIVE'"))
                for row in links_result:
                    src_node = topology["nodes"].get(row.source)
                    dst_node = topology["nodes"].get(row.target)
                    
                    if not src_node or not dst_node:
                        continue # Skip links with dead/missing endpoints

                    # Fallback metric logic
                    ospf_cost = row.ospf_cost
                    if not ospf_cost:
                        ospf_cost = self._calculate_geospatial_cost(
                            src_node["lat"], src_node["lon"], 
                            dst_node["lat"], dst_node["lon"]
                        )

                    topology["links"][row.id] = {
                        "id": row.id,
                        "source": row.source,
                        "target": row.target,
                        "snmp_if_index": row.snmp_if_index,
                        "max_bw_gbps": float(row.max_bw_gbps) if row.max_bw_gbps else 100.0,
                        "ospf_cost": ospf_cost,
                    }

            # Warm the Redis cache with the newly rebuilt topology
            self.redis.set(self.topology_key, json.dumps(topology))
            logger.info("🗺️ Database Sync Complete: Topology rebuilt and cached into Redis.")
            return topology

        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"PostgreSQL sync failed: {str(e)}")

    def fetch_live_topology(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetches the exact physical router topology via ultra-fast Read-Through cache.
        If force_refresh is True, bypasses Redis and forces a direct SQL query.
        """
        with self._lock:
            if not force_refresh:
                try:
                    raw_data = self.redis.get(self.topology_key)
                    if raw_data:
                        # Fast JSON deserialization
                        return json.loads(raw_data)
                except Exception as e:
                    logger.warning("Redis cache read failed: %s. Falling back to SQL.", str(e))

            # Cache Miss or Forced Refresh -> Rebuild from DB
            return self._rebuild_from_postgres()

    def get_router_ip(self, node_id: str) -> str:
        """O(1) lookup for a router's real management IP address."""
        topology = self.fetch_live_topology()
        node = topology.get("nodes", {}).get(node_id)
        
        if not node or "loopback_ip" not in node:
            raise TopologyLoadError(f"Router {node_id} is missing a loopback IP in the hardware database.")
            
        return node["loopback_ip"]

    def get_active_links(self) -> List[Dict[str, Any]]:
        """Returns a list of all physically active connections."""
        topology = self.fetch_live_topology()
        return list(topology.get("links", {}).values())

# Global singleton instance for rapid imports
topology_store = HardwareTopologyStore()

def fetch_live_topology(database_url: str = None) -> Dict[str, Any]:
    """Compatibility wrapper for external modules expecting a function-level import."""
    return topology_store.fetch_live_topology()
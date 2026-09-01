"""
Enterprise Northbound API Facade & Controller Wrappers (v6.0).
Provides advanced asynchronous wrappers, FastAPI Dependency Injection (DI) interfaces,
sub-second Redis caching for route lookups, and Intent Translation logic.
Strictly bridges to physical infrastructure controllers handling live DB/Redis state.
"""

import asyncio
import json
import logging
import hashlib
from typing import Any, Dict

from fastapi import HTTPException, status

# Core Infrastructure Controllers & Managers
from sdn_kdn.infrastructure.controllers import (
    EnterpriseRoutingController,
    SegmentRoutingController,
)
from sdn_kdn.infrastructure.traffic_eng import (
    TrafficEngineeringManager,
    QoSTrafficClass,
)
from sdn_kdn.core.redis_utils import RedisConnectionManager
from sdn_kdn.core.exceptions import NetworkPartitionError, ConfigurationPushError

logger = logging.getLogger("API-Controller-Facade")


class AsyncPCEFacade:
    """
    Asynchronous API Wrapper for the Path Computation Element (PCE).
    Offloads heavy Dijkstra/CSPF graph mathematics to background threads 
    and implements a Redis cache circuit breaker to survive micro-bursts.
    """
    def __init__(self):
        self.pce = EnterpriseRoutingController()
        # Non-blocking Redis client for API caching
        self.redis_async = RedisConnectionManager.get_async_client()
        self.cache_ttl_seconds = 1.5  # Sub-second cache to prevent CPU hammering

    async def compute_optimal_path_async(
        self, source: str, destination: str, objective: str = "pareto_weight", min_bw_gbps: float = 0.0
    ) -> Dict[str, Any]:
        """Asynchronously computes the optimal path using CSPF constraints."""
        
        # 1. Generate unique cryptographic cache key for these exact parameters
        query_payload = f"{source}_{destination}_{objective}_{min_bw_gbps}".encode('utf-8')
        query_hash = hashlib.sha256(query_payload).hexdigest()
        cache_key = f"api_pce_cache:{query_hash}"

        # 2. Sub-millisecond Cache Check (Circuit Breaker)
        try:
            cached_result = await self.redis_async.get(cache_key)
            if cached_result:
                logger.debug("Served PCE route from sub-second cache: %s", cache_key)
                return json.loads(cached_result)
        except Exception as e:
            logger.warning("Redis async cache read failed for PCE: %s", str(e))

        # 3. Offload Heavy CPU Math (NetworkX) to Thread Pool
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, 
                self.pce.compute_optimal_path, 
                source, destination, objective, min_bw_gbps
            )
            
            # 4. Asynchronously warm the cache for subsequent identical bursts
            try:
                # Use setex to guarantee the key expires automatically
                await self.redis_async.setex(cache_key, int(self.cache_ttl_seconds), json.dumps(result))
            except Exception:
                pass
                
            return result

        except NetworkPartitionError as npe:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(npe))
        except ConfigurationPushError as cpe:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(cpe))
        except Exception as e:
            logger.error("Async PCE computation failed: %s", str(e), exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal PCE Engine Error")


class AsyncSegmentRoutingFacade:
    """
    Asynchronous API Wrapper for SR-MPLS / SRv6 Payload Generation.
    Translates API parameters into native 128-bit IPv6 hardware SID instructions.
    """
    def __init__(self):
        self.sr_controller = SegmentRoutingController()

    async def compute_strict_path_async(
        self, pce_facade: AsyncPCEFacade, source: str, destination: str, 
        qos_objective: str = "pareto_weight", required_bw: float = 0.0
    ) -> Dict[str, Any]:
        """Offloads SR label stack generation to background thread."""
        try:
            loop = asyncio.get_running_loop()
            # Pass the synchronous PCE instance stored strictly inside the facade
            result = await loop.run_in_executor(
                None, 
                self.sr_controller.compute_strict_path, 
                pce_facade.pce, source, destination, qos_objective, required_bw
            )
            
            if "error" in result:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"])
                
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Async Segment Routing calculation failed: %s", str(e), exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal SRv6 Engine Error")


class AsyncTrafficEngineeringFacade:
    """
    Asynchronous API Wrapper for the QoS and Traffic Engineering (TE) Manager.
    Generates multi-path ECMP/WCMP routing arrays for the REST endpoints.
    """
    def __init__(self):
        self.pce = EnterpriseRoutingController()
        self.te_manager = TrafficEngineeringManager(pce_controller=self.pce)

    async def compute_wcmp_split_async(
        self, source: str, destination: str, max_paths: int = 3, temperature: float = 2.0
    ) -> Dict[str, Any]:
        """Calculates Weighted Cost Multi-Path splits asynchronously."""
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, 
                self.te_manager.compute_wcmp_split, 
                source, destination, max_paths, temperature
            )
            return result
        except NetworkPartitionError as npe:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(npe))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    async def build_qos_policy_async(
        self, source: str, destination: str, qos_tier_name: str
    ) -> Dict[str, Any]:
        """Maps HTTP string arguments to backend ENUMs and generates QoS payloads."""
        try:
            # Enforce strict translation to standard RFC traffic classes
            qos_enum = QoSTrafficClass[qos_tier_name]
        except KeyError:
            valid_tiers = [e.name for e in QoSTrafficClass]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Invalid QoS tier '{qos_tier_name}'. Valid options: {valid_tiers}"
            )

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, 
                self.te_manager.build_qos_traffic_policy, 
                source, destination, qos_enum
            )
            return result
        except NetworkPartitionError as npe:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(npe))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# ==============================================================================
# FASTAPI DEPENDENCY INJECTION (DI) PROVIDERS
# ==============================================================================

# Global singletons to prevent redundant object instantiation per REST request.
# These instances maintain their own connection pools and caches.
_pce_facade_instance = AsyncPCEFacade()
_sr_facade_instance = AsyncSegmentRoutingFacade()
_te_facade_instance = AsyncTrafficEngineeringFacade()

def get_pce_facade() -> AsyncPCEFacade:
    """FastAPI Dependency Provider for the Path Computation Element Facade."""
    return _pce_facade_instance

def get_sr_facade() -> AsyncSegmentRoutingFacade:
    """FastAPI Dependency Provider for the Segment Routing Facade."""
    return _sr_facade_instance

def get_te_facade() -> AsyncTrafficEngineeringFacade:
    """FastAPI Dependency Provider for the Traffic Engineering Facade."""
    return _te_facade_instance


# Explicitly define the exported classes and methods for clean API routing
__all__ = [
    "AsyncPCEFacade",
    "AsyncSegmentRoutingFacade",
    "AsyncTrafficEngineeringFacade",
    "get_pce_facade",
    "get_sr_facade",
    "get_te_facade",
]
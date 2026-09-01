"""
Enterprise Application-Aware Telemetry (Omnissa VDI).
Bridges OSI Layer 7 Application metrics with Layer 3 Routing Intelligence.
Detects VDI bandwidth storms and broadcasts preemptive warnings via Redis IPC.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional

import aiohttp
from aiohttp_retry import RetryClient, ExponentialRetry

import sys
import os

# Ensure root path resolution for sdn_kdn module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from sdn_kdn.core.config import settings
from sdn_kdn.core.redis_utils import RedisConnectionManager
from sdn_kdn.core.topology_store import HardwareTopologyStore

logger = logging.getLogger("Omnissa-Application-Aware")


class OmnissaVDIClient:
    """
    Asynchronous REST Client for Omnissa (VMware Horizon) environments.
    Fetches live session bandwidth and correlates application spikes 
    with physical router paths to enable predictive QoS slicing.
    """
    def __init__(self, api_url: str, client_id: str, client_secret: str):
        self.api_url = api_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        
        self.redis = RedisConnectionManager.get_async_client()
        self.topology_store = HardwareTopologyStore()
        
        # Oauth2 Token Caching
        self._auth_token: Optional[str] = None
        self._token_expiry: float = 0.0

        # Exponential backoff for resilient API communication
        self.retry_options = ExponentialRetry(attempts=3, start_timeout=1.0, max_timeout=5.0)

    async def _authenticate(self) -> str:
        """Handles OAuth2 Client Credentials grant with caching."""
        if self._auth_token and time.time() < self._token_expiry:
            return self._auth_token

        auth_endpoint = f"{self.api_url}/SAAS/auth/oauthtoken"
        payload = {"grant_type": "client_credentials"}
        auth_str = aiohttp.BasicAuth(self.client_id, self.client_secret)

        async with aiohttp.ClientSession() as session:
            async with session.post(auth_endpoint, data=payload, auth=auth_str, timeout=5.0) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._auth_token = data.get("access_token")
                    # Cache token slightly less than expiry to prevent mid-flight invalidation
                    self._token_expiry = time.time() + float(data.get("expires_in", 3600)) - 60
                    return self._auth_token
                else:
                    logger.error("Omnissa OAuth2 Failure: %s", await resp.text())
                    raise RuntimeError("Failed to authenticate with Omnissa VDI API.")

    async def fetch_active_vdi_sessions(self) -> List[Dict[str, Any]]:
        """
        Queries the Omnissa API for active streaming sessions and their Tx/Rx byte rates.
        Uses Circuit-Breaker retries to survive temporary API outages.
        """
        try:
            token = await self._authenticate()
        except Exception:
            return []

        endpoint = f"{self.api_url}/inventory/v1/sessions"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            async with RetryClient(retry_options=self.retry_options) as client:
                async with client.get(endpoint, headers=headers, timeout=5.0) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("results", [])
                    else:
                        logger.warning("Omnissa API returned non-200 status: %d", response.status)
                        return []
        except Exception as e:
            logger.error("Failed to fetch VDI telemetry: %s", str(e))
            return []
            
    async def correlate_and_broadcast(self):
        """
        Master Loop: Maps Layer 7 VDI load directly to Layer 3 hardware links.
        If a VDI load exceeds the safe threshold, alerts the AI via Redis IPC.
        """
        logger.info("🖥️ Application-Aware VDI Correlator Online.")
        
        while True:
            try:
                vdi_sessions = await self.fetch_active_vdi_sessions()
                if not vdi_sessions:
                    await asyncio.sleep(10)
                    continue

                # Fetch physical hardware state to cross-reference
                active_links = self.topology_store.get_active_links()
                link_map = {link["id"]: link for link in active_links}
                
                # Aggregate VDI bandwidth by the region/datacenter they are hosted in
                regional_load_gbps = {}
                for session in vdi_sessions:
                    region = session.get("datacenter_region", "UNKNOWN")
                    # Convert Rx Bytes to Gbps
                    rx_bytes = session.get("metrics", {}).get("rx_bytes_per_sec", 0)
                    gbps = (rx_bytes * 8) / 1e9
                    regional_load_gbps[region] = regional_load_gbps.get(region, 0.0) + gbps

                # Check if VDI load threatens any physical fiber links in that region
                for region, load in regional_load_gbps.items():
                    # For advanced use cases: Map 'region' to physical router interfaces
                    # Here we search for links terminating in the affected region
                    affected_links = [l for l in link_map.values() if region in l["id"]]
                    
                    for link in affected_links:
                        max_cap = float(link.get("max_bw_gbps", 10.0))
                        
                        # If VDI traffic alone is consuming > 40% of the link, warn the AI
                        if (load / max_cap) > 0.40:
                            logger.warning("🚨 VDI Storm Detected in %s! Load: %.2f Gbps", region, load)
                            
                            intent = {
                                "action": "application_qos_alert",
                                "target_link": link["id"],
                                "app_type": "OMNISSA_VDI",
                                "vdi_load_gbps": round(load, 2),
                                "urgency": "HIGH",
                                "timestamp": time.time()
                            }
                            # Broadcast to the AI Master Loop so it can preemptively slice QoS
                            await self.redis.publish("kdn_routing_intents", json.dumps(intent))

                await asyncio.sleep(15) # Poll VDI load every 15 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("VDI Correlator encountered an error: %s", str(e))
                await asyncio.sleep(10)
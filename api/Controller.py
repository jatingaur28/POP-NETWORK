"""
Enterprise Constrained Path Computation Element (PCE) and Segment Routing Controller.
Calculates strict CSPF, Edge-Disjoint failover paths, and BGP-LS compliant SR stacks.
Operates exclusively on live hardware state and real-time telemetry from Redis.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import redis

# Ensure root path resolution for sdn_kdn module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

try:
    from sdn_kdn.core.config import settings
    DEFAULT_REDIS_HOST = settings.redis_host
    DEFAULT_REDIS_PORT = settings.redis_port
except ImportError:
    DEFAULT_REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
    DEFAULT_REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

try:
    from sdn_kdn.core.exceptions import ConfigurationPushError, NetworkPartitionError
except ImportError:
    class NetworkPartitionError(Exception): pass
    class ConfigurationPushError(Exception): pass

try:
    from sdn_kdn.core.topology_store import fetch_live_topology
except ImportError:
    def fetch_live_topology(database_url=None):
        return {"nodes": {}, "links": {}}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [PCE-CONTROLLER] - %(message)s")
logger = logging.getLogger("Enterprise-PCE")


class EnterpriseRoutingController:
    """
    Advanced Path Computation Element (PCE).
    Integrates live telemetry to prune congested links dynamically (CSPF)
    and computes edge-disjoint redundant paths for high availability.
    """
    def __init__(self, redis_host: str = DEFAULT_REDIS_HOST, redis_port: int = DEFAULT_REDIS_PORT):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )

    def _build_live_graph(self, min_required_bw_gbps: float = 0.0) -> nx.DiGraph:
        """
        Constructs the physical NetworkX graph directly from live DB state and telemetry.
        Dynamically prunes links lacking the requested residual bandwidth (CSPF).
        """
        graph = nx.DiGraph()
        try:
            raw_state = self.redis_client.get("kdn_latest_topology")
            raw_telemetry = self.redis_client.get("latest_telemetry_cache")
            
            state = json.loads(raw_state) if raw_state else fetch_live_topology()
            live_telemetry = {t["link"]: t for t in json.loads(raw_telemetry)} if raw_telemetry else {}

            # Add Physical Router Nodes
            for node_id, data in state.get("nodes", {}).items():
                graph.add_node(str(node_id), **data)

            # Add Physical Fiber Links
            for link_id, data in state.get("links", {}).items():
                src = str(data.get("source") or data.get("src"))
                dst = str(data.get("target") or data.get("dst"))
                
                if not src or not dst:
                    continue

                # Merge live hardware telemetry
                t_data = live_telemetry.get(link_id, {})
                max_bw = float(data.get("max_bw_gbps", 100.0))
                current_util = float(t_data.get("bandwidth_gbps", 0.0))
                residual_bw = max(0.0, max_bw - current_util)

                # CSPF Pruning: Drop link if it cannot guarantee requested SLA
                if min_required_bw_gbps > 0.0 and residual_bw < min_required_bw_gbps:
                    continue

                base_cost = float(data.get("ospf_cost", 10.0))
                latency = float(t_data.get("latency_ms", data.get("latency_ms", 10.0)))
                
                # Pareto-Optimal Weight Blend (Cost, Latency, Congestion penalty)
                utilization_penalty = 1.0 + (current_util / max(0.001, max_bw)) ** 2
                pareto_weight = (base_cost * 0.4) + (latency * 0.4) * utilization_penalty

                graph.add_edge(
                    src, dst, 
                    weight=base_cost,           # Standard OSPF
                    latency=latency,            # Ultra-Low Latency
                    pareto_weight=pareto_weight,# AI Blended Weight
                    residual_bw=residual_bw,
                    **data
                )

        except Exception as e:
            logger.error("Live network graph compilation failed: %s", str(e))
            raise NetworkPartitionError(f"CSPF graph construction failed: {e}")

        return graph

    def compute_optimal_path(self, source: str, destination: str, 
                             objective: str = "weight", min_bw_gbps: float = 0.0) -> Dict[str, Any]:
        """
        Computes the optimal path based on dynamic routing objectives.
        Objectives: 'weight' (OSPF), 'latency' (Speed), 'pareto_weight' (Blended AI Cost).
        """
        graph = self._build_live_graph(min_required_bw_gbps=min_bw_gbps)

        if source not in graph or destination not in graph:
            raise ConfigurationPushError(f"Hardware nodes '{source}' or '{destination}' are down or missing.")

        try:
            # Primary CSPF Route
            path = nx.shortest_path(graph, source=source, target=destination, weight=objective)
            primary_cost = float(nx.path_weight(graph, path, weight="weight"))
            
            # Extract cumulative latency
            total_latency = sum(float(graph.get_edge_data(u, v).get("latency", 0.0)) for u, v in zip(path[:-1], path[1:]))

            # Disjoint Backup Route (Fast Reroute / FRR)
            backup_path = []
            try:
                # Find completely physically separated fiber paths
                disjoint_paths = list(nx.edge_disjoint_paths(graph, source, target=destination))
                if len(disjoint_paths) > 1:
                    # Sort disjoint paths by our routing objective and pick the second best
                    disjoint_paths.sort(key=lambda p: nx.path_weight(graph, p, weight=objective))
                    backup_path = disjoint_paths[1]
            except nx.NetworkXNoPath:
                logger.warning("No disjoint backup path available for %s -> %s", source, destination)

            return {
                "source": source,
                "destination": destination,
                "routing_objective": objective,
                "primary_path": path,
                "backup_path": backup_path,
                "total_ospf_cost": primary_cost,
                "total_expected_latency_ms": round(total_latency, 2),
                "timestamp": time.time()
            }
        except nx.NetworkXNoPath:
            raise NetworkPartitionError(f"No viable physical path meets the SLA between {source} and {destination}.")


class SegmentRoutingController:
    """
    Segment Routing Controller (SR-MPLS / SRv6).
    Translates mathematical paths into exact hardware BGP-LS payload stacks.
    """
    
    def _extract_hardware_sid(self, node_id: str, graph: nx.DiGraph, srv6: bool = False) -> str:
        """Looks up the strictly provisioned SID from the hardware database."""
        node_data = graph.nodes.get(node_id, {})
        if srv6:
            # Fetch standard 128-bit IPv6 locator
            return node_data.get("srv6_locator", f"UNPROVISIONED_SRv6_{node_id}")
        
        # Fetch standard MPLS label (SRGB)
        return str(node_data.get("sr_node_sid", f"UNPROVISIONED_MPLS_{node_id}"))

    def compute_strict_path(
        self, routing_controller: EnterpriseRoutingController, source: str, destination: str,
        qos_objective: str = "pareto_weight", required_bw: float = 0.0
    ) -> Dict[str, Any]:
        """
        Builds a highly available, hardware-deployable Segment Routing intent containing
        both the Primary active forwarding path and the FRR (Fast Re-Route) backup path.
        """
        try:
            route_data = routing_controller.compute_optimal_path(
                source, destination, objective=qos_objective, min_bw_gbps=required_bw
            )
            graph = routing_controller._build_live_graph()
        except Exception as e:
            return {"error": str(e)}

        primary_nodes = route_data["primary_path"]
        backup_nodes = route_data.get("backup_path", [])

        # Build Primary Stacks (Exclude Ingress Router)
        primary_mpls = [self._extract_hardware_sid(n, graph, srv6=False) for n in primary_nodes[1:]]
        primary_srv6 = [self._extract_hardware_sid(n, graph, srv6=True) for n in primary_nodes[1:]]

        # Build Backup Stacks for Topology Independent Loop-Free Alternate (TI-LFA)
        backup_mpls = [self._extract_hardware_sid(n, graph, srv6=False) for n in backup_nodes[1:]] if backup_nodes else []
        backup_srv6 = [self._extract_hardware_sid(n, graph, srv6=True) for n in backup_nodes[1:]] if backup_nodes else []

        return {
            "intent": "autonomous_traffic_engineering",
            "traffic_class": "strict_sla_enforcement",
            "required_bandwidth_gbps": required_bw,
            "primary_path_metrics": {
                "path_hops": primary_nodes,
                "latency_ms": route_data["total_expected_latency_ms"],
                "ospf_cost": route_data["total_ospf_cost"]
            },
            "segment_routing_payload": {
                "mpls": {
                    "primary_sid_stack": primary_mpls,
                    "backup_frr_sid_stack": backup_mpls
                },
                "srv6": {
                    "primary_locator_list": primary_srv6,
                    "backup_frr_locator_list": backup_srv6
                }
            },
            "timestamp": time.time()
        }
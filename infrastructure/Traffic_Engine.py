"""
Enterprise Traffic Engineering (TE) Manager.
Executes High-Speed Vectorized Weighted Cost Multi-Path (WCMP) mathematics,
Intent-Based QoS Slicing, and Dynamic Congestion Penalties for live hardware.
"""

import logging
from enum import Enum
from typing import Dict, Any, List, Optional

import numpy as np

try:
    from sdn_kdn.core.exceptions import DataShapeError, NetworkPartitionError
except ImportError:
    class DataShapeError(Exception): pass
    class NetworkPartitionError(Exception): pass

logger = logging.getLogger("TrafficEngineering")


class QoSTrafficClass(Enum):
    """
    Enterprise Intent-Based Slicing Classes.
    Dictates the mathematical weighting strategy for the Path Computation Element.
    """
    VIP_LOW_LATENCY = "vip_low_latency"     # HFT/Voice: Ignores cost, minimizes latency
    BULK_TRANSFER = "bulk_transfer"         # Nightly Backups: Ignores latency, minimizes cost
    STANDARD = "standard"                   # Web/Standard: Pareto-optimal balance
    GREEN_ENERGY = "green_energy"           # Eco Routing: Minimizes physical optical power/carbon


class TrafficEngineeringManager:
    """
    High-Speed mathematical utility for Traffic Engineering.
    Uses vectorized NumPy math to calculate exact traffic splitting ratios 
    and multi-objective Pareto routing weights based on live hardware telemetry.
    """
    def __init__(self, pce_controller: Optional[Any] = None):
        # Optional linkage to the active Path Computation Element for deeper graph traversal
        self.pce = pce_controller

    @staticmethod
    def _apply_live_congestion_penalty(path_metrics: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extracts live utilization from the path metrics and calculates an exponential 
        penalty multiplier. Forces the WCMP Softmax to aggressively drain traffic 
        from paths that are physically nearing hardware saturation (>80%).
        """
        penalties = np.ones(len(path_metrics), dtype=np.float32)
        
        for i, metrics in enumerate(path_metrics):
            max_cap = float(metrics.get("max_capacity_gbps", 10.0))
            live_bw = float(metrics.get("live_bandwidth_gbps", 0.0))
            
            # Avoid division by zero
            util_pct = np.clip(live_bw / max(0.001, max_cap), 0.0, 1.0)
            
            if util_pct > 0.80:
                # Exponential penalty curve for links nearing saturation
                # e.g., 85% util = 1.6x penalty, 95% util = 4.4x penalty
                penalties[i] = float(np.exp(10.0 * (util_pct - 0.80)))
                
        return penalties

    def compute_wcmp_split(
        self, 
        source: str, 
        destination: str, 
        max_paths: int = 3, 
        temperature: float = 1.5,
        live_path_metrics: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Calculates exact hardware traffic splitting ratios using the Softmax algorithm.
        
        Args:
            source: Ingress Router ID.
            destination: Egress Router ID.
            max_paths: Maximum number of parallel paths to bind.
            temperature: Softmax scaling. Higher values spread traffic more evenly; 
                         lower values strictly prefer the absolute lowest-cost path.
            live_path_metrics: Array of dicts containing active latency, cost, and utilization.
        """
        if not live_path_metrics:
            logger.warning("No live path metrics provided for WCMP split %s -> %s", source, destination)
            return {"ratios": [], "paths": [], "temperature": temperature}
            
        if len(live_path_metrics) > max_paths:
            # Sort by total internal cost and prune down to the k-shortest paths
            live_path_metrics = sorted(live_path_metrics, key=lambda p: p.get("cost", 9999))[:max_paths]

        # Extract base routing costs (fallback to 1.0)
        base_costs = np.array([float(p.get("cost", p.get("latency_ms", 1.0))) for p in live_path_metrics], dtype=np.float32)
        
        # Apply real-time hardware physics penalty
        congestion_multipliers = self._apply_live_congestion_penalty(live_path_metrics)
        effective_costs = base_costs * congestion_multipliers
        
        # --- VECTORIZED WCMP SOFTMAX COMPUTATION ---
        # 1. Scale by temperature and invert (Lower Cost = Higher Forwarding Weight)
        scaled_costs = -effective_costs / max(0.1, temperature)
        
        # 2. Shift for numerical stability to prevent float overflow (Explosive Exponents)
        scaled_costs -= np.max(scaled_costs)
        
        # 3. Calculate exponential weights
        exp_weights = np.exp(scaled_costs)
        
        # 4. Normalize to precise load-balancing percentages [0.0 -> 1.0]
        ratios = exp_weights / np.sum(exp_weights)
        
        # Format the output payload for BGP/SDN intent injection
        split_plan = []
        for i, metrics in enumerate(live_path_metrics):
            split_plan.append({
                "path_id": metrics.get("path_id", f"path_{i}"),
                "effective_cost": float(effective_costs[i]),
                "traffic_share_pct": round(float(ratios[i]) * 100.0, 2),
                "is_congested": bool(congestion_multipliers[i] > 1.0)
            })

        return {
            "source": source,
            "destination": destination,
            "temperature": temperature,
            "wcmp_ratios": split_plan
        }

    def build_qos_traffic_policy(
        self, source: str, destination: str, qos_class: QoSTrafficClass
    ) -> Dict[str, Any]:
        """
        Compiles an Intent-Based multi-criteria objective policy based on the requested SLA.
        Outputs the Alpha, Beta, and Gamma weights required by the PCE Dijkstra engine.
        """
        # Default Weights: [Cost, Latency, Congestion, Power]
        weights = {"alpha_cost": 0.33, "beta_latency": 0.33, "gamma_congestion": 0.34, "delta_power": 0.0}
        
        if qos_class == QoSTrafficClass.VIP_LOW_LATENCY:
            # HFT/Voice: Heavily bias Latency and Congestion; completely ignore financial OSPF cost
            weights = {"alpha_cost": 0.0, "beta_latency": 0.70, "gamma_congestion": 0.30, "delta_power": 0.0}
            
        elif qos_class == QoSTrafficClass.BULK_TRANSFER:
            # Backups: Heavily bias cheap financial routes; latency doesn't matter
            weights = {"alpha_cost": 0.60, "beta_latency": 0.0, "gamma_congestion": 0.40, "delta_power": 0.0}
            
        elif qos_class == QoSTrafficClass.GREEN_ENERGY:
            # Eco-Routing: Heavily penalize high optical temperatures and carbon-heavy routes
            weights = {"alpha_cost": 0.20, "beta_latency": 0.20, "gamma_congestion": 0.20, "delta_power": 0.40}

        # If a Path Computation Element is attached, we can request the route immediately
        computed_path = None
        if self.pce:
            try:
                computed_path = self.pce.compute_optimal_path(
                    source=source, 
                    destination=destination, 
                    objective="pareto_weight" # Signals the PCE to use the dynamic weight blending
                )
            except NetworkPartitionError:
                computed_path = {"error": "No physical path available"}

        return {
            "intent": "qos_policy_generation",
            "traffic_class": qos_class.value,
            "mathematical_weights": weights,
            "computed_pce_path": computed_path
        }
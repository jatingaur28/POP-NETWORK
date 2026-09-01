"""
Production Gymnasium Reinforcement Learning and Inference Environment.
Receives ONLY live hardware telemetry from Redis IPC and SQL Databases.
Maintains deterministic tensor mappings, real-time feature derivation,
and sub-millisecond state translation for PyTorch GNNs and RL agents.
"""

import os
import sys
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

# Ensure root path resolution for sdn_kdn module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

try:
    from sdn_kdn.core.exceptions import DataShapeError, KDNBaseException
except ImportError:
    class KDNBaseException(Exception): pass
    class DataShapeError(KDNBaseException): pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [AI-ROUTING-ENV] - %(message)s")
logger = logging.getLogger("AIRoutingEnv")


class WelfordStreamNormalizer:
    """
    Online Welford algorithm for running mean and variance computation.
    Enables statistical normalization over live streaming router telemetry
    without requiring batch storage or offline historical datasets.
    """
    def __init__(self, epsilon: float = 1e-6):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.epsilon = epsilon

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count > 1 else 1.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance)) + self.epsilon

    def normalize(self, x: float) -> float:
        return float((x - self.mean) / self.std)


class OfflineInferenceEnv(gym.Env):
    """
    High-Performance Reinforcement Learning Environment for Carrier Networks.
    Ingests live router telemetry from database queries and Redis pub/sub channels.
    """
    metadata = {"render_modes": []}
    NUM_LINK_FEATURES = 8

    def __init__(self, max_links: int = 250, sla_target_latency_ms: float = 150.0):
        super(OfflineInferenceEnv, self).__init__()
        self.max_links = max_links
        self.sla_target_latency_ms = sla_target_latency_ms
        self.state_lock = threading.RLock()

        # Action Space: Multi-link discrete OSPF metric adjustment actions
        # Actions: 0=Maintain, 1=Decrease Cost (Prefer), 2=Increase Cost (Deprioritize), 3=Emergency Drain
        self.action_space = spaces.MultiDiscrete([4] * self.max_links)

        # Observation Space: [Util, Norm Latency, Cost, Optical Temp, Error Rate, Velocity, Jitter, SLA Risk]
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.max_links, self.NUM_LINK_FEATURES),
            dtype=np.float32
        )

        # Deterministic Index Mapping
        self.link_to_idx: Dict[str, int] = {}
        self.idx_to_link: Dict[int, str] = {}
        self.node_links_map: Dict[str, List[str]] = {}
        self.node_to_idx: Dict[str, int] = {}

        # Pre-allocated zero-copy memory buffers
        self.state_buffer = np.zeros((self.max_links, self.NUM_LINK_FEATURES), dtype=np.float32)
        self.previous_raw_bw: Dict[str, float] = {}
        self.previous_raw_lat: Dict[str, float] = {}
        self.last_update_ts: Dict[str, float] = {}

        # Statistical streaming trackers
        self.util_normalizer = WelfordStreamNormalizer()
        self.latency_normalizer = WelfordStreamNormalizer()

    def update_topology_structure(self, topology_data: Dict[str, Any]) -> None:
        """
        Dynamically maps physical link identifiers and router nodes from the database
        into fixed row indices in the neural network memory buffer.
        """
        with self.state_lock:
            self.link_to_idx.clear()
            self.idx_to_link.clear()
            self.node_links_map.clear()
            self.node_to_idx.clear()

            nodes = topology_data.get("nodes", {})
            for n_idx, node_id in enumerate(sorted(nodes.keys())):
                self.node_to_idx[str(node_id)] = n_idx

            links = topology_data.get("links", {})
            for idx, lid in enumerate(sorted(links.keys())):
                if idx >= self.max_links:
                    logger.warning("Topology links (%d) exceeded max_links capacity (%d).", len(links), self.max_links)
                    break

                str_lid = str(lid)
                self.link_to_idx[str_lid] = idx
                self.idx_to_link[idx] = str_lid

                src = links[lid].get("source") or links[lid].get("src")
                dst = links[lid].get("target") or links[lid].get("dst")
                if src:
                    self.node_links_map.setdefault(str(src), []).append(str_lid)
                if dst:
                    self.node_links_map.setdefault(str(dst), []).append(str_lid)

            logger.info("Mapped %d physical links and %d router nodes into state buffers.", len(self.link_to_idx), len(self.node_to_idx))

    def inject_real_telemetry(self, telemetry_payload: List[Dict[str, Any]]) -> np.ndarray:
        """
        Ingests real-time telemetry metrics directly into the pre-allocated tensor buffer.
        Applies mathematical clipping, first-order derivatives, and anomaly risk estimation.
        """
        if not isinstance(telemetry_payload, list):
            raise DataShapeError("Telemetry payload must be a list of link metric dictionaries.")

        current_time = time.monotonic()

        with self.state_lock:
            # Temporal decay factor (preserves momentum while handling missed polling frames)
            self.state_buffer *= 0.98

            for t in telemetry_payload:
                lid = str(t.get("link", ""))
                if not lid or lid not in self.link_to_idx:
                    continue

                idx = self.link_to_idx[lid]

                # 1. Bandwidth & Capacity Derivations
                raw_bw = float(t.get("bandwidth_gbps", 0.0) or 0.0)
                max_cap = float(t.get("max_capacity", 10.0) or 10.0)
                max_cap_safe = max(0.001, max_cap)
                util = np.clip(raw_bw / max_cap_safe, 0.0, 1.0)
                self.util_normalizer.update(util)

                # 2. Latency & Jitter Derivations
                raw_lat = float(t.get("latency_ms", 10.0) or 10.0)
                norm_lat = np.clip(raw_lat / max(1.0, self.sla_target_latency_ms), 0.0, 5.0)
                self.latency_normalizer.update(raw_lat)

                prev_lat = self.previous_raw_lat.get(lid, raw_lat)
                jitter = abs(raw_lat - prev_lat) / max(1.0, raw_lat)

                # 3. Traffic Velocity (First Derivative: dU / dt)
                prev_bw = self.previous_raw_bw.get(lid, raw_bw)
                prev_ts = self.last_update_ts.get(lid, current_time - 1.0)
                dt = max(0.001, current_time - prev_ts)
                velocity = np.clip((raw_bw - prev_bw) / (max_cap_safe * dt), -1.0, 1.0)

                # 4. Routing Metrics & Physical Hardware Indicators
                cost = float(t.get("ospf_cost", 10.0) or 10.0) / 65535.0
                optical_temp = float(t.get("optical_temp_c", 35.0) or 35.0) / 100.0
                err_rate = float(t.get("interface_errors", 0.0) or 0.0)

                # 5. Composite SLA Breach Risk Index (0.0=Optimal, 1.0=Breached/Critical)
                sla_risk = 0.4 * util + 0.3 * norm_lat + 0.2 * jitter + 0.1 * optical_temp
                sla_risk = float(np.clip(sla_risk, 0.0, 1.0))

                # Update state vector in-place
                self.state_buffer[idx] = [
                    float(util),
                    float(norm_lat),
                    float(cost),
                    float(optical_temp),
                    float(err_rate),
                    float(velocity),
                    float(jitter),
                    float(sla_risk)
                ]

                # Update historical trackers
                self.previous_raw_bw[lid] = raw_bw
                self.previous_raw_lat[lid] = raw_lat
                self.last_update_ts[lid] = current_time

            return self.state_buffer.copy()

    def get_pytorch_graph(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Exports the current live hardware state as PyTorch Geometric-compatible tensors:
        - x: Node feature tensor (Aggregated ingress/egress load)
        - edge_index: Graph connectivity tensor (2, num_edges)
        - edge_attr: 8-dimensional link telemetry feature tensor
        """
        with self.state_lock:
            num_nodes = len(self.node_to_idx)
            num_links = len(self.link_to_idx)

            if num_nodes == 0 or num_links == 0:
                return torch.empty((0, 4)), torch.empty((2, 0), dtype=torch.long), torch.empty((0, self.NUM_LINK_FEATURES))

            # 1. Edge Index & Edge Attributes
            edge_indices: List[List[int]] = [[], []]
            edge_features: List[np.ndarray] = []

            for lid, idx in self.link_to_idx.items():
                parts = lid.split("-")
                if len(parts) >= 2:
                    src_str, dst_str = parts[0], parts[1]
                    if src_str in self.node_to_idx and dst_str in self.node_to_idx:
                        src_idx = self.node_to_idx[src_str]
                        dst_idx = self.node_to_idx[dst_str]

                        # Bidirectional connectivity
                        edge_indices[0].extend([src_idx, dst_idx])
                        edge_indices[1].extend([dst_idx, src_idx])

                        edge_features.append(self.state_buffer[idx])
                        edge_features.append(self.state_buffer[idx])

            edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long)
            edge_attr_tensor = torch.tensor(np.array(edge_features), dtype=torch.float32)

            # 2. Aggregated Node Features [In-Load, Out-Load, Node Link Count, Risk]
            node_features = np.zeros((num_nodes, 4), dtype=np.float32)
            for node_str, n_idx in self.node_to_idx.items():
                connected_links = self.node_links_map.get(node_str, [])
                if connected_links:
                    link_indices = [self.link_to_idx[l] for l in connected_links if l in self.link_to_idx]
                    if link_indices:
                        sub_matrix = self.state_buffer[link_indices]
                        avg_util = np.mean(sub_matrix[:, 0])
                        avg_lat = np.mean(sub_matrix[:, 1])
                        avg_risk = np.mean(sub_matrix[:, 7])
                        node_features[n_idx] = [avg_util, avg_lat, len(link_indices) / 10.0, avg_risk]

            node_x_tensor = torch.tensor(node_features, dtype=torch.float32)
            return node_x_tensor, edge_index_tensor, edge_attr_tensor

    def compute_objective_reward(self, dynamic_weights: Optional[Tuple[float, float, float]] = None) -> float:
        """
        Computes the global Multi-Objective SLA compliance reward across all live links.
        Penalizes congestion, latency violations, and optical thermal risks.
        """
        w_cong, w_lat, w_sla = dynamic_weights or (0.5, 0.3, 0.2)
        with self.state_lock:
            active_count = len(self.link_to_idx)
            if active_count == 0:
                return 0.0

            active_state = self.state_buffer[:active_count]
            util_penalties = np.sum(np.maximum(0.0, active_state[:, 0] - 0.85) ** 2)
            lat_penalties = np.sum(np.maximum(0.0, active_state[:, 1] - 1.0))
            sla_risks = np.sum(active_state[:, 7])

            total_penalty = w_cong * util_penalties + w_lat * lat_penalties + w_sla * sla_risks
            return float(-total_penalty / active_count)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Gymnasium step interface. In our asynchronous architecture, execution is
        dispatched over Redis while step returns the latest zero-copy state matrix.
        """
        with self.state_lock:
            reward = self.compute_objective_reward()
            obs = self.state_buffer.copy()
            info = {
                "active_links": len(self.link_to_idx),
                "mean_utilization": float(np.mean(obs[:len(self.link_to_idx), 0])) if self.link_to_idx else 0.0,
                "timestamp": time.time()
            }
            return obs, reward, False, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        with self.state_lock:
            self.state_buffer.fill(0.0)
            return self.state_buffer.copy(), {"status": "state_buffer_reset"}
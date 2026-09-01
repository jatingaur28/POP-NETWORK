"""
Tier-1 AIOps Orchestrator (The Brain).
Ingests live DB/Hardware telemetry exclusively. 
Executes Spatio-Temporal VAE, Topological RCA, BGP FlowSpec Mitigation, 
and generates Infrastructure-as-Code (IaC) Terraform intents.
"""

import sys
import os
import json
import time
import logging
from typing import Dict, Any, List, Set

import torch
import redis

# Ensure root path resolution for sdn_kdn module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from sdn_kdn.core.config import settings
from sdn_kdn.ai_engine.routing_env import OfflineInferenceEnv
from sdn_kdn.ai_engine.anomaly_detector import get_anomaly_model
from sdn_kdn.infrastructure.traffic_eng import TrafficEngineeringManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [AI-ORCHESTRATOR] - %(message)s")
logger = logging.getLogger("AIOpsPipeline")


class AutonomousAIOpsPipeline:
    """
    Enterprise AI Decision Loop.
    Strictly decoupled from random simulators. Listens to live Redis IPC hardware telemetry.
    """
    def __init__(self, vae_weights_path: str = "models/prod_vae.pth"):
        # 1. Hardware Acceleration Context
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("🚀 AI Operations Core Active on %s", self.device.type.upper())

        # 2. High-Speed IPC Message Broker (Redis)
        self.redis = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)
        self.pubsub = self.redis.pubsub()

        # 3. Core Network Intelligence Engines
        self.env = OfflineInferenceEnv(max_links=250)
        self.vae = get_anomaly_model(self.device, vae_weights_path, input_dim=8)
        self.te_manager = TrafficEngineeringManager()
        
        self.is_running = False

    def _execute_causal_rca(self, anomalous_links: List[str]) -> List[str]:
        """
        Topological Root Cause Analysis (RCA).
        Traces graph dependencies backwards to isolate the source router of a cascading failure.
        """
        root_causes: Set[str] = set()
        for link in anomalous_links:
            # Extract the source node from a standard 'SRC-DST' link string
            src_node = link.split("-")[0] if "-" in link else link
            
            # Find all upstream links feeding traffic into this node
            upstream_links = self.env.node_links_map.get(src_node, [])
            
            # If an upstream link is ALSO failing, this link is a victim, not the source
            causal_upstream = [ul for ul in upstream_links if ul in anomalous_links and ul != link]
            
            if not causal_upstream:
                root_causes.add(link)
                
        return list(root_causes)

    def _generate_iac_export(self, action_type: str, target: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infrastructure-as-Code (IaC) Payload Generator.
        Instead of blind SSH pushes, emits structured JSON that can be picked up by 
        Ansible or Terraform pipelines for safe, human-approved network orchestration.
        """
        return {
            "orchestration_type": "iac_export",
            "provider": "terraform",
            "resource": "network_routing_policy",
            "action": action_type,
            "target": target,
            "configuration": payload,
            "generated_by": "AIOps_Pipeline",
            "timestamp": time.time()
        }

    def _process_batched_telemetry(self, raw_data: str):
        """The Master AI Inference Loop. Triggered instantly upon receiving live hardware gNMI data."""
        if not self.env.link_to_idx:
            return  # Wait for SQL Database to sync the physical topology

        try:
            telemetry = json.loads(raw_data)
            
            # 1. Ingest physical metrics into zero-copy PyTorch tensors
            np_state = self.env.inject_real_telemetry(telemetry)
            state_tensor = torch.tensor(np_state, dtype=torch.float32, device=self.device)

            # 2. VAE Anomaly Detection (Dynamic thresholds based on Exponential Moving Average)
            # We use a strict 3.5 standard deviation sensitivity to prevent false positives
            anomalies, losses = self.vae.detect(state_tensor, sensitivity=3.5)
            
            # Extract actionable indices safely
            anomalous_indices = torch.nonzero(anomalies).squeeze(-1).tolist()
            if type(anomalous_indices) is int: 
                anomalous_indices = [anomalous_indices]

            if anomalous_indices:
                alert_links = [self.env.idx_to_link[idx] for idx in anomalous_indices]
                
                # 3. Causal Graph Root Cause Analysis
                root_causes = self._execute_causal_rca(alert_links)

                for r_link in root_causes:
                    idx = self.env.link_to_idx[r_link]
                    loss_val = losses[idx].item()
                    logger.critical("🛑 ROOT CAUSE ISOLATED -> %s (Anomaly Loss: %.4f)", r_link, loss_val)

                    # 4. Intent Generation: BGP FlowSpec Auto-Mitigation (Security)
                    # Protects the backbone by dropping the hostile flow at the edge hardware
                    bgp_intent = {
                        "action": "bgp_flowspec_mitigation",
                        "target_link": r_link,
                        "protocol": "tcp",
                        "action_type": "rate-limit",
                        "rate_bytes": 0,  # 0 bytes = drop malicious flow
                        "reason": f"VAE Dynamic Threshold Exceeded (Loss: {loss_val:.2f})",
                        "timestamp": time.time()
                    }
                    self.redis.publish("kdn_routing_intents", json.dumps(bgp_intent))
                    
                    # 5. Intent Generation: WCMP Traffic Steering (Resilience)
                    # Shifts legitimate user traffic away from the compromised/degraded link
                    src_node = r_link.split("-")[0] if "-" in r_link else r_link
                    dst_node = r_link.split("-")[1] if "-" in r_link else None
                    
                    if dst_node:
                        try:
                            # AI dynamically requests the Top-2 alternate paths via the TE Manager
                            failover_plan = self.te_manager.compute_wcmp_split(
                                source=src_node, 
                                destination=dst_node, 
                                max_paths=2, 
                                temperature=1.5
                            )
                            
                            # Export as safe Infrastructure-as-Code (IaC) for NetOps review
                            iac_payload = self._generate_iac_export(
                                action_type="traffic_shift",
                                target=r_link,
                                payload={
                                    "ospf_cost": 65000,  # Max out OSPF cost to drain traffic
                                    "wcmp_failover_plan": failover_plan
                                }
                            )
                            
                            # Dispatch to the CI/CD execution bus
                            self.redis.publish("kdn_iac_exports", json.dumps(iac_payload))
                            
                        except Exception as e:
                            logger.error("Failed to compute WCMP failover for %s: %s", r_link, str(e))

        except Exception as e:
            logger.error("Pipeline crashed during tensor translation: %s", str(e), exc_info=True)

    def start(self):
        """Engages the real-time hardware ingestion event loop."""
        self.is_running = True
        
        # Subscribe purely to hardware-driven state and telemetry channels
        self.pubsub.subscribe(**{
            "kdn_network_state": lambda m: self.env.update_topology_structure(json.loads(m['data']).get("topology", {})),
            "kdn_live_telemetry": lambda m: self._process_batched_telemetry(m['data'])
        })
        logger.info("🎧 AIOps Pipeline Active. Listening securely to physical hardware streams...")
        
        while self.is_running:
            try:
                # Non-blocking IPC read (Allows thread to exit gracefully)
                self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            except redis.ConnectionError:
                logger.warning("Redis IPC bus lost. Waiting for auto-reconnect...")
                time.sleep(2.0)
            except Exception:
                time.sleep(0.5)

if __name__ == "__main__":
    pipeline = AutonomousAIOpsPipeline()
    pipeline.start()
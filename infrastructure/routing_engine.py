"""
Enterprise Hardware Routing Engine (The 'Hands').
Translates high-level AI Intents into physical, multi-vendor CLI configurations.
Features BGP FlowSpec mitigation, OSPF metric steering, and an IaC Shadow Mode.
"""

import json
import logging
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

from netmiko import ConnectHandler
from netmiko.exceptions import NetmikoTimeoutException, NetmikoAuthenticationException

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from sdn_kdn.core.config import settings
from sdn_kdn.core.redis_utils import RedisConnectionManager
from sdn_kdn.core.topology_store import topology_store
from sdn_kdn.core.exceptions import ConfigurationPushError, HardwareConnectionError

logger = logging.getLogger("Hardware-Orchestrator")


class MultiVendorSyntaxBuilder:
    """Translates KDN intents into native CLI commands for Cisco, Arista, or Juniper."""
    
    @staticmethod
    def build_ospf_cost_commands(os_type: str, interface: str, cost: int) -> List[str]:
        if "cisco" in os_type or "arista" in os_type:
            return [
                f"interface {interface}",
                f"ip ospf cost {cost}",
                "exit"
            ]
        elif "juniper" in os_type:
            return [
                f"set protocols ospf area 0.0.0.0 interface {interface} metric {cost}"
            ]
        else:
            raise ValueError(f"Unsupported OS type for OSPF translation: {os_type}")

    @staticmethod
    def build_bgp_flowspec_commands(os_type: str, intent: Dict[str, Any]) -> List[str]:
        """Compiles a BGP FlowSpec policy to drop or rate-limit hostile flows."""
        action = intent.get("action_type", "rate-limit")
        rate = intent.get("rate_bytes", 0)
        
        if "cisco" in os_type or "arista" in os_type:
            # Standard Cisco/Arista Flowspec Class-Map syntax
            policy_name = f"AI_MITIGATION_{int(time.time())}"
            commands = [
                f"class-map type traffic match-all {policy_name}",
                f" match protocol {intent.get('protocol', 'tcp')}"
            ]
            if intent.get("source_prefix"):
                commands.append(f" match source-address ipv4 {intent['source_prefix']}")
            
            commands.extend([
                "exit",
                f"policy-map type pbr {policy_name}_POLICY",
                f" class type traffic {policy_name}"
            ])
            
            if action == "drop" or rate == 0:
                commands.append("  drop")
            else:
                commands.append(f"  police rate {rate} bytes")
                
            commands.extend([" exit", "exit"])
            return commands
            
        raise ValueError(f"BGP FlowSpec translation not yet supported for: {os_type}")


class HardwareRoutingEngine:
    """
    Subscribes to the Redis IPC bus and actuates physical network changes
    on the live router fleet using multi-threaded SSH connections.
    """
    def __init__(self):
        self.redis = RedisConnectionManager.get_sync_client()
        self.pubsub = self.redis.pubsub()
        self.syntax_builder = MultiVendorSyntaxBuilder()
        
        # Concurrency control: Prevent hammering the network management plane
        self.ssh_pool = ThreadPoolExecutor(max_workers=15)

    def _execute_ssh_push(self, router_ip: str, commands: List[str], intent_id: str):
        """Opens an SSH session, applies the configuration, and commits to memory."""
        device_profile = {
            'device_type': settings.netmiko_device_type,
            'host': router_ip,
            'username': settings.netmiko_user,
            'password': settings.netmiko_password,
            'session_timeout': 15,
            'auth_timeout': 10,
        }
        
        try:
            logger.info("🔌 Opening Netmiko SSH session to %s...", router_ip)
            with ConnectHandler(**device_profile) as ssh:
                # Enter config mode and push the multi-line payload
                output = ssh.send_config_set(commands)
                
                # Check for CLI rejection markers
                if "Invalid input" in output or "Incomplete command" in output:
                    logger.error("🚨 Router %s rejected syntax: %s", router_ip, output)
                    raise ConfigurationPushError(f"Syntax rejection on {router_ip}")
                
                # Save the running configuration to startup-config
                ssh.save_config()
                logger.info("✅ SUCCESS: Committed Intent [%s] to %s", intent_id, router_ip)

        except (NetmikoTimeoutException, NetmikoAuthenticationException) as e:
            logger.error("🚨 SSH Connection to %s failed: %s", router_ip, str(e))
            raise HardwareConnectionError(f"Cannot reach {router_ip}: {str(e)}")
        except Exception as e:
            logger.error("🚨 Configuration push failed on %s: %s", router_ip, str(e))

    def _route_intent(self, intent: Dict[str, Any]):
        """Parses the topological intent and dispatches it to the correct target."""
        action = intent.get("action")
        target_link = intent.get("target_link")
        intent_id = intent.get("intent_id", f"REQ-{int(time.time())}")
        
        # 1. Look up physical router metrics from the live Topology Store
        links = topology_store.get_active_links()
        link_data = next((l for l in links if l["id"] == target_link), None)
        
        if not link_data:
            logger.warning("Intent dropped: Target link '%s' not found in active topology.", target_link)
            return

        router_ip = link_data.get("router_ip")
        if_name = link_data.get("physical_interface", f"GigabitEthernet0/0/{link_data.get('snmp_if_index', 0)}")
        os_type = settings.netmiko_device_type

        if not router_ip:
            logger.warning("Intent dropped: No management IP registered for link '%s'.", target_link)
            return

        # 2. Compile commands based on the AI's requested action
        commands = []
        try:
            if action in ("ai_traffic_shift", "human_override"):
                new_cost = intent.get("new_cost", 65000)
                commands = self.syntax_builder.build_ospf_cost_commands(os_type, if_name, new_cost)
                
            elif action == "bgp_flowspec_mitigation":
                commands = self.syntax_builder.build_bgp_flowspec_commands(os_type, intent)
                
            else:
                logger.warning("Unknown intent action received: %s", action)
                return
                
        except ValueError as ve:
            logger.error("Syntax compilation failed: %s", str(ve))
            return

        # 3. Governance Gate: Autonomous vs. Shadow Mode
        if not settings.autonomous_mode:
            logger.info("🛡️ [SHADOW MODE] IaC Export Generated for %s -> %s\n%s", router_ip, target_link, "\n".join(commands))
            return

        # 4. Dispatch to Background Thread Pool
        self.ssh_pool.submit(self._execute_ssh_push, router_ip, commands, intent_id)

    def _listen_for_intents(self, message: Dict[str, Any]):
        """Unpacks Redis Pub/Sub messages cleanly."""
        try:
            intent = json.loads(message["data"])
            self._route_intent(intent)
        except json.JSONDecodeError:
            logger.error("Received malformed JSON intent payload on IPC bus.")
        except Exception as e:
            logger.error("Failed to process incoming intent: %s", str(e), exc_info=True)

    def start(self):
        """Engages the Redis Pub/Sub subscriber thread."""
        self.pubsub.subscribe(**{"kdn_routing_intents": self._listen_for_intents})
        logger.info("🎯 Hardware Routing Engine Online. Subscribed to KDN Intent Bus.")
        
        while True:
            try:
                # Keep the thread alive, allowing the callback functions to handle events
                self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            except Exception:
                time.sleep(1)

if __name__ == "__main__":
    engine = HardwareRoutingEngine()
    engine.start()
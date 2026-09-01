"""
Enterprise-Grade Master Configuration for the KDN Architecture.
Strictly parses secrets from the `.env` file to prevent credential leaks.
Validates Live Neon Serverless PostgreSQL, gNMI Hardware, BGP-LS, and AI hyperparameters.
"""

import os
import sys
import logging
from functools import lru_cache
from typing import Optional, List

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("KDNConfig")


class KDNSettings(BaseSettings):
    # =========================================================================
    # 1. ENVIRONMENT & GOVERNANCE
    # =========================================================================
    environment: str = Field(default="production", description="Runtime environment scope (production/staging)")
    autonomous_mode: bool = Field(
        default=False, 
        description="If True, AI directly pushes IaC / BGP intents. If False, operates in Shadow Mode (read-only)."
    )
    chaos_monkey_enabled: bool = Field(
        default=False, 
        description="Enables background thread to inject random physical link failures for resilience testing."
    )

    # =========================================================================
    # 2. SECURITY, RBAC & ALERTING
    # =========================================================================
    # Multi-tier API keys for Role-Based Access Control (RBAC)
    admin_api_key: Optional[str] = Field(default=None, description="Key for destructive actions (Chaos, Intent Pushes)")
    operator_api_key: Optional[str] = Field(default=None, description="Key for manual TE overrides and metric scaling")
    viewer_api_key: Optional[str] = Field(default=None, description="Key for read-only Dash UI and Topology access")
    
    # External integrations
    webhook_alert_url: Optional[str] = Field(default=None, description="Slack/PagerDuty/Opsgenie sink for AI anomalies")

    # =========================================================================
    # 3. IPC MESSAGE BROKER (REDIS)
    # =========================================================================
    redis_host: str = Field(default="127.0.0.1", description="Redis IPC Host")
    redis_port: int = Field(default=6379, ge=1, le=65535, description="Redis IPC Port")
    redis_db: int = Field(default=0, description="Redis Logical Database")
    redis_password: Optional[str] = Field(default=None, description="Secure auth for IPC bus")

    # =========================================================================
    # 4. SERVERLESS SOURCE OF TRUTH (NEON POSTGRESQL)
    # =========================================================================
    db_host: str = Field(..., description="Neon database host (e.g., ep-pooler...)")
    db_port: int = Field(default=5432, ge=1, le=65535)
    db_user: str = Field(..., description="Neon role/owner")
    db_password: str = Field(..., description="Neon secure password")
    db_name: str = Field(..., description="Neon logical database name")
    
    # Advanced Security & Pooling for Serverless Databases
    db_sslmode: str = Field(default="require", description="Mandatory SSL for Neon AWS connections")
    db_channel_binding: str = Field(default="require", description="Protects against MITM attacks")
    db_pool_size: int = Field(default=25, description="Max persistent database connections for high-throughput AI")
    db_max_overflow: int = Field(default=15, description="Max temporary overflow connections")

    @computed_field
    @property
    def database_url(self) -> str:
        """
        Dynamically generates the ultra-secure SQLAlchemy URI.
        Forces the modern 'psycopg' driver for high-performance async/sync execution.
        """
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?sslmode={self.db_sslmode}&channel_binding={self.db_channel_binding}"
        )

    # =========================================================================
    # 5. SOUTHBOUND HARDWARE INTERFACES (gNMI, SNMP, NETMIKO, BGP-LS)
    # =========================================================================
    gnmi_port: int = Field(default=57400, description="Standard gNMI Telemetry Port")
    snmp_community: str = Field(..., description="SNMPv2c read-only community string")
    snmp_timeout_sec: float = Field(default=1.5, description="Timeout for physical polling")
    
    netmiko_device_type: str = Field(default="cisco_ios", description="Target hardware OS (cisco_ios, arista_eos)")
    netmiko_user: str = Field(..., description="Router SSH Username for Configuration Pushes")
    netmiko_password: str = Field(..., description="Router SSH Password")

    gobgp_host: str = Field(default="127.0.0.1", description="GoBGP daemon for BGP-LS ingestion")
    gobgp_port: int = Field(default=50051, description="GoBGP gRPC port")

    # =========================================================================
    # 6. EXTERNAL LIVE DATA FEEDS
    # =========================================================================
    enable_weather_risk: bool = Field(default=True, description="Polls Open-Meteo for physical fiber storm risks")
    public_probe_targets: str = Field(
        default="1.1.1.1,8.8.8.8,9.9.9.9", 
        description="Comma-separated public anycast IPs for live ICMP/TCP internet weather probing"
    )

    # =========================================================================
    # 7. AI, AIOPS & TRAFFIC ENGINEERING HYPERPARAMETERS
    # =========================================================================
    model_weights_path: str = Field(default="models/prod_vae.pth", description="Path to PyTorch tensor weights")
    
    vae_sensitivity: float = Field(
        default=3.5, 
        description="Standard deviations above the Exponential Moving Average required to trigger a DDoS/Anomaly alert."
    )
    wcmp_temperature: float = Field(
        default=1.5, 
        description="Softmax temperature for ECMP load balancing. (Lower = Stricter Shortest Path)."
    )
    sla_target_latency_ms: float = Field(
        default=150.0, 
        description="Global SLA threshold. AI heavily penalizes links breaching this value."
    )
    green_energy_preference: float = Field(
        default=0.2, 
        description="Weighting factor to steer BE (Best Effort) traffic toward routes with lower Carbon/Cost metrics."
    )

    # =========================================================================
    # 8. NORTHBOUND INTERFACES (API & NOC UI)
    # =========================================================================
    api_host: str = Field(default="0.0.0.0", description="FastAPI bind address")
    api_port: int = Field(default=5001, ge=1, le=65535)
    ui_port: int = Field(default=8050, ge=1, le=65535)


    # Pydantic v2 configuration ensures it reads strictly from the .env file in the root
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Gracefully ignores unused .env variables without crashing
    )

    @property
    def parsed_probe_targets(self) -> List[str]:
        """Helper to return probe IPs as a cleaned list."""
        return [ip.strip() for ip in self.public_probe_targets.split(",") if ip.strip()]


@lru_cache()
def get_settings() -> KDNSettings:
    """
    Cached singleton instantiator.
    Prevents the application from repeatedly reading the disk, executing in O(1) time.
    """
    try:
        return KDNSettings()
    except Exception as e:
        logger.critical("🚨 CRITICAL BOOT FAILURE: Missing live credentials in .env file!")
        logger.critical("Please ensure DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, SNMP_COMMUNITY, NETMIKO_USER, and NETMIKO_PASSWORD are set.")
        logger.critical("Error Details: %s", str(e))
        sys.exit(1)  # Abort controller boot immediately to prevent unpredictable cascading failures


# The global settings instance to be imported throughout the architecture
settings = get_settings()
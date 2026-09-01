"""
KDN Enterprise Exception Hierarchy (v6.0).
Strictly defines failure domains across Live Hardware (gNMI/SNMP), 
Serverless PostgreSQL Databases (Neon), Redis IPC, and AI AIOps layers.
"""

import time
from typing import Any, Dict, Optional

class ErrorSeverity:
    """Standardized logging and alerting severity levels."""
    CRITICAL = "CRITICAL"  # Requires immediate operator intervention (e.g., Database offline)
    ERROR = "ERROR"        # Fails a specific transaction or path computation
    WARNING = "WARNING"    # Handled autonomously, but recorded for audit


class KDNBaseException(Exception):
    """
    Base exception class for all custom Knowledge-Defined Network errors.
    Supports structured debugging context, severity indexing, and JSON serialization 
    for FastAPI Northbound responses and Redis IPC broadcasts.
    """
    error_code: str = "KDN-ERR-0000"
    severity: str = ErrorSeverity.ERROR
    http_status: int = 500

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.timestamp = time.time()

    def __str__(self) -> str:
        base_str = f"[{self.severity}] [{self.error_code}] {self.message}"
        if self.context:
            base_str += f" | Context: {self.context}"
        return base_str

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the exception for REST API responses and Dash UI Alerts."""
        return {
            "error_code": self.error_code,
            "severity": self.severity,
            "exception": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp,
        }


# ==============================================================================
# 1. HARDWARE & TELEMETRY EXCEPTIONS (Series 1000)
# ==============================================================================
class HardwareConnectionError(KDNBaseException):
    """Raised when Netmiko (SSH), gNMI, or SNMP cannot reach the physical router."""
    error_code = "KDN-HW-1001"
    severity = ErrorSeverity.ERROR
    http_status = 503

class TelemetryStreamBrokenError(KDNBaseException):
    """Raised when a live gNMI subscription stream unexpectedly disconnects."""
    error_code = "KDN-HW-1002"
    severity = ErrorSeverity.WARNING
    http_status = 503

class ConfigurationPushError(KDNBaseException):
    """Raised when a router accepts an SSH connection but rejects the OSPF/BGP command configuration."""
    error_code = "KDN-HW-1003"
    severity = ErrorSeverity.ERROR
    http_status = 422


# ==============================================================================
# 2. ROUTING & TOPOLOGY EXCEPTIONS (Series 2000)
# ==============================================================================
class NetworkPartitionError(KDNBaseException):
    """Raised by the Path Computation Element (PCE) when no viable physical graph pathway exists."""
    error_code = "KDN-RT-2001"
    severity = ErrorSeverity.CRITICAL
    http_status = 404

class TopologyLoadError(KDNBaseException):
    """Raised when the topology payload from the live database is corrupted or structurally invalid."""
    error_code = "KDN-RT-2002"
    severity = ErrorSeverity.ERROR
    http_status = 500

class CSPFConstraintError(KDNBaseException):
    """Raised when a Constrained Shortest Path First (CSPF) route fails due to insufficient residual bandwidth."""
    error_code = "KDN-RT-2003"
    severity = ErrorSeverity.WARNING
    http_status = 409


# ==============================================================================
# 3. AI & AIOPS INFERENCE EXCEPTIONS (Series 3000)
# ==============================================================================
class AIInferenceError(KDNBaseException):
    """Raised when a PyTorch GPU model fails during forward-pass execution (VAE/GAT)."""
    error_code = "KDN-AI-3001"
    severity = ErrorSeverity.CRITICAL
    http_status = 500

class DataShapeError(KDNBaseException):
    """Raised when incoming live telemetry features do not match expected neural network tensor dimensions."""
    error_code = "KDN-AI-3002"
    severity = ErrorSeverity.ERROR
    http_status = 400

class RCAIsolationError(KDNBaseException):
    """Raised when the Topological Root Cause Analysis cannot confidently isolate a singular root cause."""
    error_code = "KDN-AI-3003"
    severity = ErrorSeverity.WARNING
    http_status = 409


# ==============================================================================
# 4. DATABASE & IPC (INTER-PROCESS COMMUNICATION) EXCEPTIONS (Series 4000)
# ==============================================================================
class DatabaseConnectionError(KDNBaseException):
    """Raised when the primary SQL inventory database (Neon Serverless) is unreachable."""
    error_code = "KDN-DB-4001"
    severity = ErrorSeverity.CRITICAL
    http_status = 503

class NeonPoolExhaustedError(DatabaseConnectionError):
    """Raised when the async connection pool to the Neon database is saturated."""
    error_code = "KDN-DB-4002"
    severity = ErrorSeverity.ERROR
    http_status = 503

class IPCBrokerError(KDNBaseException):
    """Raised when a microservice loses TCP connectivity to the local Redis message bus."""
    error_code = "KDN-IPC-4001"
    severity = ErrorSeverity.CRITICAL
    http_status = 503
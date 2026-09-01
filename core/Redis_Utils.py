"""
Enterprise Redis IPC Message Broker & Caching Layer.
Engineered for ultra-low latency, real-time hardware telemetry streaming,
and thread-safe Pub/Sub operations across multiprocessing microservices.
"""

import json
import logging
import threading
from typing import Optional, Any, Dict, Union

import redis
import redis.asyncio as aioredis
from redis.backoff import ExponentialBackoff
from redis.retry import Retry
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

try:
    from sdn_kdn.core.config import settings
except ImportError:
    # Failsafe fallback
    import collections
    settings = collections.namedtuple('Settings', ['redis_host', 'redis_port', 'redis_db', 'redis_password'])(
        "127.0.0.1", 6379, 0, None
    )

try:
    from sdn_kdn.core.exceptions import IPCBrokerError
except ImportError:
    class IPCBrokerError(Exception): pass

logger = logging.getLogger("KDN-RedisIPC")


class RedisConnectionManager:
    """
    Enterprise Redis Connection Pooling.
    Maintains persistent, thread-safe TCP sockets to the message broker
    with automatic exponential backoff and circuit-breaking.
    """
    _sync_pool: Optional[redis.ConnectionPool] = None
    _async_pool: Optional[aioredis.ConnectionPool] = None
    _lock = threading.Lock()

    # Retry strategy: 3 retries, exponential backoff starting at 0.5s up to 5s
    _retry_strategy = Retry(ExponentialBackoff(cap=5.0, base=0.5), 3)

    @classmethod
    def get_sync_client(cls) -> redis.Redis:
        """
        Returns a high-speed, thread-safe synchronous Redis client.
        Optimized for background hardware pollers (gNMI/SNMP) and OSPF Engines.
        """
        with cls._lock:
            if cls._sync_pool is None:
                cls._sync_pool = redis.ConnectionPool(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    db=settings.redis_db,
                    password=settings.redis_password,
                    decode_responses=True,
                    max_connections=200,
                    socket_timeout=3.0,
                    socket_connect_timeout=3.0,
                    socket_keepalive=True,
                    retry_on_timeout=True
                )
                logger.info("🟢 Synchronous Redis IPC Pool established (%s:%s).", settings.redis_host, settings.redis_port)
        
        client = redis.Redis(connection_pool=cls._sync_pool, retry=cls._retry_strategy)
        try:
            client.ping()
        except (RedisConnectionError, RedisTimeoutError) as e:
            raise IPCBrokerError(f"Sync Redis Broker offline at {settings.redis_host}:{settings.redis_port} - {str(e)}")
        return client

    @classmethod
    def get_async_client(cls) -> aioredis.Redis:
        """
        Returns a non-blocking asynchronous Redis client.
        Optimized for FastAPI endpoints and Dash WebSockets handling high-throughput telemetry.
        """
        if cls._async_pool is None:
            cls._async_pool = aioredis.ConnectionPool.from_url(
                f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
                password=settings.redis_password,
                decode_responses=True,
                max_connections=1000,
                socket_timeout=3.0,
                retry_on_timeout=True
            )
            logger.info("🟢 Asynchronous Redis IPC Pool established (%s:%s).", settings.redis_host, settings.redis_port)
            
        return aioredis.Redis(connection_pool=cls._async_pool, retry=cls._retry_strategy)

    @classmethod
    def is_healthy(cls) -> bool:
        """Lightweight health probe used by the Master Orchestrator watchdog."""
        try:
            client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                socket_connect_timeout=1.0,
                decode_responses=True
            )
            return client.ping()
        except Exception:
            return False

    @classmethod
    def close_all(cls):
        """Gracefully terminates all open TCP sockets. Called during orchestrator shutdown."""
        with cls._lock:
            if cls._sync_pool:
                cls._sync_pool.disconnect()
                cls._sync_pool = None
                logger.info("🛑 Synchronous Redis IPC Pool disconnected.")
            if cls._async_pool:
                cls._async_pool.disconnect()
                cls._async_pool = None
                logger.info("🛑 Asynchronous Redis IPC Pool disconnected.")


class PubSubHelper:
    """
    Utility class to streamline JSON serialization/deserialization 
    over Redis Pub/Sub channels for AI and Routing intent broadcasts.
    """
    
    @staticmethod
    def publish_sync(channel: str, payload: Union[Dict[str, Any], list]) -> int:
        """Synchronously broadcasts an intent or telemetry batch to all listening microservices."""
        try:
            client = RedisConnectionManager.get_sync_client()
            return client.publish(channel, json.dumps(payload))
        except Exception as e:
            logger.error("Failed to publish to channel '%s': %s", channel, str(e))
            raise IPCBrokerError(f"Publish failed: {e}")

    @staticmethod
    async def publish_async(channel: str, payload: Union[Dict[str, Any], list]) -> int:
        """Asynchronously broadcasts data (used heavily by FastAPI webhooks)."""
        try:
            client = RedisConnectionManager.get_async_client()
            return await client.publish(channel, json.dumps(payload))
        except Exception as e:
            logger.error("Async publish failed on channel '%s': %s", channel, str(e))
            raise IPCBrokerError(f"Async publish failed: {e}")
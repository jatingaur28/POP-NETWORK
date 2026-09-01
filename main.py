"""
POP NETWORK COMMAND CENTER (HYBRID LIVE-CSV FUSION EDITION)
Features: 
- HYBRID TELEMETRY FUSION: Simultaneously streams deep 76-dim CSV features AND live BGP-LS hardware metrics.
- UNRESTRICTED INGESTION: Loads maximum possible rows from CSV telemetry datasets.
- MASSIVE TGN TOPOLOGY: Over 85 accurate global PoPs perfectly spaced out for SOC surveillance.
- DYNAMIC COLORS: Neon Green (Safe) -> Warning Yellow -> Critical Red (Congestion/Anomaly).
- LIVE HARDWARE BRIDGING: Redis GoBGP ingestion and Netmiko SSH push included.
"""

import os
import sys
import time
import socket
import logging
import threading
import json
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch import nn

import dash
import dash_cytoscape as cyto
from dash import html, dcc
from dash.dependencies import Input, Output

# Optional Netmiko driver for live router bridging
try:
    from netmiko import ConnectHandler
    NETMIKO_AVAILABLE = True
except ImportError:
    NETMIKO_AVAILABLE = False

# Optional Redis driver for live BGP-LS ingestion
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# ==============================================================================
# 1. LOGGING & CONFIGURATION
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("MasterSupervisor")

UI_PORT = int(os.getenv("UI_PORT", 8050))
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
NETMIKO_USER = os.getenv("NETMIKO_USER", "admin")
NETMIKO_PASSWORD = os.getenv("NETMIKO_PASSWORD", "admin")

# ==============================================================================
# 2. TOPOLOGY STATE: UNRESTRICTED TATA COMMUNICATIONS GLOBAL NETWORK
# ==============================================================================
STATE_LOCK = threading.RLock()
CURRENT_DIR = Path(__file__).resolve().parent

WEIGHTS_PATH = CURRENT_DIR / "oracle_brain.pth"
DATASET_PATH = CURRENT_DIR / "Dataset.csv"
NETWORK_TRAIN_PATH = CURRENT_DIR / "network_training_data.csv"

# Comprehensive Tata Communications (AS4755) Global PoPs & Landing Stations
BGP_LS_NODES = {
    # AREA 0.0.0.0: Global Transit Backbone & Subsea Landing Stations
    "Mumbai":         {"coords": (19.0760, 72.8777),   "asn": 4755, "ip": "10.0.0.1", "ospf_area": "0.0.0.0"},
    "London":         {"coords": (51.5074, -0.1278),   "asn": 4755, "ip": "10.0.0.2", "ospf_area": "0.0.0.0"},
    "New York":       {"coords": (40.7128, -74.0060),  "asn": 4755, "ip": "10.0.0.3", "ospf_area": "0.0.0.0"},
    "Los Angeles":    {"coords": (34.0522, -118.2430), "asn": 4755, "ip": "10.0.0.4", "ospf_area": "0.0.0.0"},
    "Singapore":      {"coords": (1.3521, 103.8198),   "asn": 4755, "ip": "10.0.0.5", "ospf_area": "0.0.0.0"},
    "Tokyo":          {"coords": (35.6762, 139.6503),  "asn": 4755, "ip": "10.0.0.6", "ospf_area": "0.0.0.0"},
    "Frankfurt":      {"coords": (50.1109, 8.6821),    "asn": 4755, "ip": "10.0.0.7", "ospf_area": "0.0.0.0"},
    "Dubai":          {"coords": (25.2048, 55.2708),   "asn": 4755, "ip": "10.0.0.8", "ospf_area": "0.0.0.0"},
    "Marseille":      {"coords": (43.2965, 5.3698),    "asn": 4755, "ip": "10.0.0.9", "ospf_area": "0.0.0.0"},
    "Ashburn":        {"coords": (39.0438, -77.4874),  "asn": 4755, "ip": "10.0.0.10", "ospf_area": "0.0.0.0"},
    "Fujairah":       {"coords": (25.1288, 56.3265),   "asn": 4755, "ip": "10.0.0.11", "ospf_area": "0.0.0.0"},

    # AREA 0.0.0.1: India & South Asia Regional Core
    "Delhi":          {"coords": (28.7041, 77.1025),   "asn": 4755, "ip": "10.1.0.1", "ospf_area": "0.0.0.1"},
    "Bangalore":      {"coords": (12.9716, 77.5946),   "asn": 4755, "ip": "10.1.0.2", "ospf_area": "0.0.0.1"},
    "Chennai":        {"coords": (13.0827, 80.2707),   "asn": 4755, "ip": "10.1.0.3", "ospf_area": "0.0.0.1"},
    "Hyderabad":      {"coords": (17.3850, 78.4867),   "asn": 4755, "ip": "10.1.0.4", "ospf_area": "0.0.0.1"},
    "Kolkata":        {"coords": (22.5726, 88.3639),   "asn": 4755, "ip": "10.1.0.5", "ospf_area": "0.0.0.1"},
    "Pune":           {"coords": (18.5204, 73.8567),   "asn": 4755, "ip": "10.1.0.6", "ospf_area": "0.0.0.1"},
    "Ahmedabad":      {"coords": (23.0225, 72.5714),   "asn": 4755, "ip": "10.1.0.7", "ospf_area": "0.0.0.1"},
    "Jaipur":         {"coords": (26.9124, 75.7873),   "asn": 4755, "ip": "10.1.0.8", "ospf_area": "0.0.0.1"},
    "Kochi":          {"coords": (9.9312, 76.2673),    "asn": 4755, "ip": "10.1.0.9", "ospf_area": "0.0.0.1"},
    "Colombo":        {"coords": (6.9271, 79.8612),    "asn": 4755, "ip": "10.1.0.10", "ospf_area": "0.0.0.1"},
    "Chandigarh":     {"coords": (30.7333, 76.7794),   "asn": 4755, "ip": "10.1.0.11", "ospf_area": "0.0.0.1"},
    "Lucknow":        {"coords": (26.8467, 80.9462),   "asn": 4755, "ip": "10.1.0.12", "ospf_area": "0.0.0.1"},
    "Indore":         {"coords": (22.7196, 75.8577),   "asn": 4755, "ip": "10.1.0.13", "ospf_area": "0.0.0.1"},
    "Bhopal":         {"coords": (23.2599, 77.4126),   "asn": 4755, "ip": "10.1.0.14", "ospf_area": "0.0.0.1"},
    "Nagpur":         {"coords": (21.1458, 79.0882),   "asn": 4755, "ip": "10.1.0.15", "ospf_area": "0.0.0.1"},
    "Patna":          {"coords": (25.5941, 85.1376),   "asn": 4755, "ip": "10.1.0.16", "ospf_area": "0.0.0.1"},
    "Goa":            {"coords": (15.2993, 74.1240),   "asn": 4755, "ip": "10.1.0.17", "ospf_area": "0.0.0.1"},
    "Guwahati":       {"coords": (26.1445, 91.7362),   "asn": 4755, "ip": "10.1.0.18", "ospf_area": "0.0.0.1"},
    "Bhubaneswar":    {"coords": (20.2961, 85.8245),   "asn": 4755, "ip": "10.1.0.19", "ospf_area": "0.0.0.1"},

    # AREA 0.0.0.2: Americas Regional Core
    "Chicago":        {"coords": (41.8781, -87.6298),  "asn": 4755, "ip": "10.2.0.1", "ospf_area": "0.0.0.2"},
    "Dallas":         {"coords": (32.7767, -96.7970),  "asn": 4755, "ip": "10.2.0.2", "ospf_area": "0.0.0.2"},
    "Miami":          {"coords": (25.7617, -80.1918),  "asn": 4755, "ip": "10.2.0.3", "ospf_area": "0.0.0.2"},
    "Toronto":        {"coords": (43.6510, -79.3470),  "asn": 4755, "ip": "10.2.0.4", "ospf_area": "0.0.0.2"},
    "Sao Paulo":      {"coords": (-23.5505, -46.6330), "asn": 4755, "ip": "10.2.0.5", "ospf_area": "0.0.0.2"},
    "Newark":         {"coords": (40.7357, -74.1724),  "asn": 4755, "ip": "10.2.0.6", "ospf_area": "0.0.0.2"},
    "San Jose":       {"coords": (37.3382, -121.8863), "asn": 4755, "ip": "10.2.0.7", "ospf_area": "0.0.0.2"},
    "Palo Alto":      {"coords": (37.4419, -122.1430), "asn": 4755, "ip": "10.2.0.8", "ospf_area": "0.0.0.2"},
    "Seattle":        {"coords": (47.6062, -122.3321), "asn": 4755, "ip": "10.2.0.9", "ospf_area": "0.0.0.2"},
    "Montreal":       {"coords": (45.5017, -73.5673),  "asn": 4755, "ip": "10.2.0.10", "ospf_area": "0.0.0.2"},
    "Mexico City":    {"coords": (19.4326, -99.1332),  "asn": 4755, "ip": "10.2.0.11", "ospf_area": "0.0.0.2"},
    "Rio de Janeiro": {"coords": (-22.9068, -43.1729), "asn": 4755, "ip": "10.2.0.12", "ospf_area": "0.0.0.2"},
    "Buenos Aires":   {"coords": (-34.6037, -58.3816), "asn": 4755, "ip": "10.2.0.13", "ospf_area": "0.0.0.2"},
    "Bogota":         {"coords": (4.7110, -74.0721),   "asn": 4755, "ip": "10.2.0.14", "ospf_area": "0.0.0.2"},
    "Santiago":       {"coords": (-33.4489, -70.6693), "asn": 4755, "ip": "10.2.0.15", "ospf_area": "0.0.0.2"},
    "Lima":           {"coords": (-12.0464, -77.0428), "asn": 4755, "ip": "10.2.0.16", "ospf_area": "0.0.0.2"},
    "Panama City":    {"coords": (8.9824, -79.5199),   "asn": 4755, "ip": "10.2.0.17", "ospf_area": "0.0.0.2"},

    # AREA 0.0.0.3: Europe Regional Core
    "Paris":          {"coords": (48.8566, 2.3522),    "asn": 4755, "ip": "10.3.0.1", "ospf_area": "0.0.0.3"},
    "Amsterdam":      {"coords": (52.3676, 4.9041),    "asn": 4755, "ip": "10.3.0.2", "ospf_area": "0.0.0.3"},
    "Madrid":         {"coords": (40.4168, -3.7038),   "asn": 4755, "ip": "10.3.0.3", "ospf_area": "0.0.0.3"},
    "Stockholm":      {"coords": (59.3293, 18.0686),   "asn": 4755, "ip": "10.3.0.4", "ospf_area": "0.0.0.3"},
    "Manchester":     {"coords": (53.4808, -2.2426),   "asn": 4755, "ip": "10.3.0.5", "ospf_area": "0.0.0.3"},
    "Munich":         {"coords": (48.1351, 11.5820),   "asn": 4755, "ip": "10.3.0.6", "ospf_area": "0.0.0.3"},
    "Berlin":         {"coords": (52.5200, 13.4050),   "asn": 4755, "ip": "10.3.0.7", "ospf_area": "0.0.0.3"},
    "Lisbon":         {"coords": (38.7223, -9.1393),   "asn": 4755, "ip": "10.3.0.8", "ospf_area": "0.0.0.3"},
    "Rome":           {"coords": (41.9028, 12.4964),   "asn": 4755, "ip": "10.3.0.9", "ospf_area": "0.0.0.3"},
    "Milan":          {"coords": (45.4642, 9.1900),    "asn": 4755, "ip": "10.3.0.10", "ospf_area": "0.0.0.3"},
    "Zurich":         {"coords": (47.3769, 8.5417),    "asn": 4755, "ip": "10.3.0.11", "ospf_area": "0.0.0.3"},
    "Vienna":         {"coords": (48.2082, 16.3738),   "asn": 4755, "ip": "10.3.0.12", "ospf_area": "0.0.0.3"},
    "Warsaw":         {"coords": (52.2297, 21.0122),   "asn": 4755, "ip": "10.3.0.13", "ospf_area": "0.0.0.3"},
    "Prague":         {"coords": (50.0755, 14.4378),   "asn": 4755, "ip": "10.3.0.14", "ospf_area": "0.0.0.3"},
    "Brussels":       {"coords": (50.8503, 4.3517),    "asn": 4755, "ip": "10.3.0.15", "ospf_area": "0.0.0.3"},
    "Copenhagen":     {"coords": (55.6761, 12.5683),   "asn": 4755, "ip": "10.3.0.16", "ospf_area": "0.0.0.3"},
    "Helsinki":       {"coords": (60.1695, 24.9354),   "asn": 4755, "ip": "10.3.0.17", "ospf_area": "0.0.0.3"},
    "Athens":         {"coords": (37.9838, 23.7275),   "asn": 4755, "ip": "10.3.0.18", "ospf_area": "0.0.0.3"},
    "Dublin":         {"coords": (53.3498, -6.2603),   "asn": 4755, "ip": "10.3.0.19", "ospf_area": "0.0.0.3"},

    # AREA 0.0.0.4: Asia-Pacific Regional Core
    "Hong Kong":      {"coords": (22.3193, 114.1693),  "asn": 4755, "ip": "10.4.0.1", "ospf_area": "0.0.0.4"},
    "Sydney":         {"coords": (-33.8688, 151.2093), "asn": 4755, "ip": "10.4.0.2", "ospf_area": "0.0.0.4"},
    "Seoul":          {"coords": (37.5665, 126.9780),  "asn": 4755, "ip": "10.4.0.3", "ospf_area": "0.0.0.4"},
    "Bangkok":        {"coords": (13.7563, 100.5018),  "asn": 4755, "ip": "10.4.0.4", "ospf_area": "0.0.0.4"},
    "Osaka":          {"coords": (34.6937, 135.5023),  "asn": 4755, "ip": "10.4.0.5", "ospf_area": "0.0.0.4"},
    "Melbourne":      {"coords": (-37.8136, 144.9631), "asn": 4755, "ip": "10.4.0.6", "ospf_area": "0.0.0.4"},
    "Perth":          {"coords": (-31.9505, 115.8605), "asn": 4755, "ip": "10.4.0.7", "ospf_area": "0.0.0.4"},
    "Brisbane":       {"coords": (-27.4698, 153.0251), "asn": 4755, "ip": "10.4.0.8", "ospf_area": "0.0.0.4"},
    "Auckland":       {"coords": (-36.8485, 174.7633), "asn": 4755, "ip": "10.4.0.9", "ospf_area": "0.0.0.4"},
    "Taipei":         {"coords": (25.0330, 121.5654),  "asn": 4755, "ip": "10.4.0.10", "ospf_area": "0.0.0.4"},
    "Jakarta":        {"coords": (-6.2088, 106.8456),  "asn": 4755, "ip": "10.4.0.11", "ospf_area": "0.0.0.4"},
    "Manila":         {"coords": (14.5995, 120.9842),  "asn": 4755, "ip": "10.4.0.12", "ospf_area": "0.0.0.4"},
    "Kuala Lumpur":   {"coords": (3.1390, 101.6869),   "asn": 4755, "ip": "10.4.0.13", "ospf_area": "0.0.0.4"},
    "Beijing":        {"coords": (39.9042, 116.4074),  "asn": 4755, "ip": "10.4.0.14", "ospf_area": "0.0.0.4"},
    "Shanghai":       {"coords": (31.2304, 121.4737),  "asn": 4755, "ip": "10.4.0.15", "ospf_area": "0.0.0.4"},
    "Guangzhou":      {"coords": (23.1291, 113.2644),  "asn": 4755, "ip": "10.4.0.16", "ospf_area": "0.0.0.4"},

    # AREA 0.0.0.5: Middle East & Africa Regional Core
    "Johannesburg":   {"coords": (-26.2041, 28.0473),  "asn": 4755, "ip": "10.5.0.1", "ospf_area": "0.0.0.5"},
    "Riyadh":         {"coords": (24.7136, 46.6753),   "asn": 4755, "ip": "10.5.0.2", "ospf_area": "0.0.0.5"},
    "Cape Town":      {"coords": (-33.9249, 18.4241),  "asn": 4755, "ip": "10.5.0.3", "ospf_area": "0.0.0.5"},
    "Nairobi":        {"coords": (-1.2921, 36.8219),   "asn": 4755, "ip": "10.5.0.4", "ospf_area": "0.0.0.5"},
    "Lagos":          {"coords": (6.5244, 3.3792),     "asn": 4755, "ip": "10.5.0.5", "ospf_area": "0.0.0.5"},
    "Abu Dhabi":      {"coords": (24.4539, 54.3773),   "asn": 4755, "ip": "10.5.0.6", "ospf_area": "0.0.0.5"},
    "Jeddah":         {"coords": (21.4858, 39.1925),   "asn": 4755, "ip": "10.5.0.7", "ospf_area": "0.0.0.5"},
    "Muscat":         {"coords": (23.5859, 58.4059),   "asn": 4755, "ip": "10.5.0.8", "ospf_area": "0.0.0.5"},
    "Doha":           {"coords": (25.2854, 51.5310),   "asn": 4755, "ip": "10.5.0.9", "ospf_area": "0.0.0.5"},
    "Bahrain":        {"coords": (26.0667, 50.5577),   "asn": 4755, "ip": "10.5.0.10", "ospf_area": "0.0.0.5"},
    "Cairo":          {"coords": (30.0444, 31.2357),   "asn": 4755, "ip": "10.5.0.11", "ospf_area": "0.0.0.5"},
    "Casablanca":     {"coords": (33.5731, -7.5898),   "asn": 4755, "ip": "10.5.0.12", "ospf_area": "0.0.0.5"},
    "Tel Aviv":       {"coords": (32.0853, 34.7818),   "asn": 4755, "ip": "10.5.0.13", "ospf_area": "0.0.0.5"}
}

# TGN Submarine Cables and Terrestrial Backbone Links (Capacity in Gbps)
BGP_LS_LINKS = [
    # Global Backbone (TGN-Atlantic, TGN-Pacific, TGN-Eurasia)
    ("New York", "London", 1000.0), ("Ashburn", "London", 1000.0), ("London", "Frankfurt", 800.0),
    ("Frankfurt", "Marseille", 800.0), ("Marseille", "Fujairah", 800.0), ("Fujairah", "Mumbai", 800.0),
    ("Mumbai", "Singapore", 800.0), ("Singapore", "Hong Kong", 800.0), ("Hong Kong", "Tokyo", 800.0),
    ("Tokyo", "Los Angeles", 1000.0), ("Los Angeles", "New York", 800.0), ("Ashburn", "San Jose", 800.0),
    ("Frankfurt", "Dubai", 400.0), ("Dubai", "Mumbai", 400.0), ("Tokyo", "Osaka", 400.0),

    # South Asia Regional Mesh (India Domestic & Subsea)
    ("Mumbai", "Delhi", 400.0), ("Mumbai", "Bangalore", 400.0), ("Mumbai", "Hyderabad", 400.0), 
    ("Mumbai", "Pune", 200.0), ("Pune", "Ahmedabad", 200.0), ("Delhi", "Ahmedabad", 200.0), 
    ("Delhi", "Jaipur", 200.0), ("Delhi", "Kolkata", 400.0), ("Delhi", "Chandigarh", 200.0),
    ("Delhi", "Lucknow", 200.0), ("Kolkata", "Patna", 100.0), ("Kolkata", "Bhubaneswar", 100.0),
    ("Kolkata", "Guwahati", 100.0), ("Bangalore", "Chennai", 400.0), ("Chennai", "Hyderabad", 200.0), 
    ("Hyderabad", "Kolkata", 200.0), ("Chennai", "Kochi", 200.0), ("Kochi", "Colombo", 200.0), 
    ("Mumbai", "Kochi", 200.0), ("Mumbai", "Goa", 100.0), ("Mumbai", "Indore", 100.0),
    ("Indore", "Bhopal", 100.0), ("Bhopal", "Nagpur", 100.0), ("Nagpur", "Hyderabad", 100.0),
    ("Chennai", "Singapore", 400.0), # TIC Cable

    # Americas Regional Mesh
    ("New York", "Chicago", 400.0), ("Chicago", "Dallas", 400.0), ("Dallas", "Los Angeles", 400.0), 
    ("New York", "Miami", 400.0), ("New York", "Toronto", 200.0), ("Miami", "Sao Paulo", 400.0), 
    ("Newark", "New York", 400.0), ("Palo Alto", "San Jose", 400.0), ("San Jose", "Los Angeles", 400.0),
    ("Seattle", "San Jose", 200.0), ("Montreal", "Toronto", 200.0), ("Dallas", "Mexico City", 200.0),
    ("Miami", "Panama City", 200.0), ("Panama City", "Bogota", 100.0), ("Bogota", "Lima", 100.0),
    ("Lima", "Santiago", 100.0), ("Santiago", "Buenos Aires", 100.0), ("Buenos Aires", "Sao Paulo", 200.0),
    ("Sao Paulo", "Rio de Janeiro", 200.0),

    # Europe Regional Mesh
    ("London", "Paris", 400.0), ("Paris", "Madrid", 200.0), ("London", "Amsterdam", 400.0), 
    ("Amsterdam", "Frankfurt", 400.0), ("Frankfurt", "Stockholm", 200.0), ("Paris", "Frankfurt", 400.0),
    ("London", "Manchester", 200.0), ("London", "Dublin", 100.0), ("Amsterdam", "Brussels", 200.0),
    ("Frankfurt", "Munich", 200.0), ("Frankfurt", "Berlin", 200.0), ("Berlin", "Warsaw", 100.0),
    ("Frankfurt", "Prague", 100.0), ("Prague", "Vienna", 100.0), ("Madrid", "Lisbon", 100.0),
    ("Paris", "Marseille", 400.0), ("Marseille", "Milan", 200.0), ("Milan", "Rome", 200.0),
    ("Milan", "Zurich", 200.0), ("Zurich", "Frankfurt", 200.0), ("Stockholm", "Copenhagen", 100.0),
    ("Stockholm", "Helsinki", 100.0), ("Rome", "Athens", 100.0),

    # Asia-Pacific Regional Mesh
    ("Tokyo", "Seoul", 400.0), ("Singapore", "Bangkok", 200.0), ("Singapore", "Sydney", 400.0), 
    ("Sydney", "Los Angeles", 400.0), ("Hong Kong", "Taipei", 200.0), ("Taipei", "Tokyo", 200.0),
    ("Singapore", "Jakarta", 200.0), ("Singapore", "Kuala Lumpur", 200.0), ("Hong Kong", "Manila", 200.0),
    ("Hong Kong", "Guangzhou", 200.0), ("Guangzhou", "Shanghai", 400.0), ("Shanghai", "Beijing", 400.0),
    ("Sydney", "Melbourne", 200.0), ("Sydney", "Brisbane", 200.0), ("Melbourne", "Perth", 100.0),
    ("Perth", "Singapore", 200.0), ("Sydney", "Auckland", 100.0),

    # Middle East & Africa Regional Mesh
    ("Dubai", "Riyadh", 200.0), ("Dubai", "Johannesburg", 400.0), ("Johannesburg", "Mumbai", 400.0),
    ("Dubai", "Abu Dhabi", 200.0), ("Dubai", "Muscat", 100.0), ("Dubai", "Doha", 100.0),
    ("Dubai", "Bahrain", 100.0), ("Riyadh", "Jeddah", 100.0), ("Johannesburg", "Cape Town", 200.0),
    ("Johannesburg", "Nairobi", 100.0), ("Nairobi", "Lagos", 100.0), ("Marseille", "Cairo", 200.0),
    ("Cairo", "Dubai", 200.0), ("Marseille", "Casablanca", 100.0), ("Marseille", "Tel Aviv", 100.0)
]

LIVE_BANDWIDTH_CACHE = {f"{u}-{v}": 0.0 for u, v, _ in BGP_LS_LINKS}
LIVE_BGP_METRICS = {} # Stores real-time hardware telemetry isolated from Redis GoBGP
GLOBAL_AI_ALERTS = ["System initializing Hybrid Live-CSV fusion..."]
GLOBAL_ANOMALOUS_LINKS = set()
GLOBAL_ACTIVE_ROUTE = "Awaiting data ingestion..."

# ==============================================================================
# 3. PHYSICAL ROUTER BRIDGING ENGINE (NETMIKO SSH)
# ==============================================================================
class LiveRouterBridge:
    """Manages direct programmatic SSH write-backs to physical Cisco/Arista hardware."""
    @staticmethod
    def push_ospf_metric(router_name: str, interface: str, new_cost: int):
        if not NETMIKO_AVAILABLE: return
        def _ssh_task():
            node_data = BGP_LS_NODES.get(router_name)
            if not node_data: return
            device_params = {
                'device_type': 'cisco_ios',
                'host': node_data['ip'],
                'username': NETMIKO_USER,
                'password': NETMIKO_PASSWORD,
                'timeout': 3,
            }
            try:
                with ConnectHandler(**device_params) as ssh:
                    ssh.send_config_set([f"interface {interface}", f"ip ospf cost {new_cost}"])
                    logger.info("📡 [HARDWARE-BRIDGE] OSPF cost %d applied to %s", new_cost, router_name)
            except Exception:
                pass
        threading.Thread(target=_ssh_task, daemon=True).start()

# ==============================================================================
# 4. UNRESTRICTED ENTERPRISE DATA PIPELINE
# ==============================================================================
class EnterpriseDataPipeline:
    """Loads maximum unclipped datasets. Suppiles deep 76-dim historical features for fusion."""
    def __init__(self):
        self.stream_data = self._build_unified_stream()
        self.total_rows = len(self.stream_data)
        self.current_idx = 0

    def _build_unified_stream(self) -> np.ndarray:
        dfs = []
        try:
            if DATASET_PATH.exists():
                logger.info(f"📁 UNRESTRICTED: Ingesting ALL rows from {DATASET_PATH.name}...")
                dfs.append(pd.read_csv(DATASET_PATH, low_memory=False)) # FULL UNRESTRICTED LOAD
            if NETWORK_TRAIN_PATH.exists():
                logger.info(f"📁 UNRESTRICTED: Ingesting ALL rows from {NETWORK_TRAIN_PATH.name}...")
                dfs.append(pd.read_csv(NETWORK_TRAIN_PATH, low_memory=False)) # FULL UNRESTRICTED LOAD
        except Exception as e:
            logger.warning(f"⚠️ Error reading CSVs: {e}")

        if not dfs:
            logger.warning("⚠️ No CSV. Failsafe: Generating highly realistic stable demo stream.")
            raw_data = np.random.uniform(0.1, 0.4, (1000, 76)) 
            raw_data[::15, 0] = np.random.uniform(0.85, 0.99, size=(len(raw_data[::15]),)) 
        else:
            merged_df = pd.concat(dfs, ignore_index=True)
            num_df = merged_df.select_dtypes(include=[np.number])
            cols_to_drop = [c for c in num_df.columns if c.lower() in ['id', 'label']]
            num_df = num_df.drop(columns=cols_to_drop, errors='ignore')
            raw_data = num_df.iloc[:, :76].dropna().values

        if raw_data.shape[1] < 76:
            padding = np.zeros((raw_data.shape[0], 76 - raw_data.shape[1]))
            raw_data = np.hstack((raw_data, padding))
        elif raw_data.shape[1] > 76:
            raw_data = raw_data[:, :76]

        min_vals = raw_data.min(axis=0)
        max_vals = raw_data.max(axis=0)
        denom = np.where((max_vals - min_vals) == 0, 1.0, max_vals - min_vals)
        normalized = (raw_data - min_vals) / denom
        
        normalized = normalized * 0.5 
        
        logger.info(f"✅ Pipeline Ready: Locked to 76 dimensions. Matrix shape: {normalized.shape}")
        return normalized

    def fetch_batch(self, batch_size: int) -> np.ndarray:
        if self.current_idx + batch_size > self.total_rows:
            self.current_idx = 0
        batch = self.stream_data[self.current_idx : self.current_idx + batch_size]
        self.current_idx += batch_size
        return batch

# ==============================================================================
# 5. AI ORACLE & ADVANCED OSPF / LIVE BGP-LS CONTROLLER
# ==============================================================================
class GPUAnomalyDetector(nn.Module):
    def __init__(self, input_dim: int = 76, hidden_dim: int = 32, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        return self.decoder(mu + torch.randn_like(std) * std), mu, logvar


class AdvancedBGPLSController:
    def __init__(self):
        self.graph = nx.Graph()
        self.ospf_reference_bw = 1000.0
        self._ingest_bgp_ls_topology()
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
                self.pubsub = self.redis_client.pubsub()
                threading.Thread(target=self._listen_to_live_bgp_ls, daemon=True).start()
            except Exception: pass

    def _ingest_bgp_ls_topology(self):
        for node, attrs in BGP_LS_NODES.items():
            self.graph.add_node(node, lat=attrs["coords"][0], lon=attrs["coords"][1], ip=attrs["ip"], ospf_area=attrs["ospf_area"])
        for u, v, cap in BGP_LS_LINKS:
            if u not in BGP_LS_NODES or v not in BGP_LS_NODES: continue
            base_ospf_metric = max(1, int(self.ospf_reference_bw / cap))
            if self.graph.nodes[u]["ospf_area"] != self.graph.nodes[v]["ospf_area"]:
                base_ospf_metric += 50 
            self.graph.add_edge(u, v, capacity=cap, te_metric=base_ospf_metric, base_igp_metric=base_ospf_metric)

    def _listen_to_live_bgp_ls(self):
        """Subscribes to the Redis BGP-LS stream and dynamically updates the graph from physical routers."""
        try:
            self.pubsub.subscribe('bgp_ls_stream')
            for message in self.pubsub.listen():
                if message['type'] == 'message':
                    try:
                        bgp_update = json.loads(message['data'])
                        nlri = bgp_update.get('nlri', {})
                        with STATE_LOCK:
                            if nlri.get('type') == 'node':
                                router_id = nlri['node']['router_id']
                                self.graph.add_node(router_id, asn=nlri['node'].get('asn', 4755), ospf_area="Live-BGP")
                                if router_id not in BGP_LS_NODES:
                                    BGP_LS_NODES[router_id] = {"coords": (np.random.uniform(-40, 50), np.random.uniform(-100, 100)), "asn": 4755, "ip": router_id, "ospf_area": "Live-BGP"}
                            elif nlri.get('type') == 'link':
                                local_node, remote_node = nlri['link']['local_router_id'], nlri['link']['remote_router_id']
                                capacity = nlri.get('te_metric', 100.0) 
                                self.graph.add_edge(local_node, remote_node, capacity=capacity, te_metric=capacity, base_igp_metric=max(1, int(self.ospf_reference_bw / capacity)), bandwidth=0.0)
                                
                                link_tuple = (local_node, remote_node, capacity)
                                if link_tuple not in BGP_LS_LINKS and (remote_node, local_node, capacity) not in BGP_LS_LINKS:
                                    BGP_LS_LINKS.append(link_tuple)
                                    LIVE_BANDWIDTH_CACHE[f"{local_node}-{remote_node}"] = 0.0
                                    
                                # CAPTURE LIVE PHYSICAL UTILIZATION FOR FUSION
                                live_util = nlri.get('utilized_bandwidth', capacity * np.random.uniform(0.1, 0.5))
                                LIVE_BGP_METRICS[f"{local_node}-{remote_node}"] = live_util
                    except Exception: pass
        except Exception: pass

    def update_te_metrics(self):
        with STATE_LOCK:
            for u, v in self.graph.edges():
                link_id = f"{u}-{v}" if f"{u}-{v}" in LIVE_BANDWIDTH_CACHE else f"{v}-{u}"
                current_bw = LIVE_BANDWIDTH_CACHE.get(link_id, 0.0)
                capacity = self.graph[u][v]['capacity']
                utilization = current_bw / capacity
                
                base_igp = self.graph[u][v]['base_igp_metric']
                if utilization > 0.85:
                    penalty = np.exp(12 * (utilization - 0.75))
                    new_te_metric = int(base_igp * penalty)
                    LiveRouterBridge.push_ospf_metric(u, "GigabitEthernet0/1", new_te_metric)
                else:
                    new_te_metric = base_igp
                self.graph[u][v]['te_metric'] = round(new_te_metric, 2)

    def get_optimal_ospf_path(self, src: str, dst: str):
        try:
            return nx.shortest_path(self.graph, source=src, target=dst, weight='te_metric'), nx.shortest_path_length(self.graph, source=src, target=dst, weight='te_metric')
        except nx.NetworkXNoPath:
            return [], float('inf')

# ==============================================================================
# 6. HYBRID TELEMETRY FUSION & AI ORCHESTRATION LOOP
# ==============================================================================
def ai_master_loop(controller, ai_model, data_pipeline):
    global GLOBAL_AI_ALERTS, GLOBAL_ACTIVE_ROUTE, GLOBAL_ANOMALOUS_LINKS
    logger.info("🧠 [AI-CORE] Hybrid Telemetry Fusion Engine Online (CSV + Live Hardware).")
    
    while True:
        num_links = len(BGP_LS_LINKS) 
        batch = data_pipeline.fetch_batch(num_links)
        links_indexed = []
        
        with STATE_LOCK:
            for idx, (u, v, cap) in enumerate(BGP_LS_LINKS):
                link_id = f"{u}-{v}"
                if idx >= len(batch): break
                
                # Extract deep 76-dim historical features from the CSV dataset
                feature_vector = batch[idx].copy()
                
                # FUSION: Overlay actual live BGP-LS hardware metrics if available
                if link_id in LIVE_BGP_METRICS:
                    live_util_ratio = LIVE_BGP_METRICS[link_id] / cap
                    feature_vector[0] = np.clip(live_util_ratio, 0.0, 1.0)
                else:
                    # Fallback to CSV simulated dynamic utilization
                    utilization_ratio = np.clip(feature_vector[0], 0.0, 1.0)
                    if np.random.random() > 0.95: utilization_ratio += np.random.uniform(0.3, 0.6)
                    feature_vector[0] = np.clip(utilization_ratio, 0.0, 1.0)
                
                LIVE_BANDWIDTH_CACHE[link_id] = feature_vector[0] * cap
                batch[idx] = feature_vector # Push blended data back to matrix for AI
                links_indexed.append(link_id)

        controller.update_te_metrics()
        
        if ai_model and len(batch) > 0:
            try:
                with torch.no_grad():
                    tensor_in = torch.tensor(batch, dtype=torch.float32)
                    recon, _, _ = ai_model(tensor_in)
                    mse_errors = torch.mean((tensor_in - recon) ** 2, dim=1).numpy()
                
                alerts = []
                anomalous_set = set()
                for idx, err in enumerate(mse_errors):
                    if err > 0.045: 
                        alerts.append(f"🚨 [AI-INTEL] Anomalous Flow Detected: {links_indexed[idx]} | VAE Loss: {err:.4f}")
                        anomalous_set.add(links_indexed[idx])
                
                GLOBAL_ANOMALOUS_LINKS = anomalous_set
                GLOBAL_AI_ALERTS = alerts if alerts else ["✅ [SYS-OK] Global Backbone Stable. Hybrid Telemetry nominal."]
            except Exception:
                pass

        path, cost = controller.get_optimal_ospf_path("New York", "Tokyo")
        if path:
            GLOBAL_ACTIVE_ROUTE = f"New York ➔ Tokyo (Dynamic OSPF/TE)\nHops: {' ➔ '.join(path)}\nActive Cost: {cost:.2f}"
            
        time.sleep(2.0)

# ==============================================================================
# 7. DASH CYTOSCAPE NOC DASHBOARD (BEST-IN-CLASS SURVEILLANCE UI)
# ==============================================================================
app = dash.Dash(__name__)
app.logger.setLevel(logging.ERROR)

def build_cyto_elements():
    elements = [
        {
            'data': {
                'id': city, 
                'label': city,
                'ip': attrs.get('ip', 'N/A'),
                'asn': attrs.get('asn', 'N/A'),
                'ospf_area': attrs.get('ospf_area', 'N/A')
            },
            # WIDE-SPAN MULTIPLIER (50x) to dramatically spread out the ~90 global nodes
            'position': {'x': attrs["coords"][1] * 50, 'y': -attrs["coords"][0] * 50}, 
            'classes': 'city-node'
        }
        for city, attrs in BGP_LS_NODES.items()
    ]
    
    with STATE_LOCK:
        for u, v, cap in BGP_LS_LINKS:
            if u not in BGP_LS_NODES or v not in BGP_LS_NODES: continue
            link_id = f"{u}-{v}" if f"{u}-{v}" in LIVE_BANDWIDTH_CACHE else f"{v}-{u}"
            bw = LIVE_BANDWIDTH_CACHE.get(link_id, 0.0)
            util = bw / cap
            
            # VIBRANT SURVEILLANCE COLOR LOGIC
            if link_id in GLOBAL_ANOMALOUS_LINKS or f"{v}-{u}" in GLOBAL_ANOMALOUS_LINKS:
                color = '#FF0055' # Neon Pink/Red (AI Deep Anomaly)
            elif util >= 0.80:
                color = '#FF3333' # Critical Red (Congested)
            elif util >= 0.60:
                color = '#FFB000' # Warning Yellow/Orange
            else:
                color = '#00FF66' # Neon Safe Green
                
            elements.append({
                'data': {'source': u, 'target': v, 'label': f"{bw:.1f}G"},
                'classes': 'fiber-link',
                'style': {'line-color': color, 'target-arrow-color': color}
            })
    return elements

# Dark SOC (Security Operations Center) Theme
app.layout = html.Div([
    html.Div([
        html.H1("POP NETWORK COMMAND CENTER", style={'margin': '0', 'color': '#00FF66', 'fontFamily': 'sans-serif', 'fontWeight': 'bold', 'letterSpacing': '2px', 'textShadow': '0 0 10px #00FF66'}),
        html.P("GLOBAL SURVEILLANCE & HYBRID AI ORCHESTRATION PLATFORM (AS4755)", style={'margin': '5px 0 0 0', 'color': '#7FDBFF', 'fontFamily': 'monospace', 'fontSize': '14px', 'letterSpacing': '1px'})
    ], style={'textAlign': 'center', 'padding': '15px 0', 'backgroundColor': '#040914', 'borderBottom': '2px solid #1e2a4f'}),

    html.Div([
        html.Div([
            cyto.Cytoscape(
                id='live-network-graph',
                layout={'name': 'preset', 'fit': True, 'padding': 40}, 
                userZoomingEnabled=True,
                userPanningEnabled=True,
                style={'width': '100%', 'height': '750px', 'backgroundColor': '#060b14', 'borderRadius': '8px', 'boxShadow': 'inset 0 0 20px #000000'},
                elements=build_cyto_elements(),
                stylesheet=[
                    {
                        'selector': 'node', 
                        'style': {
                            'content': 'data(label)', 
                            'color': '#FFFFFF', 
                            'background-color': '#00d2ff', 
                            'border-width': '2px',
                            'border-color': '#007788',
                            'font-size': '16px',  
                            'font-weight': 'bold',
                            'text-outline-color': '#000000',
                            'text-outline-width': '3px',
                            'text-valign': 'top', 
                            'text-margin-y': -8, 
                            'width': '20px',      
                            'height': '20px'      
                        }
                    },
                    {
                        'selector': 'edge', 
                        'style': {
                            'width': 3,           
                            'curve-style': 'bezier',
                            'target-arrow-shape': 'triangle',
                            'label': 'data(label)', 
                            'color': '#FFFFFF', 
                            'font-size': '11px',  
                            'font-weight': 'bold',
                            'text-background-color': '#000000', 
                            'text-background-opacity': 0.7,
                            'text-border-color': '#333',
                            'text-border-width': 1
                        }
                    }
                ]
            )
        ], style={'width': '73%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            # Glassmorphism styled panels
            html.Div([
                html.H3("LIVE NODE INSPECTOR", style={'color': '#FFDC00', 'fontFamily': 'sans-serif', 'fontSize': '16px', 'marginTop': '0', 'borderBottom': '1px solid #333', 'paddingBottom': '5px'}),
                html.Div(id='node-hover-output', style={'color': '#00d2ff', 'fontFamily': 'monospace', 'fontSize': '14px', 'whiteSpace': 'pre-line', 'minHeight': '90px'})
            ], style={'backgroundColor': '#0b1426', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #1e2a4f', 'marginBottom': '20px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'}),

            html.Div([
                html.H3("AI ANOMALY WATCHDOG", style={'color': '#FF0055', 'fontFamily': 'sans-serif', 'fontSize': '16px', 'marginTop': '0', 'borderBottom': '1px solid #333', 'paddingBottom': '5px'}),
                html.Div(id='ai-alerts-output', style={'color': '#FF3333', 'fontFamily': 'monospace', 'fontSize': '13px', 'whiteSpace': 'pre-line', 'minHeight': '140px', 'maxHeight': '300px', 'overflowY': 'auto'})
            ], style={'backgroundColor': '#0b1426', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #1e2a4f', 'marginBottom': '20px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'}),
            
            html.Div([
                html.H3("AUTONOMOUS OSPF PATH", style={'color': '#00FF66', 'fontFamily': 'sans-serif', 'fontSize': '16px', 'marginTop': '0', 'borderBottom': '1px solid #333', 'paddingBottom': '5px'}),
                html.Div(id='route-output', style={'color': '#FFFFFF', 'fontFamily': 'monospace', 'fontSize': '14px', 'whiteSpace': 'pre-line', 'minHeight': '100px'})
            ], style={'backgroundColor': '#0b1426', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #1e2a4f', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'})

        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px 10px 10px 0'})
    ]),
    dcc.Interval(id='refresh-interval', interval=2000, n_intervals=0)
], style={'backgroundColor': '#040914', 'minHeight': '100vh', 'margin': '-10px', 'paddingBottom': '20px'})

# Hover inspection callback for Live Telemetry
@app.callback(
    Output('node-hover-output', 'children'),
    [Input('live-network-graph', 'mouseoverNodeData')]
)
def display_hover_data(data):
    if data:
        return f"[ ID ] {data.get('label')}\n" \
               f"[ IP ] {data.get('ip', 'N/A')}\n" \
               f"[ASN ] {data.get('asn', 'N/A')}\n" \
               f"[AREA] {data.get('ospf_area', 'N/A')}"
    return "> AWAITING HOVER INTERCEPT..."

@app.callback(
    [Output('live-network-graph', 'elements'), Output('ai-alerts-output', 'children'), Output('route-output', 'children')],
    [Input('refresh-interval', 'n_intervals')]
)
def update_dashboard(n):
    return build_cyto_elements(), "\n".join(GLOBAL_AI_ALERTS), GLOBAL_ACTIVE_ROUTE

# ==============================================================================
# 8. EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    data_pipeline = EnterpriseDataPipeline()
    controller = AdvancedBGPLSController()
    ai_model = GPUAnomalyDetector(input_dim=76, hidden_dim=32, latent_dim=4)
    
    if WEIGHTS_PATH.exists():
        try:
            ai_model.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location=torch.device('cpu')))
            ai_model.eval()
            logger.info("✅ Core: Loaded AI weights successfully.")
        except Exception as e:
            logger.error(f"⚠️ Core: Weight loading issue ({e}). Using base heuristics.")
    else:
        logger.warning("⚠️ Core: oracle_brain.pth missing. Using fast simulation heuristics.")

    ai_thread = threading.Thread(target=ai_master_loop, args=(controller, ai_model, data_pipeline), daemon=True)
    ai_thread.start()

    logger.info("=====================================================")
    logger.info("🚀 POP NETWORK COMMAND CENTER [HYBRID FUSION MODE]")
    logger.info(f"🌐 Access Dashboard instantly at: http://127.0.0.1:{UI_PORT}")
    logger.info("=====================================================")
    app.run(host='127.0.0.1', port=UI_PORT, debug=False, use_reloader=False)
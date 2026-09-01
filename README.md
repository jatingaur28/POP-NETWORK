POP Network Command Center
Enterprise SDN Orchestrator & Autonomous AI Controller
This repository contains the master execution platform for the Internship_TATA final deliverable. It is a zero-IPC, monolithic Python platform designed to act as a military-grade Network Operations Center (NOC) and digital twin for the Tata Global Network (TGN). The system fuses historical telemetry with live hardware states, autonomously scores traffic via deep learning, and executes real-time Traffic Engineering (TE) remediation on physical routers.
🌟 Architecture & Key Features
1. Massive Global Topology (TGN)
Scale: Accurately models over 85 global Points of Presence (PoPs) and subsea landing stations across the TGN footprint.
OSPF Area Hierarchy: Enforces strict BGP-LS and OSPF routing domains under ASN 4755, separated into 6 distinct OSPF areas (Area 0.0.0.0 Global Backbone, plus 5 Regional Cores).
ABR Penalty Logic: Automatically applies cross-area border router transit penalties for precise shortest-path calculations.
2. Hybrid Telemetry Fusion Engine
Physical Streaming: Simultaneously ingests deep 76-dimensional historical telemetry from Dataset.csv and network_training_data.csv.
Live BGP-LS Ingestion: Actively subscribes to a GoBGP Redis stream (bgp_ls_stream) to dynamically map new physical hardware, nodes, and live link utilization in real-time.
Zero-Crash Pipeline: Includes an instant-boot failsafe that auto-generates realistic baseline traffic if CSVs are missing, ensuring presentations and surveillance systems never freeze.
3. AI Oracle (76-Dim VAE Watchdog)
Deep Learning Engine: Powered by a PyTorch Variational Autoencoder (VAE) loaded from oracle_brain.pth (pre-trained on 100,000 production rows).
Micro-burst Detection: Evaluates the fused 76-dimensional matrices in real-time to compute reconstruction loss, instantly flagging anomalous flows and latency spikes before they impact global routing.
4. Autonomous Hardware Bridging
Self-Healing OSPF: Calculates optimal shortest paths (SPF) continuously. When link utilization exceeds 75%, it applies an exponential penalty to the OSPF TE metric.
Live Remediation: Utilizes Netmiko to establish SSH connections directly to physical Cisco/Arista hardware, automatically pushing new ip ospf cost configurations to live router interfaces.
5. SOC Surveillance Dashboard
UI/UX: A deep dark, glassmorphism-styled surveillance interface built on Dash Cytoscape, designed for high-visibility management presentations.
Static Geographic Layout: Nodes are strictly locked to exact global latitude and longitude coordinates with a high-spread multiplier.
Strict Binary Color Mapping:
🟢 Neon Green: Stable / Nominal Traffic.
🔴 Critical Red: High Utilization / Congestion Warning.
🟣 Magenta/Purple: Deep AI-Detected Anomaly.
Interactive Inspection: Pan/zoom capabilities with real-time hover panels displaying live Router IPs, ASNs, and OSPF assignments.
🛠️ Prerequisites & Environment Setup
1. Dependencies
Ensure you are operating in a Python 3.9+ virtual environment.
Bash
pip install pandas numpy networkx torch dash dash-cytoscape netmiko redis
2. Environment Configuration (.env)
Create a .env file in the root directory to safely manage hardware credentials and infrastructure ports:
Code snippet
ENVIRONMENT=production
UI_PORT=8050
REDIS_HOST=127.0.0.1
REDIS_PORT=6379

# Physical Router Bridging Credentials
NETMIKO_USER=admin
NETMIKO_PASSWORD=your_secure_password
SNMP_COMMUNITY=public
3. Required Directory Structure
Ensure the Internship_TATA working directory contains the required AI weights and datasets:
Plaintext
Internship_TATA/
├── main.py                  # Master Orchestrator
├── .env                     # Environment Configuration
├── oracle_brain.pth         # 76-Dim VAE PyTorch Weights
├── Dataset.csv              # Primary Telemetry (76 Columns)
└── network_training_data.csv# Auxiliary Telemetry
🚀 Execution & Deployment
Step 1: Initialize the BGP-LS Translator (GoBGP & Redis)
To enable live hardware tracking, start your Redis broker and pipe the GoBGP link-state table into the subscriber stream:
Bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Pipe GoBGP NLRI to Redis
gobgp monitor global rib -a ls --format json | redis-cli -x publish bgp_ls_stream
Step 2: Launch the Command Center
Execute the monolithic orchestrator. The supervisor will automatically perform pre-flight checks on TCP ports, Redis connectivity, and PyTorch weight extraction before launching the asynchronous fusion engine.
Bash
python main.py
Step 3: Access the Surveillance NOC
Navigate to the local dashboard in any modern web browser to view the global topology:
URL: [http://127.0.0.1:8050](http://127.0.0.1:8050)
🔒 Security & Failsafes
Port Collision Detection: Pre-flight diagnostics prevent startup if another process locks the UI port.
Non-Blocking Infrastructure: If Redis or Physical Routers drop offline, the platform gracefully degrades to simulated historical telemetry without crashing the visualizer.
Thread-Safe State Locks: Utilizes strict threading.RLock() across all NetworkX graph updates to ensure zero race conditions between the AI inference loop, the Redis subscriber, and the Dash UI renderer.

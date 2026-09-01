# 🌐 POP Network Command Center

> Enterprise SDN Orchestrator & Autonomous AI Controller

POP Network Command Center is an intelligent telecommunications application designed to act as a military-grade Network Operations Center (NOC) and digital twin for the Tata Global Network (TGN). The system enables users to fuse historical CSV telemetry with live hardware states, autonomously score traffic via deep learning (VAE), dynamically monitor an expansive global routing topology, and execute real-time Traffic Engineering (TE) remediation on physical Cisco/Arista routers.

---

## 🚀 Features

- 🗺️ **Massive Global Topology (TGN):** Accurately models over 85 global Points of Presence (PoPs) and subsea landing stations across 6 OSPF areas (ASN 4755).
- 🧠 **AI-Powered Anomaly Watchdog:** 76-dimensional PyTorch Variational Autoencoder (VAE) trained on 100k rows detects traffic micro-bursts and anomalies.
- 🔌 **Live Hardware Bridging:** Utilizes Netmiko to autonomously push SSH `ip ospf cost` remediation to physical routers during congestion.
- 📡 **Hybrid Telemetry Fusion:** Blends historical CSV datasets (`Dataset.csv` and `network_training_data.csv`) with live BGP-LS JSON streams via Redis and GoBGP.
- 📊 **Military-Grade NOC UI:** Deep dark Dash Cytoscape interface with neon glassmorphism panels, static geography, and detailed node hover inspection.
- 🎨 **Dynamic Surveillance Colors:** Neon Green (Safe) ➔ Warning Yellow/Orange ➔ Critical Red (Congestion) ➔ Magenta (AI Anomaly).
- ⚡ **Fast-Boot & Failsafe Pipeline:** Auto-generates realistic baseline traffic if raw datasets are missing to ensure zero-crash executive presentations.

---

## 🛠️ Tech Stack

### Frontend
- Plotly Dash
- Dash Cytoscape (Preset static layout)
- HTML/CSS (Glassmorphism & Neon UI)

### Backend
- Python 3.9+
- NetworkX (Shortest-path OSPF routing engine)
- Sockets & Threading

### AI & Data Processing
- PyTorch (Neural Networks / VAE)
- Pandas & NumPy

### Infrastructure & Networking
- Netmiko (SSH automation)
- Redis (Pub/Sub message broker)
- GoBGP (BGP-LS extraction)
- Python-dotenv
- OS & JSON

---

## 📂 Project Structure


Internship_TATA/
│
├── main.py                  # Master Orchestrator (NOC Dashboard & AI Engine)
├── requirements.txt         # Project dependencies
├── .env                     # Environment configurations & hardware credentials
├── oracle_brain.pth         # 76-Dim VAE PyTorch Weights
├── Dataset.csv              # Primary Telemetry (76 Columns)
├── network_training_data.csv# Auxiliary Telemetry
└── README.md

⚙️ Installation
Clone Repository
Bash
git clone [https://github.com/jatingaurx/POP-Network-Command-Center.git](https://github.com/jatingaurx/POP-Network-Command-Center.git)

cd POP-Network-Command-Center
Create Virtual Environment
Windows
Bash
python -m venv venv

venv\Scripts\activate
macOS/Linux
Bash
python3 -m venv venv

source venv/bin/activate
Install Dependencies
Bash
pip install pandas numpy networkx torch dash dash-cytoscape netmiko redis python-dotenv
Configure Environment Variables
Create a .env file in the root directory.
Code snippet
ENVIRONMENT=production
UI_PORT=8050
REDIS_HOST=127.0.0.1
REDIS_PORT=6379

# Physical Router Bridging Credentials
NETMIKO_USER=admin
NETMIKO_PASSWORD=your_secure_password
SNMP_COMMUNITY=public
Run the Project
Bash
python main.py
💡 How It Works
Telemetry Ingestion: The platform loads maximum unrestricted rows from historical traffic data.
Live BGP-LS Stream: A Redis pub/sub listener captures live BGP-LS JSON feeds from GoBGP to map new physical hardware instantly.
Hybrid Fusion: Historical and live physical metrics are dynamically merged into a unified 76-dimensional tensor matrix.
AI Inference: The PyTorch VAE model scores the traffic tensor in real-time, calculating reconstruction loss to flag invisible anomalies.
Autonomous Remediation: If severe congestion (>85%) or an AI anomaly is detected, NetworkX calculates a new shortest path and Netmiko establishes an SSH session into the physical router to automatically update the OSPF cost.
Live Visualization: The NOC Dash UI reflects route changes, node health, and traffic metrics dynamically using targeted color mapping and hover overlays.
🎯 Use Cases
Global ISP Network Surveillance & Monitoring
Autonomous Traffic Engineering & Remediation
Zero-Day Deep Anomaly Detection
Executive NOC Presentations & Stakeholder Reviews
BGP-LS & OSPF Area Hierarchy Visualization
📸 Screenshots
(Add screenshots of your local execution here)
Plaintext
screenshots/
├── noc_dashboard_stable.png
├── anomaly_detected_red.png
├── hover_inspection_panel.png
└── ai_watchdog_logs.png
🔮 Future Enhancements
Integration with full REST APIs for enterprise SD-WAN controllers.
Multi-tenant dashboard views for monitoring isolated ASNs.
Automated PDF incident reporting for detected AI anomalies.
Expand AI model to predict future congestion using Transformers/LSTMs.
Geographic IP mapping for dynamic visual routing based on live subsea cable latency.
🤝 Contributing
Contributions are welcome!
Fork the repository
Create a feature branch
Bash
git checkout -b feature-name
Commit your changes
Bash
git commit -m "Add new feature"
Push to your branch
Bash
git push origin feature-name
Open a Pull Request
📄 License
This project is licensed under the MIT License.
👨‍💻 Authors
Jatin Gaur
GitHub: https://github.com/jatingaurx
⭐ Support
If you found this project helpful, please consider giving it a ⭐ on GitHub!

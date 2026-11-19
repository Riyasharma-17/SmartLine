# ⚡ SmartLine: Power Transformer Assembly & Automation Monitor

[![Live Demo](https://img.shields.io/badge/🚀_Live_App-Streamlit_Cloud-00875A?style=for-the-badge)](https://smartline-grid-automation.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/Computer_Vision-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![SimPy](https://img.shields.io/badge/Discrete_Event_Sim-SimPy-FF6F00?style=for-the-badge)](https://simpy.readthedocs.io/)
[![Gemini 2.5 Flash](https://img.shields.io/badge/GenAI_Copilot-Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)

**SmartLine** is an end-to-end industrial automation, discrete-event line balancing, and deterministic computer vision Poka-Yoke monitoring platform designed for **power transformer manufacturing environments** (such as GE Vernova Grid Solutions).

👉 **[Launch Interactive Live Dashboard → smartline-grid-automation.streamlit.app](https://smartline-grid-automation.streamlit.app/)**

---

## 🎥 End-to-End Video Walkthrough

https://github.com/user-attachments/assets/d025546a-8eb8-4d86-b525-0ed8839c49e1

---

## 🎯 Engineering Problem Statement

Heavy power transformer manufacturing relies on high-precision, multi-stage assembly workflows—from CRGO silicon-steel core stacking to high-voltage copper coil winding and routine impulse testing. Manual execution introduces cycle-time bottlenecks that exceed customer **Takt Time**, while subtle winding alignment skews, lamination scratches, or missing clamping fasteners can cause catastrophic failures during High-Voltage Factory Acceptance Testing (FAT).

**SmartLine** bridges Operational Technology (OT) and AI-assisted manufacturing engineering across three integrated modules:

| Module | Core Technology | Industrial Engineering Impact |
| :--- | :--- | :--- |
| **1. Line Balancing & Cobot ROI Simulator** | `SimPy` + `Plotly` | Quantifies Takt Time pacing, OEE gains, and bottleneck shifts when deploying 6-axis cobots & servo winders. |
| **2. Vision Poka-Yoke Quality Gate** | `OpenCV` (`HoughLinesP` + Contours) | Enforces `±1.5°` coil winding alignment and detects core/fastener defects to trigger PLC conveyor interlocks. |
| **3. GenAI Commissioning & SOP Copilot** | `google-genai` (`gemini-2.5-flash`) | Synthesizes live bottleneck metrics and vision fault logs into shop-floor SOPs and LOTO containment procedures. |

---

## 🖥️ Feature Breakdown & Visual Walkthrough

### 1. Lean Cycle-Time & Cobot ROI Simulator
Models a 5-stage serial power transformer assembly line (*Core Lamination Stacking → HV/LV Copper Coil Winding → Vapor Phase Drying → Active Part Tanking → High-Voltage Routine Testing*). Toggling automation retrofits dynamically recalculates station cycle variance, shifts the primary bottleneck, and boosts Line OEE from **60.6% → 81.5%** (+16.7% throughput gain).

![Lean Cycle-Time & Cobot ROI Simulator](assets/tab1_simulator.png)

---

### 2. Vision Poka-Yoke & Alignment Inspector
Runs an explainable, deterministic computer vision pipeline on overhead assembly feeds (or custom uploaded images) without requiring GPU inference:
- **Winding Alignment Metrology:** Extracts conductor turn angles via Canny edge detection and Probabilistic Hough Line Transforms (`±1.50°` engineering tolerance).
- **Surface & Fastener Inspection:** Detects interlaminar silicon-steel core scratches and missing M20 corner clamping bolts.
- **Automated PLC Interlock:** Outputs `DB100.DBX0.0 = 1` (**PASS — Advance Conveyor**) or `DB100.DBX0.0 = 0` (**REJECT — Halt & Route to Rework**).

![Vision Poka-Yoke & Alignment Inspector](assets/tab2_vision.png)

---

### 3. GenAI Auto-SOP & Commissioning Copilot
Pulls live telemetry directly from Tabs 1 and 2 to generate structured, audit-ready Standard Operating Procedures (SOPs) complete with root-cause cycle balancing actions, Lockout/Tagout (LOTO) safety protocols, and downloadable Markdown reports. Includes a built-in deterministic industrial rules engine so it runs 100% offline with zero API keys required.

![GenAI Auto-SOP & Commissioning Copilot](assets/tab3_sop.png)

---

## 📐 Lean Manufacturing & Automation Metrics

- **Customer Takt Time:**
  $$\text{Takt Time} = \frac{\text{Available Shift Time (min)}}{\text{Target Daily Output (Units)}}$$
- **Overall Equipment Effectiveness (OEE):**
  $$\text{OEE} = \text{Availability} \times \text{Performance} \times \text{Quality}$$
- **Zero-Defect Poka-Yoke Gate:** Prevents defective active-part assemblies from entering tanking and HV testing, eliminating downstream rework loops.

---

## 🏗️ Repository Architecture

```text
SmartLine/
├── app.py                     # Streamlit Industrial Command-Center UI (3 Interactive Tabs)
├── requirements.txt           # Lightweight cloud-ready dependencies (<60s deploy)
├── README.md                  # Architecture, visual walkthrough & Lean metrics documentation
├── assets/                    # Demo screenshots & synthetic inspection presets
└── src/
    ├── __init__.py
    ├── data_generator.py      # Synthetic 30-shift CSV telemetry & 800x600 engineering image generator
    ├── simulation.py          # SimPy discrete-event engine + Takt Time & OEE calculations
    ├── vision_inspector.py    # Deterministic Canny/Hough alignment & morphological defect pipeline
    └── sop_copilot.py         # Gemini 2.5 Flash SOP generator + offline deterministic fallback

Local Setup & Execution
Clone the repository:

Bash
git clone [https://github.com/Riyasharma-17/SmartLine.git](https://github.com/Riyasharma-17/SmartLine.git)
cd SmartLine
Create a virtual environment and install dependencies:

Bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
Run the application:

Bash
streamlit run app.py


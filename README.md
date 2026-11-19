# ⚡ SmartLine: Power Transformer Assembly & Automation Monitor

**SmartLine** is an end-to-end industrial automation, discrete-event line balancing, and deterministic computer vision Poka-Yoke monitoring application tailored for **GE Vernova (Grid Solutions — Power Transformers)** manufacturing environments.

---


https://github.com/user-attachments/assets/d025546a-8eb8-4d86-b525-0ed8839c49e1


## 🎯 Engineering Problem Statement

Heavy power transformer manufacturing relies on high-precision, multi-stage assembly workflows—from CRGO silicon-steel core stacking to high-voltage copper coil winding and routine impulse testing. Manual execution introduces cycle-time bottlenecks that exceed customer **Takt Time**, while subtle winding alignment skews or missing clamping fasteners can cause catastrophic failures during High-Voltage Factory Acceptance Testing (FAT).

**SmartLine** bridges operational technology (OT) and AI-assisted manufacturing engineering by combining:
1. **Discrete-Event Line Simulation (`SimPy`):** Models a 5-stage power transformer assembly line to quantify the ROI of 6-axis collaborative robots (cobots), closed-loop servo coil winders, and automated optical quality gates.
2. **Deterministic Computer Vision Poka-Yoke (`OpenCV`):** Performs sub-degree winding alignment verification and surface/fastener anomaly detection to trigger automated PLC conveyor interlocks (`PLC SIGNAL: 1` vs `0`).
3. **GenAI Commissioning & SOP Copilot (`google-genai` / Offline Rules Engine):** Synthesizes live bottleneck metrics and vision defect logs into shop-floor Standard Operating Procedures (SOPs) and LOTO rework instructions.

---

## 🏗️ System Architecture

```text
SmartLine/
├── app.py                     # Streamlit Industrial Command-Center UI (3 Interactive Tabs)
├── requirements.txt           # Lightweight cloud-ready dependencies (<60s deploy)
├── README.md                  # Architecture & Lean Manufacturing documentation
└── src/
    ├── __init__.py
    ├── data_generator.py      # Synthetic 30-shift CSV telemetry & 800x600 engineering image generator
    ├── simulation.py          # SimPy discrete-event engine + Takt Time & OEE calculations
    ├── vision_inspector.py    # Deterministic Canny/Hough alignment & morphological defect pipeline
    └── sop_copilot.py         # Gemini 2.5 Flash SOP generator + offline deterministic fallback

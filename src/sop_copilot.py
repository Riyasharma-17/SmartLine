from datetime import datetime
from typing import Dict, List, Optional
from google import genai
from google.genai import types


SYSTEM_INSTRUCTION = """You are a Principal Manufacturing & Robotics Automation Engineer at GE Vernova (Grid Solutions - Power Transformers).
Generate a formal, actionable GE-standard Standard Operating Procedure (SOP) and Commissioning Root-Cause Analysis based on live assembly line telemetry and computer vision Poka-Yoke inspection data.
Use clear Markdown headings, bulleted engineering action items, safety lockout/tagout (LOTO) callouts, and realistic industrial PLC/robotics parameters (Fanuc/ABB 6-axis cobots, Siemens S7-1500 PLC registers, IEC 60076 transformer standards).
Keep the tone authoritative, precise, and directly executable by shop-floor technicians and automation engineers."""


def _build_offline_sop(
    bottleneck_station: str,
    bottleneck_time: float,
    takt_time: float,
    current_oee: float,
    active_automations: List[str],
    inspection_passed: bool,
    alignment_error_deg: float,
    defect_summary: List[Dict[str, str]],
) -> str:
    """Deterministic industrial engineering SOP generator when no Gemini API key is configured."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    takt_gap = round(bottleneck_time - takt_time, 2)
    takt_assessment = (
        f"**CRITICAL PACING DEFICIT (+{takt_gap} min over Takt)** — Line is starving downstream stations."
        if takt_gap > 0
        else f"**BALANCED FLOW ({abs(takt_gap)} min buffer under Takt)** — Line meets shift customer demand."
    )

    automations_md = "\n".join([f"- {item}" for item in active_automations])

    # Station-specific engineering root-cause & balancing guidance
    station_playbooks = {
        "Core Lamination Stacking": (
            "Manual step-lap stacking of grain-oriented electrical steel (CRGO) sheets introduces high ergonomic fatigue "
            "and micro-air-gap variance, increasing no-load core losses.",
            "Calibrate 6-axis cobot vacuum end-of-arm tooling (EOAT) pressure to 45 kPa, verify laser seam-alignment "
            "sensor offset (±0.15 mm), and execute parallel sub-pack staging.",
        ),
        "HV/LV Copper Coil Winding": (
            "Conductor tension fluctuations and manual transposition alignment slow down multi-layer disc winding "
            "and risk axial short-circuit forces under fault conditions.",
            "Engage closed-loop servo tensioner at 185 N ± 3 N, lock mandrel encoder index pulse, and verify optical "
            "pitch-angle feedback prior to pressboard spacer insertion.",
        ),
        "Vapor Phase Drying & Insulation": (
            "Vacuum autoclave thermal ramp and kerosene vapor condensation cycle time is governed by cellulose moisture "
            "extraction kinetics (< 0.5% residual moisture per IEC 60076).",
            "Pre-heat active part assembly to 65°C in staging buffer and optimize vacuum pump-down curve to recover "
            "4.5 minutes per batch without compromising insulation dielectric strength.",
        ),
        "Active Part Tanking & Fastening": (
            "Overhead bridge crane lowering into the transformer tank and multi-pass M20 flange torque sequencing create "
            "alignment dwell time.",
            "Deploy laser-guided crane spreader bars with DC electric nutrunners programmed to 320 Nm two-stage "
            "cross-pattern torque verification.",
        ),
        "High-Voltage Routine Testing": (
            "Induced overvoltage, turns-ratio (TTR), and lightning impulse test bay setup requires manual HV bushing lead "
            "changeovers.",
            "Implement automated pneumatic HV test busbar switching and pre-stage test leads during tanking.",
        ),
    }

    root_cause_text, corrective_text = station_playbooks.get(
        bottleneck_station,
        (
            "Manual cycle variance exceeds target shift Takt threshold.",
            "Rebalance operator work elements and verify PLC cycle timer interlocks.",
        ),
    )

    if inspection_passed:
        cv_section = f"""### 3. Vision Poka-Yoke Quality Gate Verification
- **PLC Interlock State:** `DB100.DBX0.0 = 1` (**PASS — Advance Conveyor to Next Cell**)
- **Measured Coil Alignment Deviation:** `{alignment_error_deg:.2f}°` (Compliant within `±1.50°` engineering tolerance)
- **Active Defects Detected:** `0`
- **Action:** No containment required. Archive high-resolution inspection frame with unit serial hash for IEC 60076 Factory Acceptance Test (FAT) traceability."""
    else:
        defect_lines = "\n".join(
            [
                f"- **[{d['Defect ID']}] {d['Component']}**: {d['Anomaly Type']} *(Severity: {d['Severity']}, Confidence: {d['Confidence']})*"
                for d in defect_summary
            ]
        )
        cv_section = f"""### 3. Vision Poka-Yoke Containment & Rework Protocol
- **PLC Interlock State:** `DB100.DBX0.0 = 0` (**REJECT — Line Halted & Diverted to Rework Spur**)
- **Measured Coil Alignment Deviation:** `{alignment_error_deg:.2f}°` (Tolerance limit: `±1.50°`)
- **Detected Anomalies ({len(defect_summary)}):**
{defect_lines}

#### Mandatory Containment Actions:
1. **Lockout / Tagout (LOTO):** Place upstream feeder conveyor in `MANUAL_HOLD` before technician entry into the vision cell light curtain.
2. **Mechanical Remediation:**
   - *If Winding Skew > 1.5°:* Release top axial clamping jack, re-index winding mandrel to `0.0° ± 0.2°`, and re-torque radial pressboard blocks.
   - *If Core Scratch Detected:* Measure gouge depth with eddy-current gauge; apply C-5 inorganic insulating varnish coating if interlaminar shorting resistance < 50 Ω.
   - *If Fastener Missing:* Install Grade 8.8 M20 zinc-plated hex bolt with Belleville spring washer; torque to `320 Nm` and mark with tamper-evident torque stripe.
3. **Re-Verification:** Re-trigger camera inspection capture (`TRIG_CAM_01`) and confirm `PLC SIGNAL: 1` before clearing fault latch."""

    return f"""# GE Vernova — Power Transformer Assembly Standard Operating Procedure (SOP)
**Document ID:** `GEV-PT-SOP-{datetime.now().strftime("%Y%m%d")}-04` | **Generated:** {timestamp} | **Engine:** SmartLine Offline Industrial Rules Engine

---

### 1. Live Production Line Telemetry Summary
- **Primary Line Bottleneck:** `{bottleneck_station}` (`{bottleneck_time:.1f} min/unit`)
- **Customer Takt Time Target:** `{takt_time:.2f} min/unit`
- **Flow Status:** {takt_assessment}
- **Current Line OEE:** `{current_oee:.1f}%`
- **Active Automation Retrofits:**
{automations_md}

---

### 2. Bottleneck Root-Cause & Cycle-Time Balancing Plan
- **Engineering Root Cause ({bottleneck_station}):** {root_cause_text}
- **Corrective Automation Action:** {corrective_text}
- **PLC & Safety Commissioning Check:** Verify Profinet heartbeat (`10 ms` cycle) between cell controller and line supervisory PLC; validate safety scanner warning/protective fields (`ISO 13855`) prior to raising line speed override above `75%`.

---

{cv_section}

---

### 4. Sign-Off & Shift Handover Verification
- [ ] Line Supervisor verified bottleneck cycle time trend on SCADA HMI
- [ ] Quality Engineer signed off on Vision Poka-Yoke PLC interlock log
- [ ] Robotics Technician confirmed EOAT & safety light-curtain diagnostics
"""


def generate_industrial_sop(
    api_key: Optional[str],
    bottleneck_station: str,
    bottleneck_time: float,
    takt_time: float,
    current_oee: float,
    throughput_gain_pct: float,
    active_automations: List[str],
    inspection_passed: bool,
    alignment_error_deg: float,
    defect_summary: List[Dict[str, str]],
) -> tuple[str, str]:
    """
    Generate an industrial SOP using Gemini 2.5 Flash if an API key is provided,
    falling back gracefully to the deterministic offline engineering engine.
    Returns (sop_markdown, source_mode_label).
    """
    cleaned_key = (api_key or "").strip()

    if cleaned_key:
        prompt = f"""Generate a formal GE Vernova Power Transformer Assembly SOP and Commissioning Root-Cause Analysis using these live line metrics:

- Primary Bottleneck Station: {bottleneck_station} ({bottleneck_time:.2f} min/unit)
- Target Shift Takt Time: {takt_time:.2f} min/unit
- Current Line OEE: {current_oee:.1f}%
- Projected Throughput Gain vs Baseline: +{throughput_gain_pct:.1f}%
- Active Automation Retrofits: {", ".join(active_automations)}
- Vision Poka-Yoke Inspection Result: {"PASS (PLC=1)" if inspection_passed else "REJECT (PLC=0)"}
- Measured Copper Coil Alignment Error: {alignment_error_deg:.2f} degrees (Tolerance: ±1.50 degrees)
- Detected Vision Defects: {defect_summary if defect_summary else "None"}

Structure the response into:
1. Executive Telemetry & Line Balance Assessment
2. Root-Cause Analysis & Robotics/PLC Optimization Steps for {bottleneck_station}
3. Vision Poka-Yoke Quality Containment & Rework Procedure (specific to the exact defects above)
4. Safety (LOTO / ISO 10218-2 Collaborative Robot) & Shift Handover Checklist"""

        try:
            client = genai.Client(api_key=cleaned_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.3,
                ),
            )
            if response and response.text:
                return response.text.strip(), "Gemini 2.5 Flash (Live Cloud AI)"
        except Exception as exc:
            offline_sop = _build_offline_sop(
                bottleneck_station=bottleneck_station,
                bottleneck_time=bottleneck_time,
                takt_time=takt_time,
                current_oee=current_oee,
                active_automations=active_automations,
                inspection_passed=inspection_passed,
                alignment_error_deg=alignment_error_deg,
                defect_summary=defect_summary,
            )
            fallback_notice = (
                f"> **Note:** Live Gemini API call encountered an error (`{type(exc).__name__}`). "
                "Seamlessly switched to the built-in SmartLine Deterministic Industrial Rules Engine.\n\n"
            )
            return fallback_notice + offline_sop, "Offline Industrial Engine (Fallback)"

    offline_sop = _build_offline_sop(
        bottleneck_station=bottleneck_station,
        bottleneck_time=bottleneck_time,
        takt_time=takt_time,
        current_oee=current_oee,
        active_automations=active_automations,
        inspection_passed=inspection_passed,
        alignment_error_deg=alignment_error_deg,
        defect_summary=defect_summary,
    )
    return offline_sop, "Offline Industrial Engine"
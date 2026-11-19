import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import streamlit as st

from src.data_generator import initialize_all_assets
from src.simulation import run_line_simulation
from src.sop_copilot import generate_industrial_sop
from src.vision_inspector import inspect_transformer_assembly


# =============================================================================
# PAGE CONFIGURATION & INDUSTRIAL COMMAND-CENTER STYLING
# =============================================================================
st.set_page_config(
    page_title="SmartLine | Power Transformer Assembly & Automation Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Clean Industrial Typography & Layout */
    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    /* GE Vernova Inspired Executive Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0B1F2A 0%, #10383A 60%, #006B48 100%);
        border-left: 6px solid #00A86B;
        border-radius: 10px;
        padding: 1.35rem 1.75rem;
        margin-bottom: 1.25rem;
        color: #F4F7F6;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.14);
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 0.25rem 0;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #CBE4DC;
        margin: 0;
    }
    /* KPI Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #D9E2EC;
        border-top: 4px solid #00875A;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 1px 4px rgba(16, 42, 67, 0.06);
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #334E68;
        font-size: 0.85rem;
    }
    div[data-testid="stMetricValue"] {
        color: #102A43;
        font-weight: 700;
    }
    /* PLC Signal Status Banners */
    .plc-pass {
        background-color: #E6F7F0;
        border: 1.5px solid #00875A;
        border-left: 6px solid #00875A;
        color: #004D33;
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        font-weight: 700;
        font-family: monospace;
        font-size: 1.02rem;
        margin-bottom: 0.9rem;
    }
    .plc-reject {
        background-color: #FDECEA;
        border: 1.5px solid #D93025;
        border-left: 6px solid #D93025;
        color: #7A120B;
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        font-weight: 700;
        font-family: monospace;
        font-size: 1.02rem;
        margin-bottom: 0.9rem;
    }
    /* Sidebar Badge */
    .status-badge-online {
        background-color: #E6F7F0;
        color: #006644;
        border: 1px solid #00875A;
        padding: 0.35rem 0.65rem;
        border-radius: 5px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.35rem;
    }
    .status-badge-offline {
        background-color: #F0F4F8;
        color: #334E68;
        border: 1px solid #BCCCDC;
        padding: 0.35rem 0.65rem;
        border-radius: 5px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# RUNTIME DATA & ASSET INITIALIZATION (ZERO-FAILURE STARTUP)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_runtime_assets():
    return initialize_all_assets()


df_historical_shifts, inspection_presets = load_runtime_assets()


# =============================================================================
# HEADER BANNER
# =============================================================================
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">⚡ SmartLine: Power Transformer Assembly & Automation Monitor</div>
        <div class="hero-subtitle">
            Discrete-Event Line Balancing (SimPy) • Deterministic CV Poka-Yoke Inspection (OpenCV) • GenAI Commissioning SOP Copilot
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("### 🏭 Shift Demand Parameters")
    target_output = st.slider(
        "Target Daily Output (Units/Shift)",
        min_value=6,
        max_value=16,
        value=12,
        step=1,
        help="Customer shift demand used to compute Takt Time.",
    )
    available_time_min = st.slider(
        "Available Shift Time (Minutes)",
        min_value=360,
        max_value=540,
        value=480,
        step=15,
        help="Net planned production minutes per shift (excluding scheduled breaks).",
    )

    st.divider()
    st.markdown("### 🤖 Automation Retrofit Toggles")
    cobot_core_stacking = st.toggle(
        "Deploy 6-Axis Cobot at Core Stacking",
        value=True,
        help="Automates CRGO lamination pick-and-place (-38% cycle time, -75% variance).",
    )
    servo_coil_winding = st.toggle(
        "Deploy Servo-Controlled Automated Winding Tensioner",
        value=True,
        help="Closed-loop HV/LV copper conductor tensioning (-30% cycle time).",
    )
    vision_poka_yoke = st.toggle(
        "Deploy Inline Vision Poka-Yoke Gate",
        value=True,
        help="Automated optical defect gating boosts line Quality factor from 89.0% to 99.2%.",
    )

    st.divider()
    st.markdown("### 🧠 GenAI Copilot Configuration")
    gemini_api_key = st.text_input(
        "Gemini API Key (Optional)",
        type="password",
        placeholder="AIzaSy...",
        help="Optional. Leave blank to run the built-in Offline Deterministic Industrial Engine.",
    )
    if gemini_api_key.strip():
        st.markdown('<span class="status-badge-online">● Gemini 2.5 Flash Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge-offline">● Offline Industrial Engine Ready</span>', unsafe_allow_html=True)


# =============================================================================
# RUN CORE SIMULATION & DEFAULT VISION STATE
# =============================================================================
sim_res = run_line_simulation(
    target_output=target_output,
    available_time_min=available_time_min,
    cobot_core_stacking=cobot_core_stacking,
    servo_coil_winding=servo_coil_winding,
    vision_poka_yoke=vision_poka_yoke,
)


# =============================================================================
# MAIN DASHBOARD TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs(
    [
        "📊 1. Lean Cycle-Time & Cobot ROI Simulator",
        "🔍 2. Vision Poka-Yoke & Alignment Inspector",
        "📋 3. GenAI Auto-SOP & Commissioning Copilot",
    ]
)


# -----------------------------------------------------------------------------
# TAB 1: LEAN CYCLE-TIME & COBOT ROI SIMULATOR
# -----------------------------------------------------------------------------
with tab1:
    # 4 Top KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric(
            label="Target Takt Time",
            value=f"{sim_res.takt_time:.1f} min",
            delta=f"{target_output} units / {available_time_min} min",
            delta_color="off",
        )

    with kpi2:
        bn_delta = round(sim_res.current_bottleneck_time - sim_res.baseline_bottleneck_time, 1)
        st.metric(
            label="Bottleneck Cycle Time",
            value=f"{sim_res.current_bottleneck_time:.1f} min",
            delta=f"{bn_delta:+.1f} min vs manual ({sim_res.current_bottleneck_station.split()[0]})",
            delta_color="inverse",
        )

    with kpi3:
        oee_delta = round(sim_res.current_oee - sim_res.baseline_oee, 1)
        st.metric(
            label="Line OEE %",
            value=f"{sim_res.current_oee:.1f}%",
            delta=f"{oee_delta:+.1f}% vs baseline ({sim_res.baseline_oee:.1f}%)",
        )

    with kpi4:
        st.metric(
            label="Throughput Gain %",
            value=f"+{sim_res.throughput_gain_pct:.1f}%",
            delta=f"{sim_res.current_units_per_shift} vs {sim_res.baseline_units_per_shift} units/shift",
        )

    st.markdown("#### Assembly Line Cycle-Time Balance vs. Takt Threshold")

    col_chart1, col_chart2 = st.columns([1.35, 1.0])

    with col_chart1:
        df_st = sim_res.station_df
        bar_colors = [
            "#D93025" if c > sim_res.takt_time else "#00875A"
            for c in df_st["Current Cycle (min)"]
        ]

        fig_takt = go.Figure()
        fig_takt.add_trace(
            go.Bar(
                x=df_st["Station Name"],
                y=df_st["Current Cycle (min)"],
                marker_color=bar_colors,
                text=[f"{v:.1f}m" for v in df_st["Current Cycle (min)"]],
                textposition="outside",
                name="Active Cycle Time (min)",
            )
        )
        fig_takt.add_hline(
            y=sim_res.takt_time,
            line_dash="dash",
            line_color="#D93025",
            line_width=2.5,
            annotation_text=f"Takt Time Limit: {sim_res.takt_time:.1f} min",
            annotation_position="top right",
            annotation_font_color="#D93025",
        )
        fig_takt.update_layout(
            title="Current Station Cycle Times vs. Customer Takt Time",
            yaxis_title="Cycle Time (Minutes)",
            yaxis_range=[0, max(65, float(df_st["Baseline Cycle (min)"].max()) + 10)],
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=50, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_takt, use_container_width=True)

    with col_chart2:
        fig_comp = go.Figure()
        fig_comp.add_trace(
            go.Bar(
                name="Baseline (Manual)",
                x=df_st["Station ID"],
                y=df_st["Baseline Cycle (min)"],
                marker_color="#829AB1",
            )
        )
        fig_comp.add_trace(
            go.Bar(
                name="Current Configuration",
                x=df_st["Station ID"],
                y=df_st["Current Cycle (min)"],
                marker_color="#00875A",
            )
        )
        fig_comp.update_layout(
            title="Before vs. After Automation Retrofit",
            barmode="group",
            yaxis_title="Cycle Time (Minutes)",
            yaxis_range=[0, max(65, float(df_st["Baseline Cycle (min)"].max()) + 10)],
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("#### Station-by-Station Discrete-Event Simulation Summary")
    st.dataframe(
        df_st,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("View Historical 30-Shift Baseline Telemetry Log (Auto-Generated CSV)"):
        st.dataframe(df_historical_shifts.head(25), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# TAB 2: VISION POKA-YOKE & ALIGNMENT INSPECTOR
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("#### Inline Optical Inspection & PLC Interlock Verification")
    ctrl_col1, ctrl_col2 = st.columns([1.2, 1.0])

    with ctrl_col1:
        selected_preset = st.selectbox(
            "Select Synthetic Transformer Inspection Preset",
            options=list(inspection_presets.keys()),
            index=1,  # Default to Winding Skew Defect so recruiter immediately sees CV detection in action
        )

    with ctrl_col2:
        uploaded_file = st.file_uploader(
            "Or Upload Custom Assembly Image (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            help="Overrides preset selector with your custom uploaded image.",
        )

    # Load BGR image for OpenCV pipeline
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        raw_rgb = np.array(pil_img)
        input_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
        active_source_name = f"Uploaded: {uploaded_file.name}"
    else:
        preset_path = inspection_presets[selected_preset]
        input_bgr = cv2.imread(str(preset_path))
        raw_rgb = cv2.cvtColor(cv2.resize(input_bgr, (800, 600)), cv2.COLOR_BGR2RGB)
        active_source_name = selected_preset

    cv_res = inspect_transformer_assembly(input_bgr)

    # Store latest inspection state in session_state for Tab 3 SOP Copilot
    st.session_state["latest_cv_result"] = cv_res
    st.session_state["latest_cv_source"] = active_source_name

    # PLC Status Banner
    if cv_res.passed:
        st.markdown(
            f'<div class="plc-pass">🟢 {cv_res.plc_banner_text} &nbsp;|&nbsp; SOURCE: {active_source_name}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="plc-reject">🔴 {cv_res.plc_banner_text} &nbsp;|&nbsp; SOURCE: {active_source_name}</div>',
            unsafe_allow_html=True,
        )

    # Vision KPI Strip
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        st.metric(
            label="Coil Alignment Error",
            value=f"{cv_res.alignment_error_deg:.2f}°",
            delta=cv_res.angle_status,
            delta_color="normal" if cv_res.alignment_error_deg <= 1.5 else "inverse",
        )
    with v_col2:
        st.metric(
            label="Detected Anomalies",
            value=f"{cv_res.defect_count}",
            delta="Zero-defect threshold" if cv_res.defect_count == 0 else "Containment required",
            delta_color="normal" if cv_res.defect_count == 0 else "inverse",
        )
    with v_col3:
        st.metric(
            label="PLC Register State",
            value=f"DB100.DBX0.0 = {cv_res.plc_signal_code}",
            delta="Conveyor Enabled" if cv_res.plc_signal_code == 1 else "Line Interlock Halted",
            delta_color="normal" if cv_res.plc_signal_code == 1 else "inverse",
        )

    # Side-by-Side Raw vs Annotated Visual Comparison
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("**Raw Overhead Camera Feed (800×600)**")
        st.image(raw_rgb, use_container_width=True)
    with img_col2:
        st.markdown("**CV Annotated Poka-Yoke Output**")
        st.image(cv_res.annotated_rgb, use_container_width=True)

    st.markdown("#### Automated Defect & Metrology Log")
    if cv_res.defect_summary:
        st.dataframe(pd.DataFrame(cv_res.defect_summary), use_container_width=True, hide_index=True)
    else:
        st.success("All geometric winding tolerances (±1.5°), core lamination surfaces, and M20 clamping bolts verified PASS.")


# -----------------------------------------------------------------------------
# TAB 3: GENAI AUTO-SOP & COMMISSIONING COPILOT
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("#### Automated Standard Operating Procedure (SOP) & Commissioning Copilot")
    st.caption(
        "Synthesizes live line balancing telemetry (Tab 1) and Computer Vision Poka-Yoke defect diagnostics (Tab 2) "
        "into an actionable GE Vernova shop-floor SOP."
    )

    latest_cv = st.session_state.get("latest_cv_result", cv_res)

    ctx1, ctx2, ctx3, ctx4 = st.columns(4)
    ctx1.info(f"**Bottleneck:** {sim_res.current_bottleneck_station} ({sim_res.current_bottleneck_time:.1f}m)")
    ctx2.info(f"**Takt Target:** {sim_res.takt_time:.1f}m | **OEE:** {sim_res.current_oee:.1f}%")
    ctx3.info(f"**Vision Gate:** {'PASS (PLC=1)' if latest_cv.passed else 'REJECT (PLC=0)'}")
    ctx4.info(f"**Coil Skew:** {latest_cv.alignment_error_deg:.2f}° ({latest_cv.defect_count} defects)")

    if "generated_sop_md" not in st.session_state:
        initial_sop, initial_mode = generate_industrial_sop(
            api_key=gemini_api_key,
            bottleneck_station=sim_res.current_bottleneck_station,
            bottleneck_time=sim_res.current_bottleneck_time,
            takt_time=sim_res.takt_time,
            current_oee=sim_res.current_oee,
            throughput_gain_pct=sim_res.throughput_gain_pct,
            active_automations=sim_res.active_automations,
            inspection_passed=latest_cv.passed,
            alignment_error_deg=latest_cv.alignment_error_deg,
            defect_summary=latest_cv.defect_summary,
        )
        st.session_state["generated_sop_md"] = initial_sop
        st.session_state["generated_sop_mode"] = initial_mode

    btn_col1, btn_col2 = st.columns([1.2, 1.0])
    with btn_col1:
        if st.button("⚡ Generate Industrial SOP & Troubleshooting Guide", type="primary", use_container_width=True):
            with st.spinner("Synthesizing engineering SOP from live line & vision telemetry..."):
                sop_md, mode_used = generate_industrial_sop(
                    api_key=gemini_api_key,
                    bottleneck_station=sim_res.current_bottleneck_station,
                    bottleneck_time=sim_res.current_bottleneck_time,
                    takt_time=sim_res.takt_time,
                    current_oee=sim_res.current_oee,
                    throughput_gain_pct=sim_res.throughput_gain_pct,
                    active_automations=sim_res.active_automations,
                    inspection_passed=latest_cv.passed,
                    alignment_error_deg=latest_cv.alignment_error_deg,
                    defect_summary=latest_cv.defect_summary,
                )
                st.session_state["generated_sop_md"] = sop_md
                st.session_state["generated_sop_mode"] = mode_used

    with btn_col2:
        st.download_button(
            label="📥 Export SOP as Markdown (.md)",
            data=st.session_state["generated_sop_md"],
            file_name="GE_Vernova_SmartLine_SOP.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.caption(f"Active Synthesis Engine: **{st.session_state['generated_sop_mode']}**")
    st.markdown("---")
    st.markdown(st.session_state["generated_sop_md"])
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import simpy


@dataclass
class SimulationResult:
    takt_time: float
    target_output: int
    available_time_min: float
    baseline_bottleneck_station: str
    baseline_bottleneck_time: float
    current_bottleneck_station: str
    current_bottleneck_time: float
    baseline_oee: float
    current_oee: float
    baseline_units_per_shift: int
    current_units_per_shift: int
    throughput_gain_pct: float
    idle_time_saved_min: float
    active_automations: List[str]
    station_df: pd.DataFrame


def _run_simpy_shift(
    station_means: List[float],
    station_stds: List[float],
    available_time_min: float,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Discrete-event simulation (SimPy) of a 5-stage serial transformer assembly line
    with finite inter-station buffers (capacity=1) to capture blocking/starving behavior.
    """
    rng = np.random.default_rng(seed=seed)
    env = simpy.Environment()

    num_stations = len(station_means)
    stations = [simpy.Resource(env, capacity=1) for _ in range(num_stations)]
    busy_time = [0.0 for _ in range(num_stations)]
    completed_units = 0

    def transformer_unit(unit_id: int):
        nonlocal completed_units
        for idx in range(num_stations):
            with stations[idx].request() as req:
                yield req
                proc_duration = max(
                    12.0,
                    float(rng.normal(station_means[idx], station_stds[idx])),
                )
                start_t = env.now
                yield env.timeout(proc_duration)
                end_t = min(env.now, available_time_min)
                if end_t > start_t:
                    busy_time[idx] += end_t - start_t
        if env.now <= available_time_min:
            completed_units += 1

    def line_feeder():
        unit_id = 1
        while env.now < available_time_min:
            env.process(transformer_unit(unit_id))
            unit_id += 1
            # Release new core kit paced slightly faster than bottleneck to test line balance
            inter_arrival = max(15.0, min(station_means) * 0.85)
            yield env.timeout(inter_arrival)

    env.process(line_feeder())
    env.run(until=available_time_min)

    utilizations = [min(100.0, (bt / available_time_min) * 100.0) for bt in busy_time]
    total_idle_min = sum(max(0.0, available_time_min - bt) for bt in busy_time)

    return {
        "completed_units": completed_units,
        "utilizations": utilizations,
        "total_idle_min": total_idle_min,
    }


def run_line_simulation(
    target_output: int,
    available_time_min: float,
    cobot_core_stacking: bool = False,
    servo_coil_winding: bool = False,
    vision_poka_yoke: bool = False,
) -> SimulationResult:
    """
    Evaluate baseline manual line vs. automated retrofit configuration using SimPy + Lean metrics.
    """
    takt_time = round(available_time_min / max(1, target_output), 2)

    base_stations = [
        {
            "id": "ST-01",
            "name": "Core Lamination Stacking",
            "base_mean": 52.0,
            "base_std": 5.8,
        },
        {
            "id": "ST-02",
            "name": "HV/LV Copper Coil Winding",
            "base_mean": 48.0,
            "base_std": 4.5,
        },
        {
            "id": "ST-03",
            "name": "Vapor Phase Drying & Insulation",
            "base_mean": 40.0,
            "base_std": 1.8,
        },
        {
            "id": "ST-04",
            "name": "Active Part Tanking & Fastening",
            "base_mean": 46.0,
            "base_std": 4.2,
        },
        {
            "id": "ST-05",
            "name": "High-Voltage Routine Testing",
            "base_mean": 35.0,
            "base_std": 2.4,
        },
    ]

    # Configure automated cycle times & variances
    current_means = []
    current_stds = []
    modes = []

    for st in base_stations:
        mean_val = st["base_mean"]
        std_val = st["base_std"]
        mode_label = "Manual Standard"

        if st["id"] == "ST-01" and cobot_core_stacking:
            # ~38% cycle time reduction + tight robotic repeatability
            mean_val = round(mean_val * 0.62, 2)
            std_val = round(std_val * 0.25, 2)
            mode_label = "6-Axis Cobot Cell"

        if st["id"] == "ST-02" and servo_coil_winding:
            # ~30% cycle time reduction + closed-loop tension control
            mean_val = round(mean_val * 0.70, 2)
            std_val = round(std_val * 0.30, 2)
            mode_label = "Servo Tensioner + Auto-Guide"

        if st["id"] == "ST-05" and vision_poka_yoke:
            # Inline Poka-Yoke prevents downstream HV test rework delays
            mean_val = round(mean_val * 0.90, 2)
            std_val = round(std_val * 0.50, 2)
            mode_label = "Vision-Verified Fast Track"

        current_means.append(mean_val)
        current_stds.append(std_val)
        modes.append(mode_label)

    base_means = [s["base_mean"] for s in base_stations]
    base_stds = [s["base_std"] for s in base_stations]

    # Run discrete-event simulations
    base_sim = _run_simpy_shift(base_means, base_stds, available_time_min, seed=42)
    curr_sim = _run_simpy_shift(current_means, current_stds, available_time_min, seed=42)

    # Identify bottlenecks
    base_bn_idx = int(np.argmax(base_means))
    curr_bn_idx = int(np.argmax(current_means))

    base_bn_station = base_stations[base_bn_idx]["name"]
    base_bn_time = base_means[base_bn_idx]

    curr_bn_station = base_stations[curr_bn_idx]["name"]
    curr_bn_time = current_means[curr_bn_idx]

    # OEE = Availability x Performance x Quality
    # Baseline: Availability = 88.5%, Performance = min(1.0, Takt / Bottleneck), Quality = 89.0%
    base_availability = 0.885
    base_performance = min(1.0, takt_time / base_bn_time)
    base_quality = 0.890
    baseline_oee = round(base_availability * base_performance * base_quality * 100.0, 1)

    # Automated OEE improvements
    curr_availability = 0.885 + (0.035 if cobot_core_stacking else 0.0) + (0.025 if servo_coil_winding else 0.0)
    curr_performance = min(1.0, takt_time / curr_bn_time)
    curr_quality = 0.992 if vision_poka_yoke else (0.935 if servo_coil_winding else 0.890)
    current_oee = round(min(99.5, curr_availability * curr_performance * curr_quality * 100.0), 1)

    # Effective shift throughput capacity based on bottleneck cycle time & OEE
    base_units = max(1, int(np.floor((available_time_min / base_bn_time) * (base_quality + 0.05))))
    curr_units = max(1, int(np.floor((available_time_min / curr_bn_time) * (curr_quality + 0.005))))

    # Ensure monotonic positive reflection when bottlenecks drop
    raw_rate_gain = ((base_bn_time - curr_bn_time) / base_bn_time) * 100.0
    quality_gain = ((curr_quality - base_quality) / base_quality) * 100.0
    throughput_gain_pct = round(max(0.0, raw_rate_gain + quality_gain * 0.45), 1)

    idle_time_saved_min = round(max(0.0, base_sim["total_idle_min"] - curr_sim["total_idle_min"]), 1)

    rows = []
    for idx, st in enumerate(base_stations):
        delta_takt = round(current_means[idx] - takt_time, 2)
        status = "CRITICAL BOTTLENECK" if current_means[idx] > takt_time else "WITHIN TAKT"
        rows.append(
            {
                "Station ID": st["id"],
                "Station Name": st["name"],
                "Operating Mode": modes[idx],
                "Baseline Cycle (min)": base_means[idx],
                "Current Cycle (min)": current_means[idx],
                "Std Dev (±min)": current_stds[idx],
                "Takt Target (min)": takt_time,
                "Variance vs Takt (min)": delta_takt,
                "Line Utilization (%)": round(curr_sim["utilizations"][idx], 1),
                "Takt Status": status,
            }
        )

    station_df = pd.DataFrame(rows)

    active_automations = []
    if cobot_core_stacking:
        active_automations.append("6-Axis Cobot Cell (Core Lamination Stacking)")
    if servo_coil_winding:
        active_automations.append("Servo-Controlled Automated Winding Tensioner (HV/LV Coils)")
    if vision_poka_yoke:
        active_automations.append("Inline Vision Poka-Yoke Quality Gate")
    if not active_automations:
        active_automations.append("None (100% Manual Baseline Line)")

    return SimulationResult(
        takt_time=takt_time,
        target_output=target_output,
        available_time_min=available_time_min,
        baseline_bottleneck_station=base_bn_station,
        baseline_bottleneck_time=base_bn_time,
        current_bottleneck_station=curr_bn_station,
        current_bottleneck_time=curr_bn_time,
        baseline_oee=baseline_oee,
        current_oee=current_oee,
        baseline_units_per_shift=base_units,
        current_units_per_shift=curr_units,
        throughput_gain_pct=throughput_gain_pct,
        idle_time_saved_min=idle_time_saved_min,
        active_automations=active_automations,
        station_df=station_df,
    )
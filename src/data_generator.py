import os
from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets" / "inspections"


STATIONS_CONFIG = [
    {
        "station_id": "ST-01",
        "station_name": "Core Lamination Stacking",
        "baseline_mean_min": 52.0,
        "baseline_std_min": 5.8,
        "manual_operators": 3,
        "process_type": "Mechanical Assembly",
    },
    {
        "station_id": "ST-02",
        "station_name": "HV/LV Copper Coil Winding",
        "baseline_mean_min": 48.0,
        "baseline_std_min": 4.5,
        "manual_operators": 2,
        "process_type": "Precision Winding",
    },
    {
        "station_id": "ST-03",
        "station_name": "Vapor Phase Drying & Insulation",
        "baseline_mean_min": 40.0,
        "baseline_std_min": 1.8,
        "manual_operators": 1,
        "process_type": "Thermal Batch",
    },
    {
        "station_id": "ST-04",
        "station_name": "Active Part Tanking & Fastening",
        "baseline_mean_min": 46.0,
        "baseline_std_min": 4.2,
        "manual_operators": 3,
        "process_type": "Heavy Crane & Torque",
    },
    {
        "station_id": "ST-05",
        "station_name": "High-Voltage Routine Testing",
        "baseline_mean_min": 35.0,
        "baseline_std_min": 2.4,
        "manual_operators": 2,
        "process_type": "HV Impulse & Ratio",
    },
]


PRESET_DEFINITIONS: Dict[str, str] = {
    "Preset 1: Nominal Assembly (PASS)": "preset_1_nominal_pass.png",
    "Preset 2: Winding Skew Defect (REJECT)": "preset_2_winding_skew_reject.png",
    "Preset 3: Core Scratch & Missing Fastener (REJECT)": "preset_3_scratch_fastener_reject.png",
}


def ensure_directories() -> None:
    """Create runtime data and synthetic image asset directories if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def generate_shift_logs(csv_path: Path = DATA_DIR / "shift_production_logs.csv", force: bool = False) -> pd.DataFrame:
    """
    Generate 30 historical shifts of telemetry across the 5-stage power transformer assembly line.
    Automatically writes to CSV if missing.
    """
    ensure_directories()
    if csv_path.exists() and not force:
        return pd.read_csv(csv_path)

    rng = np.random.default_rng(seed=42)
    records = []

    for shift_num in range(1, 31):
        shift_label = f"SHIFT-{shift_num:03d}"
        shift_type = "Day" if shift_num % 2 == 1 else "Night"
        shift_penalty = 0.0 if shift_type == "Day" else 1.4

        for st in STATIONS_CONFIG:
            cycle_time = max(
                20.0,
                float(rng.normal(st["baseline_mean_min"] + shift_penalty, st["baseline_std_min"])),
            )
            rework_flag = int(
                rng.random() < (0.14 if st["station_id"] in ("ST-01", "ST-02") else 0.05)
            )
            downtime_min = round(float(rng.exponential(2.5)) if rng.random() < 0.35 else 0.0, 2)
            units_completed = max(6, int(480.0 // cycle_time))

            records.append(
                {
                    "shift_id": shift_label,
                    "shift_type": shift_type,
                    "station_id": st["station_id"],
                    "station_name": st["station_name"],
                    "process_type": st["process_type"],
                    "cycle_time_min": round(cycle_time, 2),
                    "unplanned_downtime_min": downtime_min,
                    "rework_triggered": rework_flag,
                    "units_completed": units_completed,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    return df


def _create_base_transformer_canvas(
    skew_deg: float = 0.2,
    include_scratch: bool = False,
    missing_bottom_right_bolt: bool = False,
) -> np.ndarray:
    """
    Synthesize an 800x600 top-down engineering inspection image of a 3-limb power transformer
    active core and HV/LV copper coil assembly in BGR format.
    """
    width, height = 800, 600
    rng = np.random.default_rng(seed=108)

    # 1. Industrial brushed steel inspection fixture background
    bg = np.full((height, width, 3), (44, 48, 52), dtype=np.uint8)
    grain = rng.integers(-8, 9, size=(height, width, 1), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    # Outer mounting bed frame
    cv2.rectangle(bg, (45, 35), (755, 565), (72, 78, 84), thickness=3)
    cv2.rectangle(bg, (55, 45), (745, 555), (32, 36, 40), thickness=-1)

    # 2. Silicon-steel laminated core yoke & 3 vertical limbs (silver-slate metallic)
    core_color = (158, 166, 174)
    core_border = (112, 120, 128)
    # Top and bottom core yokes
    cv2.rectangle(bg, (110, 75), (690, 145), core_color, thickness=-1)
    cv2.rectangle(bg, (110, 455), (690, 525), core_color, thickness=-1)
    cv2.rectangle(bg, (110, 75), (690, 145), core_border, thickness=2)
    cv2.rectangle(bg, (110, 455), (690, 525), core_border, thickness=2)

    # Lamination grain lines across top and bottom yokes
    for y_line in range(82, 142, 6):
        cv2.line(bg, (115, y_line), (685, y_line), (132, 140, 148), 1)
    for y_line in range(462, 522, 6):
        cv2.line(bg, (115, y_line), (685, y_line), (132, 140, 148), 1)

    # Three vertical core limbs (Phase A, Phase B, Phase C)
    limb_centers_x = [215, 400, 585]
    for cx in limb_centers_x:
        cv2.rectangle(bg, (cx - 38, 145), (cx + 38, 455), (145, 152, 160), thickness=-1)
        for lx in range(cx - 34, cx + 36, 5):
            cv2.line(bg, (lx, 146), (lx, 454), (120, 128, 136), 1)

    # 3. Wound HV/LV Copper Coils on each limb (drawn on rotatable sub-layer for tilt simulation)
    coil_layer = np.zeros_like(bg)
    coil_mask = np.zeros((height, width), dtype=np.uint8)

    for idx, cx in enumerate(limb_centers_x):
        # Tilt center coil (Phase B) or all coils slightly depending on defect mode
        angle = skew_deg if (abs(skew_deg) > 1.0 and idx == 1) else (skew_deg * 0.25 if abs(skew_deg) <= 1.0 else 0.3)

        sub = np.zeros_like(bg)
        sub_m = np.zeros((height, width), dtype=np.uint8)

        x1, y1, x2, y2 = cx - 68, 175, cx + 68, 425
        # Copper base block (BGR: warm electrolytic copper)
        cv2.rectangle(sub, (x1, y1), (x2, y2), (46, 118, 205), thickness=-1)
        cv2.rectangle(sub_m, (x1, y1), (x2, y2), 255, thickness=-1)

        # Individual horizontal copper conductor turns (high contrast for Hough line inspection)
        for turn_y in range(y1 + 8, y2 - 4, 10):
            cv2.line(sub, (x1 + 3, turn_y), (x2 - 3, turn_y), (22, 65, 135), thickness=2)
            cv2.line(sub, (x1 + 3, turn_y + 3), (x2 - 3, turn_y + 3), (95, 168, 242), thickness=1)

        # Vertical insulation pressboard cooling strips
        for strip_x in [cx - 42, cx, cx + 42]:
            cv2.rectangle(sub, (strip_x - 4, y1 + 2), (strip_x + 4, y2 - 2), (185, 212, 228), thickness=-1)

        cv2.rectangle(sub, (x1, y1), (x2, y2), (24, 58, 110), thickness=2)

        # Rotate coil by `angle` degrees around limb centroid
        rot_mat = cv2.getRotationMatrix2D((float(cx), 300.0), -angle, 1.0)
        sub_rot = cv2.warpAffine(sub, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
        mask_rot = cv2.warpAffine(sub_m, rot_mat, (width, height), flags=cv2.INTER_NEAREST)

        coil_layer = np.where(mask_rot[:, :, None] > 0, sub_rot, coil_layer)
        coil_mask = cv2.bitwise_or(coil_mask, mask_rot)

    bg = np.where(coil_mask[:, :, None] > 0, coil_layer, bg)

    # 4. Four Corner Clamping Bolts on the Top & Bottom Core Yokes
    bolt_positions = [
        (145, 110),  # Top-Left
        (655, 110),  # Top-Right
        (145, 490),  # Bottom-Left
        (655, 490),  # Bottom-Right
    ]

    for b_idx, (bx, by) in enumerate(bolt_positions):
        if missing_bottom_right_bolt and b_idx == 3:
            # Empty dark threaded bore (missing fastener anomaly)
            cv2.circle(bg, (bx, by), 15, (18, 20, 22), thickness=-1)
            cv2.circle(bg, (bx, by), 15, (65, 70, 75), thickness=2)
            cv2.circle(bg, (bx, by), 9, (10, 12, 14), thickness=-1)
        else:
            # Intact hex flange clamping bolt
            cv2.circle(bg, (bx, by), 16, (95, 102, 110), thickness=-1)
            cv2.circle(bg, (bx, by), 16, (45, 50, 55), thickness=2)
            cv2.circle(bg, (bx, by), 10, (205, 212, 218), thickness=-1)
            # Hex head inner facet
            hex_pts = []
            for k in range(6):
                theta = np.deg2rad(60 * k + 15)
                hex_pts.append([int(bx + 7 * np.cos(theta)), int(by + 7 * np.sin(theta))])
            cv2.polylines(bg, [np.array(hex_pts, dtype=np.int32)], isClosed=True, color=(80, 86, 92), thickness=2)

    # 5. Optional Deep Surface Scratch across Top Silicon-Steel Yoke
    if include_scratch:
        # Jagged high-contrast dark gouge across top core lamination steel
        scratch_pts = np.array(
            [
                [265, 86],
                [310, 98],
                [358, 112],
                [412, 121],
                [468, 134],
            ],
            dtype=np.int32,
        )
        cv2.polylines(bg, [scratch_pts], isClosed=False, color=(20, 22, 25), thickness=5)
        cv2.polylines(bg, [scratch_pts + np.array([0, 3])], isClosed=False, color=(225, 232, 238), thickness=1)

    return bg


def ensure_inspection_presets(force: bool = False) -> Dict[str, Path]:
    """
    Generate the 3 synthetic transformer inspection presets on disk if they do not exist.
    Returns mapping of preset display names to file paths.
    """
    ensure_directories()

    specs = {
        "Preset 1: Nominal Assembly (PASS)": (
            ASSETS_DIR / PRESET_DEFINITIONS["Preset 1: Nominal Assembly (PASS)"],
            dict(skew_deg=0.2, include_scratch=False, missing_bottom_right_bolt=False),
        ),
        "Preset 2: Winding Skew Defect (REJECT)": (
            ASSETS_DIR / PRESET_DEFINITIONS["Preset 2: Winding Skew Defect (REJECT)"],
            dict(skew_deg=4.8, include_scratch=False, missing_bottom_right_bolt=False),
        ),
        "Preset 3: Core Scratch & Missing Fastener (REJECT)": (
            ASSETS_DIR / PRESET_DEFINITIONS["Preset 3: Core Scratch & Missing Fastener (REJECT)"],
            dict(skew_deg=0.3, include_scratch=True, missing_bottom_right_bolt=True),
        ),
    }

    resolved: Dict[str, Path] = {}
    for label, (filepath, kwargs) in specs.items():
        if not filepath.exists() or force:
            img_bgr = _create_base_transformer_canvas(**kwargs)
            cv2.imwrite(str(filepath), img_bgr)
        resolved[label] = filepath

    return resolved


def initialize_all_assets() -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """Entrypoint called at startup to ensure CSV logs and test images are ready."""
    df_logs = generate_shift_logs()
    presets = ensure_inspection_presets()
    return df_logs, presets
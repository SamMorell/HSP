"""
hsp_core.py
Framework-agnostic core logic for Hansen Solubility Parameter (HSP) distance ranking.

Designed to be imported by Streamlit/Flask/FastAPI wrappers.

Expected input data columns (case-insensitive; many aliases accepted):
- name (or Solvent, Material, Solvent/Name)
- dD (dispersion)
- dP (polar)
- dH (hydrogen bonding)

Outputs a ranked table with:
- rank
- target_name
- name
- dD, dP, dH
- delta_d, delta_p, delta_h
- distance
- compatibility  (Excellent/Good/Poor)
"""

from __future__ import annotations

import io
from typing import Dict, Union, Any

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment


REQUIRED_COLS = ["name", "dD", "dP", "dH"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common header variations to canonical: name, dD, dP, dH"""
    colmap: Dict[str, str] = {}

    for c in df.columns:
        k = str(c).strip().lower()

        # Name
        if k in {"name", "solvent", "solvent/name", "solvent / name", "material", "material name", "material_name"}:
            colmap[c] = "name"
            continue

        # dD (dispersion)
        if k in {"dd", "δd", "d_d", "dispersion", "delta d", "delta_d", "d (dispersion)", "δd (dispersion)"}:
            colmap[c] = "dD"
            continue

        # dP (polar)
        if k in {"dp", "δp", "d_p", "polar", "delta p", "delta_p", "p (polar)", "δp (polar)"}:
            colmap[c] = "dP"
            continue

        # dH (hydrogen bonding)
        if k in {
            "dh",
            "δh",
            "d_h",
            "hbond",
            "h-bond",
            "h-bonding",
            "hbonding",
            "hydrogen bonding",
            "hydrogen-bonding",
            "delta h",
            "delta_h",
            "h (h-bonding)",
            "δh (h-bonding)",
        }:
            colmap[c] = "dH"
            continue

    out = df.rename(columns=colmap).copy()
    return out


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    df = df.copy()
    df["name"] = df["name"].astype(str).str.strip()

    for c in ["dD", "dP", "dH"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["dD", "dP", "dH"]).reset_index(drop=True)
    return df


def load_input_data(fileobj_or_path: Union[str, io.BytesIO, Any]) -> pd.DataFrame:
    """Read an Excel file-like object (Streamlit uploader) or path into a standardized DataFrame."""
    df = pd.read_excel(fileobj_or_path)
    df = _normalize_columns(df)
    df = _validate_and_clean(df)
    return df


def hansen_distance(
    dD: float, dP: float, dH: float,
    target_dD: float, target_dP: float, target_dH: float
) -> float:
    """Hansen distance Ra using a common form."""
    return float(np.sqrt(4.0 * (dD - target_dD) ** 2 + (dP - target_dP) ** 2 + (dH - target_dH) ** 2))


def _compatibility_bucket(distance: float) -> str:
    if distance <= 5:
        return "Excellent"
    if distance <= 10:
        return "Good"
    return "Poor"



def calculate_hsp_distances(df: 'pd.DataFrame', target=None, round_to: int = 2, **kwargs) -> 'pd.DataFrame':
    """
    Compute ranked Hansen distance vs a user-defined target dict.

    target = {"name": "Hydrocarbon Resin", "dD": 17.0, "dP": 9.8, "dH": 9.4}
    """
    df = _normalize_columns(df)
    df = _validate_and_clean(df)

    if target is None:
        raise ValueError("target is required. Expected dict: {'name':..., 'dD':..., 'dP':..., 'dH':...}")

    target_name = str(target.get("name", "Target")).strip() or "Target"
    td = float(target["dD"])
    tp = float(target["dP"])
    th = float(target["dH"])

    out = df.copy()
    out["delta_d"] = (out["dD"] - td).abs()
    out["delta_p"] = (out["dP"] - tp).abs()
    out["delta_h"] = (out["dH"] - th).abs()

    out["distance"] = np.sqrt(4.0 * (out["dD"] - td) ** 2 + (out["dP"] - tp) ** 2 + (out["dH"] - th) ** 2)
    out["compatibility"] = out["distance"].apply(_compatibility_bucket)

    for c in ["delta_d", "delta_p", "delta_h", "distance"]:
        out[c] = out[c].round(round_to)

    out = out.sort_values("distance", ascending=True).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    out.insert(1, "target_name", target_name)

    return out

KEEP_HEADER_CASE = {"dD", "dP", "dH"}


def _pretty_header(name: str) -> str:
    """Format a column header for export/UI: underscores -> spaces, Title Case; keep dD/dP/dH as-is."""
    if name in KEEP_HEADER_CASE:
        return name
    return str(name).replace("_", " ").strip().title()


def export_results_excel(df: pd.DataFrame, sheet_name: str = "Results") -> bytes:
    """Export results DataFrame to XLSX bytes with formatted, left-aligned headers."""
    # Make a copy so we don't mutate callers
    out_df = df.copy()
    out_df.columns = [_pretty_header(c) for c in out_df.columns]

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name=sheet_name)

        # Left-align header row (row 1)
        ws = writer.book[sheet_name]
        for cell in ws[1]:
            cell.alignment = Alignment(horizontal="left", vertical="center")

    return output.getvalue()

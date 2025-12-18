"""
hsp_streamlit_1.0.0.py

- Target dropdowns driven by local Excel files in: data/target_materials/
- Optional default candidate dataset in: data/candidates/Default_Materials.xlsx
- Keeps manual target entry editable at all times.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

import hsp.core.hsp_core as core
from hsp.core.hsp_core import (
    load_input_data,
    calculate_hsp_distances,
    export_results_excel,
)

APP_VERSION = "1.0.0"

st.caption("RUNNING: hsp_streamlit_1.0.0.py")

# --- Your preferred names ---
TARGET_DIRNAME = "target_materials"
CANDIDATES_DIRNAME = "candidates"

TARGET_PRODUCTS_XLSX = "Products.xlsx"
TARGET_PRODUCT_TYPES_XLSX = "Product_Types.xlsx"
TARGET_SUPPLIERS_XLSX = "Suppliers.xlsx"

DEFAULT_CANDIDATE_XLSX = "Default_Materials.xlsx"


def find_repo_root(start: Path) -> Path:
    """Walk upward until we find pyproject.toml (repo root)."""
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start  # fallback


# -----------------------------
# Helpers
# -----------------------------
def _find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _candidate_locations(filename: str) -> List[Path]:
    """
    Search locations relative to the repo root (pyproject.toml),
    not relative to the script folder.
    """
    here = Path(__file__).resolve().parent
    repo_root = find_repo_root(here)

    return [
        repo_root / filename,
        repo_root / "data" / filename,
        repo_root / "data" / TARGET_DIRNAME / filename,
        repo_root / "data" / CANDIDATES_DIRNAME / filename,
    ]



def _safe_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize to name / dD / dP / dH.

    IMPORTANT:
    - We must rename Product → name BEFORE calling the core validator,
      because the core requires 'name' to exist.
    """
    # First-pass rename (Excel-friendly aliases)
    df = df.rename(
        columns={
            "Product": "name",
            "Material": "name",
            "Name": "name",
            "Solvent": "name",
            "Solvent/Name": "name",
            "δD": "dD",
            "δP": "dP",
            "δH": "dH",
        }
    )

    # Let the core do any additional normalization AFTER
    if hasattr(core, "_normalize_columns"):
        df = core._normalize_columns(df)  # type: ignore[attr-defined]

    return df



def _safe_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate/clean name/dD/dP/dH using your core helper if present."""
    if hasattr(core, "_validate_and_clean"):
        return core._validate_and_clean(df)  # type: ignore[attr-defined]

    df = df.copy()
    required = ["name", "dD", "dP", "dH"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df["name"] = df["name"].astype(str).str.strip()
    for c in ["dD", "dP", "dH"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["dD", "dP", "dH"]).reset_index(drop=True)


def _read_excel_all_sheets(path: Path) -> List[Tuple[str, pd.DataFrame]]:
    xls = pd.ExcelFile(path)
    return [(sh, pd.read_excel(path, sheet_name=sh)) for sh in xls.sheet_names]


@st.cache_data(show_spinner=False)
def load_target_catalog() -> pd.DataFrame:
    """
    Build a unified catalog from:
      Products.xlsx
      Product_Types.xlsx (tabs like Solvents/Polymers/Resins...)
      Suppliers.xlsx (tabs can be supplier names; may also contain Supplier/Product Type columns)

    Returns:
      product, product_type, supplier, dD, dP, dH, source_workbook, source_sheet, notes
    """
    rows: list[pd.DataFrame] = []

    # --- Products.xlsx ---
    p = _find_first_existing(_candidate_locations(TARGET_PRODUCTS_XLSX))
    if p:
        # Prefer sheet "Products", else first sheet
        try:
            df = pd.read_excel(p, sheet_name="Products")
            sh = "Products"
        except Exception:
            sh = pd.ExcelFile(p).sheet_names[0]
            df = pd.read_excel(p, sheet_name=sh)

        df = _safe_clean(_safe_normalize(df))

        rows.append(pd.DataFrame({
            "product": df["name"],
            "product_type": "Products",
            "supplier": df.get("supplier", "Unknown"),
            "dD": df["dD"], "dP": df["dP"], "dH": df["dH"],
            "source_workbook": p.name,
            "source_sheet": sh,
            "notes": df.get("notes", ""),
        }))

    # --- Product_Types.xlsx (tabs = product types) ---
    p = _find_first_existing(_candidate_locations(TARGET_PRODUCT_TYPES_XLSX))
    if p:
        for sh, df in _read_excel_all_sheets(p):
            df = _safe_clean(_safe_normalize(df))
            rows.append(pd.DataFrame({
                "product": df["name"],
                "product_type": sh,  # tab name
                "supplier": df.get("supplier", "Generic"),
                "dD": df["dD"], "dP": df["dP"], "dH": df["dH"],
                "source_workbook": p.name,
                "source_sheet": sh,
                "notes": df.get("notes", ""),
            }))

    # --- Suppliers.xlsx (tabs can be supplier names; allow Supplier/Product Type columns too) ---
    p = _find_first_existing(_candidate_locations(TARGET_SUPPLIERS_XLSX))
    if p:
        for sh, df in _read_excel_all_sheets(p):
            df = df.rename(columns={
                "Product": "name",
                "Material": "name",
                "Name": "name",
                "Manufacturer": "supplier",
                "Supplier": "supplier",
                "Type / Application": "product_type",
                "Product Type": "product_type",
                "Notes": "notes",
                "δD": "dD", "δP": "dP", "δH": "dH",
            })
            df = _safe_clean(_safe_normalize(df))

            supplier_col = df["supplier"] if "supplier" in df.columns else sh
            ptype_col = df["product_type"] if "product_type" in df.columns else "Supplier Products"
            notes_col = df["notes"] if "notes" in df.columns else ""

            rows.append(pd.DataFrame({
                "product": df["name"],
                "product_type": ptype_col,
                "supplier": supplier_col,
                "dD": df["dD"], "dP": df["dP"], "dH": df["dH"],
                "source_workbook": p.name,
                "source_sheet": sh,
                "notes": notes_col,
            }))

    if not rows:
        return pd.DataFrame(columns=[
            "product","product_type","supplier","dD","dP","dH",
            "source_workbook","source_sheet","notes"
        ])

    cat = pd.concat(rows, ignore_index=True)
    cat["product"] = cat["product"].astype(str).str.strip()
    cat["supplier"] = cat["supplier"].astype(str).str.strip()
    cat["product_type"] = cat["product_type"].astype(str).str.strip()
    for c in ["dD","dP","dH"]:
        cat[c] = pd.to_numeric(cat[c], errors="coerce")
    return cat.dropna(subset=["dD","dP","dH"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_default_candidates() -> Optional[pd.DataFrame]:
    """Optional candidate dataset bundled in repo."""
    p = _find_first_existing(_candidate_locations(DEFAULT_CANDIDATE_XLSX))
    if not p:
        return None
    df = pd.read_excel(p)
    return _safe_clean(_safe_normalize(df))


def _ensure_target_state():
    st.session_state.setdefault("target_name", "Hydrocarbon Resin")
    st.session_state.setdefault("target_dD", 17.0)
    st.session_state.setdefault("target_dP", 9.8)
    st.session_state.setdefault("target_dH", 9.4)


def _apply_target_from_pick(pick: pd.Series):
    st.session_state.target_name = str(pick["product"])
    st.session_state.target_dD = float(pick["dD"])
    st.session_state.target_dP = float(pick["dP"])
    st.session_state.target_dH = float(pick["dH"])


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title=f"HSP Solubility Matcher v{APP_VERSION}", layout="wide")
st.title("HSP Solubility Matcher")
st.caption(f"v{APP_VERSION}")

st.header("1) Data (Candidate Materials to Rank)")
uploaded_file = st.file_uploader("Upload candidates Excel (.xlsx/.xls)", type=["xlsx", "xls"])

default_df = load_default_candidates()
use_default = False
if default_df is not None:
    use_default = st.toggle(f"Use bundled default dataset ({DEFAULT_CANDIDATE_XLSX})", value=False)

st.header("2) Target Material Properties")

_ensure_target_state()
catalog = load_target_catalog()

target_mode = st.radio("Target input mode", ["Pick from library", "Manual entry"], horizontal=True)

if target_mode == "Pick from library":
    if catalog.empty:
        st.warning(
            f"Could not find target workbooks. Put them in ./data/{TARGET_DIRNAME}/:\n"
            f"- {TARGET_PRODUCTS_XLSX}\n- {TARGET_PRODUCT_TYPES_XLSX}\n- {TARGET_SUPPLIERS_XLSX}"
        )
    else:
        browse_by = st.selectbox("Browse by", ["Products", "Product Types", "Suppliers"])
        subset = catalog.copy()

        if browse_by == "Suppliers":
            supplier = st.selectbox("Supplier", sorted(subset["supplier"].unique()))
            subset = subset[subset["supplier"] == supplier]
            tab = st.selectbox("Workbook Tab", sorted(subset["source_sheet"].unique()))
            subset = subset[subset["source_sheet"] == tab]
            product = st.selectbox("Product", sorted(subset["product"].unique()))
            pick = subset[subset["product"] == product].iloc[0]

        elif browse_by == "Product Types":
            tab = st.selectbox("Workbook Tab", sorted(subset["source_sheet"].unique()))
            subset = subset[subset["source_sheet"] == tab]
            product = st.selectbox("Product", sorted(subset["product"].unique()))
            pick = subset[subset["product"] == product].iloc[0]

        else:
            product = st.selectbox("Product", sorted(subset["product"].unique()))
            pick = subset[subset["product"] == product].iloc[0]

        _apply_target_from_pick(pick)
        st.caption(f"Selected from {pick['source_workbook']} → tab {pick['source_sheet']}")

# Always-show editable manual inputs
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    target_name = st.text_input("Material Name", key="target_name")
with c2:
    target_dD = st.number_input("δD (Dispersion)", key="target_dD", step=0.1, format="%.2f")
with c3:
    target_dP = st.number_input("δP (Polar)", key="target_dP", step=0.1, format="%.2f")
with c4:
    target_dH = st.number_input("δH (H-bonding)", key="target_dH", step=0.1, format="%.2f")

target = {"name": target_name, "dD": target_dD, "dP": target_dP, "dH": target_dH}

st.header("3) Results")

materials_df = None
if uploaded_file is not None:
    # ✅ Read the uploaded Excel ourselves so we can normalize "Product" -> "name"
    uploaded_df = pd.read_excel(uploaded_file)
    uploaded_df = _safe_normalize(uploaded_df)
    uploaded_df = _safe_clean(uploaded_df)
    materials_df = uploaded_df

    st.success(f"Loaded uploaded candidates: {len(materials_df)} rows")

elif use_default and default_df is not None:
    materials_df = default_df
    st.info(f"Using bundled candidates: {len(materials_df)} rows")
else:
    st.info("Upload a candidates file (or enable the bundled default dataset) to proceed.")
    st.stop()

st.dataframe(materials_df.head(20), use_container_width=True)

if st.button("Calculate Solubility"):
    results_df = calculate_hsp_distances(materials_df, target=target, round_to=2)
    st.dataframe(results_df, use_container_width=True)

    xlsx_bytes = export_results_excel(results_df, sheet_name="HSP Results")
    st.download_button(
        "Download Results (XLSX)",
        data=xlsx_bytes,
        file_name="hsp_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


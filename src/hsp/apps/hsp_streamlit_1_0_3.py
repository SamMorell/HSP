from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

import pandas as pd
import streamlit as st

import hsp.core.hsp_core as core
from hsp.core.hsp_core import (
    calculate_hsp_distances,
    export_results_excel,
)

APP_VERSION = "1.0.3"

# --- Your preferred names ---
TARGET_DIRNAME = "target_materials"
CANDIDATES_DIRNAME = "candidates"

# Updated workbook names
TARGET_MATERIALS_XLSX = "Materials.xlsx"
TARGET_MATERIAL_TYPES_XLSX = "Material_Types.xlsx"
TARGET_SUPPLIERS_XLSX = "Suppliers.xlsx"

DEFAULT_CANDIDATE_XLSX = "default_dataset.xlsx"

#----- Helpers -----------------
KEEP_CASE = {"dD", "dP", "dH"}

def pretty_col(col: str) -> str:
    if col in KEEP_CASE:
        return col
    # replace underscores and title case
    return str(col).replace("_", " ").strip().title()

def pretty_headers_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [pretty_col(c) for c in df2.columns]
    return df2
#--------------------------------

def find_repo_root(start: Path) -> Path:
    """Walk upward until we find pyproject.toml (repo root)."""
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start  # fallback


def _find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def get_script_timestamp() -> str:
    script_path = Path(__file__)
    ts = script_path.stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

def _candidate_locations(filename: str) -> List[Path]:
    """Search locations relative to the repo root (pyproject.toml)."""
    here = Path(__file__).resolve().parent
    repo_root = find_repo_root(here)

    return [
        repo_root / filename,
        repo_root / "data" / filename,
        repo_root / "data" / "defaults" / filename,
        repo_root / "data" / TARGET_DIRNAME / filename,
        repo_root / "data" / CANDIDATES_DIRNAME / filename,
    ]


def _safe_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to name / dD / dP / dH."""
    df = df.rename(
        columns={
            "Material": "name",
            "Product": "name",
            "Name": "name",
            "Solvent": "name",
            "Solvent/Name": "name",
            "δD": "dD",
            "δP": "dP",
            "δH": "dH",
        }
    )

    # Let core normalize further if available
    if hasattr(core, "_normalize_columns"):
        df = core._normalize_columns(df)  # type: ignore[attr-defined]

    return df


def _safe_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate/clean name/dD/dP/dH using core helper if present."""
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


def _read_excel_all_sheets(path: Path) -> List[Tuple[int, str, pd.DataFrame]]:
    """Return (sheet_order, sheet_name, df) for each sheet in workbook order."""
    xls = pd.ExcelFile(path)
    out: List[Tuple[int, str, pd.DataFrame]] = []
    for i, sh in enumerate(xls.sheet_names):
        out.append((i, sh, pd.read_excel(path, sheet_name=sh)))
    return out


def _unique_preserve_order(values: pd.Series) -> List[str]:
    """Return unique string values in first-seen order (no sorting)."""
    seen: set[str] = set()
    out: List[str] = []
    for v in values.dropna().astype(str):
        v = v.strip()
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


@st.cache_data(show_spinner=False)
def load_target_catalog() -> pd.DataFrame:
    """
    Build a unified catalog from:
      Materials.xlsx (sheet: Materials, first column: Material)
      Material_Types.xlsx (tabs like Solvents/Polymers/Resins...)
      Suppliers.xlsx (Material/Supplier columns, preserving tab + row order)
    """
    rows: list[pd.DataFrame] = []

    # --- Materials.xlsx ---
    p = _find_first_existing(_candidate_locations(TARGET_MATERIALS_XLSX))
    if p:
        try:
            df = pd.read_excel(p, sheet_name="Materials")
            sh = "Materials"
            sh_order = 0
        except Exception:
            xls = pd.ExcelFile(p)
            sh = xls.sheet_names[0]
            sh_order = 0
            df = pd.read_excel(p, sheet_name=sh)

        df = _safe_clean(_safe_normalize(df))
        df = df.reset_index(drop=True)
        row_order = list(range(len(df)))

        rows.append(
            pd.DataFrame(
                {
                    "product": df["name"],
                    "product_type": "Materials",
                    "supplier": df.get("supplier", "Unknown"),
                    "dD": df["dD"],
                    "dP": df["dP"],
                    "dH": df["dH"],
                    "source_workbook": p.name,
                    "source_sheet": sh,
                    "sheet_order": sh_order,
                    "row_order": row_order,
                    "notes": df.get("notes", ""),
                }
            )
        )

    # --- Material_Types.xlsx ---
    p = _find_first_existing(_candidate_locations(TARGET_MATERIAL_TYPES_XLSX))
    if p:
        for sh_order, sh, df in _read_excel_all_sheets(p):
            df = _safe_clean(_safe_normalize(df))
            df = df.reset_index(drop=True)
            row_order = list(range(len(df)))

            rows.append(
                pd.DataFrame(
                    {
                        "product": df["name"],
                        "product_type": sh,
                        "supplier": df.get("supplier", "Generic"),
                        "dD": df["dD"],
                        "dP": df["dP"],
                        "dH": df["dH"],
                        "source_workbook": p.name,
                        "source_sheet": sh,
                        "sheet_order": sh_order,
                        "row_order": row_order,
                        "notes": df.get("notes", ""),
                    }
                )
            )

    # --- Suppliers.xlsx ---
    p = _find_first_existing(_candidate_locations(TARGET_SUPPLIERS_XLSX))
    if p:
        for sh_order, sh, df in _read_excel_all_sheets(p):
            df = df.rename(
                columns={
                    "Product": "name",
                    "Material": "name",
                    "Name": "name",
                    "Manufacturer": "supplier",
                    "Supplier": "supplier",
                    "Type / Application": "product_type",
                    "Product Type": "product_type",
                    "Notes": "notes",
                    "δD": "dD",
                    "δP": "dP",
                    "δH": "dH",
                }
            )
            df = _safe_clean(_safe_normalize(df))
            df = df.reset_index(drop=True)
            row_order = list(range(len(df)))

            supplier_col = df["supplier"] if "supplier" in df.columns else sh
            ptype_col = df["product_type"] if "product_type" in df.columns else "Supplier Materials"
            notes_col = df["notes"] if "notes" in df.columns else ""

            rows.append(
                pd.DataFrame(
                    {
                        "product": df["name"],
                        "product_type": ptype_col,
                        "supplier": supplier_col,
                        "dD": df["dD"],
                        "dP": df["dP"],
                        "dH": df["dH"],
                        "source_workbook": p.name,
                        "source_sheet": sh,
                        "sheet_order": sh_order,
                        "row_order": row_order,
                        "notes": notes_col,
                    }
                )
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "product",
                "product_type",
                "supplier",
                "dD",
                "dP",
                "dH",
                "source_workbook",
                "source_sheet",
                "sheet_order",
                "row_order",
                "notes",
            ]
        )

    cat = pd.concat(rows, ignore_index=True)
    cat["product"] = cat["product"].astype(str).str.strip()
    cat["supplier"] = cat["supplier"].astype(str).str.strip()
    cat["product_type"] = cat["product_type"].astype(str).str.strip()
    for c in ["dD", "dP", "dH"]:
        cat[c] = pd.to_numeric(cat[c], errors="coerce")
    cat["sheet_order"] = pd.to_numeric(cat["sheet_order"], errors="coerce")
    cat["row_order"] = pd.to_numeric(cat["row_order"], errors="coerce")
    return cat.dropna(subset=["dD", "dP", "dH"]).reset_index(drop=True)


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


def main():
    # --- Page config MUST be inside main, before UI ---
    st.set_page_config(page_title=f"Hansen Solubility Parameters v{APP_VERSION}", layout="wide")

    # --- Header GIF (safe: only try if file exists) ---
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    gif_path = repo_root / "assets" / "sammorell.com_animated_header_no_loop.gif"
    if gif_path.exists():
        st.image(str(gif_path), width=360)

    # --- CSS (your styles) ---
    st.markdown(
        """
        <style>
        .hsp-title {
            color: #000000;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 2.2rem;
            font-weight: 600;
            margin-bottom: 0.15em;
            line-height: 1.1;
        }
        .hsp-section {
            color: #FFA500;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 1.6rem;
            font-weight: 600;
            margin-top: .1em;
            margin-bottom: 0.1em;
            padding-bottom: 0.1em;
            border-top: 2px solid rgb(255, 165, 0);

            /* tighten spacing right after a section header */
            margin-bottom: 0.0rem !important; padding-bottom: 0.0rem !important;
        }

        /* radio block tends to add top padding; pull it up a bit */
        div[data-testid="stRadio"] { margin-top: -30px !important; }

        /* also reduce spacing Streamlit inserts between blocks */
        #div.block-container { padding-top: 1.4rem; }

        </style>
        """,

        unsafe_allow_html=True,
    )

    # --- Title + Version ---
    st.markdown('<div class="hsp-title">Hansen Solubility Parameters</div>', unsafe_allow_html=True)
    
    
    st.caption(f"{APP_VERSION} ({get_script_timestamp()})")
    
    # st.caption(f"v{APP_VERSION}")
    # st.caption("RUNNING: hsp_streamlit_1.0.2 (via shim)")

    # ===== Target Materials =====
    st.markdown('<div class="hsp-section">Target Materials</div>', unsafe_allow_html=True)

    _ensure_target_state()
    catalog = load_target_catalog()

    target_mode = st.radio("", ["Pick from library", "Manual entry"], horizontal=True)

    if target_mode == "Pick from library":
        missing_msgs: List[str] = []
        if _find_first_existing(_candidate_locations(TARGET_MATERIALS_XLSX)) is None:
            missing_msgs.append(f"- {TARGET_MATERIALS_XLSX} (sheet: Materials, first column: Material)")
        if _find_first_existing(_candidate_locations(TARGET_MATERIAL_TYPES_XLSX)) is None:
            missing_msgs.append(f"- {TARGET_MATERIAL_TYPES_XLSX} (tabs remain; first column: Material)")

        if missing_msgs and catalog.empty:
            st.warning(
                f"Could not find target workbooks. Put them in ./data/{TARGET_DIRNAME}/:\n"
                + "\n".join(missing_msgs)
                + f"\n- {TARGET_SUPPLIERS_XLSX}"
            )
        elif catalog.empty:
            st.warning(f"Target catalog is empty. Check your Excel files under ./data/{TARGET_DIRNAME}/.")
        else:
            browse_by = st.selectbox("Browse by", ["Materials", "Material Types", "Suppliers"])
            subset = catalog.copy()

            if browse_by == "Suppliers":
                subset = subset[subset["source_workbook"] == TARGET_SUPPLIERS_XLSX]
                if subset.empty:
                    st.warning(f"No rows found from {TARGET_SUPPLIERS_XLSX}.")
                else:
                    subset_sorted = subset.sort_values(["sheet_order", "row_order"], kind="stable")
                    supplier_options = _unique_preserve_order(subset_sorted["supplier"])
                    supplier = st.selectbox("Supplier", supplier_options)

                    subset_supplier = subset_sorted[subset_sorted["supplier"] == supplier]
                    tabs_df = subset_supplier[["source_sheet", "sheet_order"]].drop_duplicates()
                    tabs_df = tabs_df.sort_values("sheet_order", kind="stable")
                    tab_options = tabs_df["source_sheet"].astype(str).tolist()

                    tab = st.selectbox("Workbook Tab", tab_options)
                    subset_tab = (
                        subset_supplier[subset_supplier["source_sheet"] == tab]
                        .sort_values("row_order", kind="stable")
                    )

                    material_options = _unique_preserve_order(subset_tab["product"])
                    material = st.selectbox("Material", material_options)
                    pick = subset_tab[subset_tab["product"] == material].iloc[0]

            elif browse_by == "Material Types":
                subset = subset[subset["source_workbook"] == TARGET_MATERIAL_TYPES_XLSX]
                if subset.empty:
                    st.warning(f"No rows found from {TARGET_MATERIAL_TYPES_XLSX}.")
                else:
                    tabs_df = subset[["source_sheet", "sheet_order"]].drop_duplicates()
                    tabs_df = tabs_df.sort_values("sheet_order", kind="stable")
                    tab_options = tabs_df["source_sheet"].astype(str).tolist()

                    tab = st.selectbox("Workbook Tab", tab_options)
                    subset_tab = subset[subset["source_sheet"] == tab].sort_values("row_order", kind="stable")

                    material_options = _unique_preserve_order(subset_tab["product"])
                    material = st.selectbox("Material", material_options)
                    pick = subset_tab[subset_tab["product"] == material].iloc[0]

            else:
                subset = subset[subset["source_workbook"] == TARGET_MATERIALS_XLSX]
                if subset.empty:
                    st.warning(f"No rows found from {TARGET_MATERIALS_XLSX}.")
                else:
                    subset = subset.sort_values("row_order", kind="stable")
                    material_options = _unique_preserve_order(subset["product"])
                    material = st.selectbox("Material", material_options)
                    pick = subset[subset["product"] == material].iloc[0]

            if "pick" in locals():
                _apply_target_from_pick(pick)
                st.caption(f"Selected from {pick['source_workbook']} → tab {pick['source_sheet']}")

    # Always-show editable manual inputs
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        target_name = st.text_input("Name", key="target_name")
    with c2:
        target_dD = st.number_input("δD – Dispersion", key="target_dD", step=0.1, format="%.2f")
    with c3:
        target_dP = st.number_input("δP – Polar", key="target_dP", step=0.1, format="%.2f")
    with c4:
        target_dH = st.number_input("δH – Hydrogen Bonding", key="target_dH", step=0.1, format="%.2f")

    target = {"name": target_name, "dD": target_dD, "dP": target_dP, "dH": target_dH}

    # ===== Candidate Materials =====
    st.markdown('<div class="hsp-section">Candidate Materials</div>', unsafe_allow_html=True)

    default_df = load_default_candidates()
    use_default = False
    if default_df is not None:
        use_default = st.toggle(f"Use {DEFAULT_CANDIDATE_XLSX}", value=False)

    uploaded_file = st.file_uploader("Or, upload candidate materials (.xlsx/.xls)", type=["xlsx", "xls"])

    # ===== Results =====
    #st.markdown('<div class="hsp-section">Results</div>', unsafe_allow_html=True)

    materials_df = None
    if uploaded_file is not None:
        uploaded_df = pd.read_excel(uploaded_file)
        uploaded_df = _safe_normalize(uploaded_df)
        uploaded_df = _safe_clean(uploaded_df)
        materials_df = uploaded_df
        st.success(f"Using uploaded candidates: {len(materials_df)} rows")

    elif use_default and default_df is not None:
        materials_df = default_df
        st.info(f"Using default_dataset.xlsx: {len(materials_df)} rows")
    else:
        #st.info("Upload a candidates file or use default_dataset.xlsx")
        st.stop()

    #st.dataframe(materials_df, use_container_width=True)
    st.dataframe(pretty_headers_df(materials_df), use_container_width=True)

    
    if st.button("Calculate Solubility"):
        results_df = calculate_hsp_distances(materials_df, target=target, round_to=2)
        #st.dataframe(results_df, use_container_width=True)
        results_display = pretty_headers_df(results_df)
        st.dataframe(results_display, use_container_width=True)


        xlsx_bytes = export_results_excel(results_df, sheet_name="HSP Results")
        st.download_button(
            "Download Results (XLSX)",
            data=xlsx_bytes,
            file_name="hsp_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        
if __name__ == "__main__":
    main()

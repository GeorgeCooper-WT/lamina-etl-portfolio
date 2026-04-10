"""
SQLModel ORM models mapping the SiteConfiguration JSON schema to a relational database.

Tables:
  sites  ->  inverters  ->  dc_combiner_boxes  ->  strings

All primary keys are UUIDs for cross-site scalability.
"""

import uuid as _uuid
from datetime import date
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _new_uuid() -> str:
    """Generate a new UUID as a string (SQLite-friendly)."""
    return str(_uuid.uuid4())


# ---------------------------------------------------------------------------
# sites
# ---------------------------------------------------------------------------
class Site(SQLModel, table=True):
    __tablename__ = "sites"

    id: str = Field(default_factory=_new_uuid, primary_key=True)
    name: str
    total_capacity_mwp: float
    commissioning_date: date

    # Relationships
    inverters: List["Inverter"] = Relationship(back_populates="site")


# ---------------------------------------------------------------------------
# inverters
# ---------------------------------------------------------------------------
class Inverter(SQLModel, table=True):
    __tablename__ = "inverters"

    id: str = Field(default_factory=_new_uuid, primary_key=True)
    site_id: str = Field(foreign_key="sites.id")
    label: str  # JSON key: inverter_id  (e.g. "INV-50")
    model_name: str
    max_ac_kva: float  # JSON key: max_ac_output_kva
    max_dc_kwp: Optional[float] = None  # JSON key: max_dc_power_kwp
    efficiency_max: Optional[float] = None
    mpp_trackers: Optional[int] = None
    mppt_voltage_min_v: Optional[float] = None
    mppt_voltage_max_v: Optional[float] = None
    valid_from: date
    valid_to: Optional[date] = None  # NULL = still active

    # Relationships
    site: Optional[Site] = Relationship(back_populates="inverters")
    dc_combiner_boxes: List["DCCombinerBox"] = Relationship(back_populates="inverter")


# ---------------------------------------------------------------------------
# dc_combiner_boxes
# ---------------------------------------------------------------------------
class DCCombinerBox(SQLModel, table=True):
    __tablename__ = "dc_combiner_boxes"

    id: str = Field(default_factory=_new_uuid, primary_key=True)
    inverter_id: str = Field(foreign_key="inverters.id")
    label: str  # JSON key: box_id  (e.g. "AJB-50")
    fuses_rating_a: Optional[float] = None
    switch_type: Optional[str] = None
    spd_voltage_v: Optional[float] = None
    spd_current_ka: Optional[float] = None
    aggregated_v_mpp_v: Optional[float] = None
    aggregated_i_mpp_a: Optional[float] = None
    aggregated_p_mpp_kw: Optional[float] = None
    valid_from: date
    valid_to: Optional[date] = None  # NULL = still active

    # Relationships
    inverter: Optional[Inverter] = Relationship(back_populates="dc_combiner_boxes")
    strings: List["StringRow"] = Relationship(back_populates="dc_combiner_box")


# ---------------------------------------------------------------------------
# module_specs  (full manufacturer spec sheet, one row per module type)
# ---------------------------------------------------------------------------
class ModuleSpec(SQLModel, table=True):
    """
    Full manufacturer spec-sheet for a PV module type.

    Populated from the component database via BOM enrichment.
    Multiple StringRow records reference a single ModuleSpec row,
    avoiding duplication of spec data across hundreds of strings.
    """

    __tablename__ = "module_specs"

    id: str = Field(default_factory=_new_uuid, primary_key=True)

    # --- Identification ---
    model_number: str = Field(index=True)  # e.g. "TSM-250 PC/PA05A"
    manufacturer: Optional[str] = None  # e.g. "Trina Solar"
    series_name: Optional[str] = None  # e.g. "TSM PC/PA05A"
    cell_type: Optional[str] = None  # e.g. "Polycrystalline"
    cell_count: Optional[int] = None  # e.g. 60

    # --- STC (Standard Test Conditions: 1000 W/m², 25 °C, AM 1.5) ---
    stc_pmax_wp: Optional[float] = None  # 250
    stc_vmpp_v: Optional[float] = None  # 30.3
    stc_impp_a: Optional[float] = None  # 8.27
    stc_voc_v: Optional[float] = None  # 38.0
    stc_isc_a: Optional[float] = None  # 8.79
    stc_efficiency_pct: Optional[float] = None  # 15.3

    # --- NOCT (Nominal Operating Cell Temperature: 800 W/m², 20 °C) ---
    noct_pmax_wp: Optional[float] = None  # 183
    noct_vmpp_v: Optional[float] = None  # 27.3
    noct_impp_a: Optional[float] = None  # 6.7
    noct_voc_v: Optional[float] = None  # 34.8
    noct_isc_a: Optional[float] = None  # 6.99

    # --- Temperature Coefficients ---
    temp_coeff_pmax_pct_per_c: Optional[float] = None  # -0.41
    temp_coeff_voc_pct_per_c: Optional[float] = None  # -0.32
    temp_coeff_isc_pct_per_c: Optional[float] = None  # 0.053
    noct_temp_c: Optional[str] = None  # "44±2 °C"
    operating_temp_min_c: Optional[float] = None  # -40
    operating_temp_max_c: Optional[float] = None  # 85

    # --- Electrical Limits ---
    max_system_voltage_v: Optional[float] = None  # 1000
    series_fuse_rating_a: Optional[float] = None  # 15
    power_tolerance_pct: Optional[str] = None  # "0 ~ +3 %"

    # --- Physical ---
    height_mm: Optional[float] = None  # 1650
    width_mm: Optional[float] = None  # 992
    depth_mm: Optional[float] = None  # 35
    weight_kg: Optional[float] = None  # 18.6
    cell_size_mm: Optional[str] = None  # "156×156 mm"
    glass_thickness_mm: Optional[float] = None  # 3.2
    frame_type: Optional[str] = None  # "Anodized Aluminium"
    connector_type: Optional[str] = None  # "MC4"
    cable_cross_section_mm2: Optional[float] = None  # 4
    cable_length_mm: Optional[int] = None  # 1000
    junction_box_protection_class: Optional[str] = None  # "IP 65"

    # --- Warranty ---
    product_warranty_years: Optional[int] = None  # 10
    power_warranty_description: Optional[str] = None  # "10yr@90%, 25yr@80%"
    annual_degradation_pct: Optional[float] = None  # 0.8

    # Relationships
    strings: List["StringRow"] = Relationship(back_populates="module_spec")


# ---------------------------------------------------------------------------
# strings
# ---------------------------------------------------------------------------
class StringRow(SQLModel, table=True):
    """
    Named StringRow to avoid collision with Python's built-in `str`.
    Maps to the 'strings' table.
    """

    __tablename__ = "strings"

    id: str = Field(default_factory=_new_uuid, primary_key=True)
    box_id: str = Field(foreign_key="dc_combiner_boxes.id")
    module_spec_id: Optional[str] = Field(
        default=None, foreign_key="module_specs.id"
    )  # FK to full manufacturer spec sheet
    label: str  # JSON key: string_id  (e.g. "String 50.1.A")
    module_count: int  # JSON key: modules.count
    module_model_name: Optional[str] = None  # JSON key: modules.model_name
    module_p_wp: float  # JSON key: modules.peak_power_wp
    module_vmp: Optional[float] = None  # JSON key: modules.vmp
    module_imp: Optional[float] = None  # JSON key: modules.imp
    module_voc: Optional[float] = None  # JSON key: modules.voc
    module_isc: Optional[float] = None  # JSON key: modules.isc
    module_temp_coeff_pmax: Optional[float] = None
    tilt: Optional[float] = None
    azimuth: Optional[float] = None
    tracking_type: str = "Fixed"
    v_mpp_v: Optional[float] = None
    i_mpp_a: Optional[float] = None
    p_mpp_kw: Optional[float] = None
    valid_from: date
    valid_to: Optional[date] = None  # NULL = still active

    # Relationships
    dc_combiner_box: Optional[DCCombinerBox] = Relationship(back_populates="strings")
    module_spec: Optional[ModuleSpec] = Relationship(back_populates="strings")

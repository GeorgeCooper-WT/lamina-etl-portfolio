from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import List, Optional, Dict
from datetime import date


class DCCombinerBox(BaseModel):
    box_id: str  # e.g., "AJB 50"
    inverter_id: str  # Reference to parent inverter
    string_ids: List[str]
    fuses_rating_a: Optional[float]  # e.g., 16A
    # Hardware and aggregated outputs explicitly shown in the AJB diagram
    switch_type: Optional[str] = Field(
        None, description="Type of disconnect switch (e.g., FPST)"
    )
    spd_voltage_v: Optional[float] = Field(
        None, description="Surge Protection Device voltage rating (e.g., 1000V)"
    )
    spd_current_ka: Optional[float] = Field(
        None, description="Surge Protection Device kilo-amp rating (e.g., 30KA)"
    )
    aggregated_v_mpp_v: Optional[float] = Field(
        None, description="Total Vmpp leaving the box (e.g., 600V)"
    )
    aggregated_i_mpp_a: Optional[float] = Field(
        None, description="Total Impp leaving the box (e.g., 33.68A)"
    )
    aggregated_p_mpp_kw: Optional[float] = Field(
        None, description="Total Pmpp leaving the box (e.g., 20KW)"
    )

    # --- Range / repeat pattern (SLD dotted-line convention) -----------------
    implied_range_end: Optional[int] = Field(
        None,
        description=(
            "If this AJB is the low-numbered bookend of a repeated range "
            "(e.g. AJB-1 shown with a dotted line to AJB-50), set this to the "
            "high-end number (50).  The system will clone this box's "
            "structure for every number in the range.  Leave null when standalone."
        ),
    )


class DCCombinerBoxExtractResponse(BaseModel):
    boxes: List[DCCombinerBox]

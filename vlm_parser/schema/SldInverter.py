from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import List, Optional, Dict
from datetime import date


class Inverter(BaseModel):
    inverter_id: str  # e.g., "INV-1"
    input_dc_box_ids: List[str]  # Only the IDs, not the full objects
    model_name: str
    max_ac_output_kva: float
    max_dc_power_kwp: Optional[float] = Field(
        None, description="Max allowed DC input for clipping analysis"
    )
    efficiency_max: Optional[float] = Field(
        None, description="Peak efficiency from OEM specs"
    )
    mpp_trackers: Optional[int] = Field(
        None, description="Number of independent MPPT inputs"
    )
    mppt_voltage_min_v: Optional[float] = Field(
        None, description="Lower bound for peak efficiency"
    )
    mppt_voltage_max_v: Optional[float] = Field(
        None, description="Upper bound for peak efficiency"
    )

    # --- Range / repeat pattern (SLD dotted-line convention) -----------------
    # When an SLD shows e.g. INV-1 ... INV-50 with a dotted line, the LLM
    # should set implied_range_end to the *other* bookend inverter's numeric
    # suffix.  Leave None when the inverter is standalone or is the high-end
    # bookend (only the low-numbered bookend carries the range).

    implied_range_end: Optional[int] = Field(
        None,
        description=(
            "If this inverter is the low-numbered bookend of a repeated range "
            "(e.g. INV-1 shown with a dotted line to INV-50), set this to the "
            "high-end number (50).  The system will clone this inverter's "
            "structure for every number in the range.  Leave null when standalone."
        ),
    )


class InverterExtractResponse(BaseModel):
    inverters: List[Inverter]

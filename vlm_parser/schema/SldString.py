from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import List, Optional, Dict
from datetime import date


class String(BaseModel):
    string_id: str
    box_id: str  # Reference to parent DC box
    module_ids: List[str]  # Only the IDs, not the full objects
    tilt: Optional[float] = Field(None, description="Angle from horizontal")
    azimuth: Optional[float] = Field(None, description="Degrees from North")
    tracking_type: str = "Fixed"  # e.g., Fixed, Single-Axis, Dual-Axis

    # Extracted directly from the SLD String lines
    v_mpp_v: Optional[float] = Field(
        None, description="String voltage at max power (e.g., 600V)"
    )
    i_mpp_a: Optional[float] = Field(
        None, description="String current at max power (e.g., 8.42A)"
    )
    p_mpp_kw: Optional[float] = Field(
        None, description="String power at max power (e.g., 5kW)"
    )


class StringExtractResponse(BaseModel):
    strings: List[String]

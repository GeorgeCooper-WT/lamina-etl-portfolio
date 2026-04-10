from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import List, Optional, Dict
from datetime import date


class SolarModule(BaseModel):
    model_name: Optional[str] = Field(None, description="OEM model name/number")
    peak_power_wp: float
    temp_coeff_pmax: Optional[float] = Field(
        None, description="Critical for Physics-ML (%/°C)"
    )
    voc: Optional[float] = Field(
        None, description="Open Circuit Voltage per module (V)"
    )
    isc: Optional[float] = Field(
        None, description="Short Circuit Current per module (A)"
    )
    vmp: Optional[float] = Field(
        None, description="Voltage at Maximum Power per module (V)"
    )
    imp: Optional[float] = Field(
        None, description="Current at Maximum Power per module (A)"
    )
    count_per_string: int = Field(description="Number of modules in series per string")
    applies_to_strings: List[str] = Field(
        description=(
            "List of string IDs that use this module type. "
            "Use 'all' as a single entry if every string on the site uses this module."
        )
    )


class ModuleExtractResponse(BaseModel):
    modules: List[SolarModule]

from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import List, Optional, Dict
from datetime import date
from schema.SldInverter import Inverter


class Metadata(BaseModel):
    site_name: str
    total_capacity_mwp: float
    commissioning_date: date
    version: int = 1
    valid_from: date
    valid_to: Optional[date] = None
    inverters: Optional[List[Inverter]] = None

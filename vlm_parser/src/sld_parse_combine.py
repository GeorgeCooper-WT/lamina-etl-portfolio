"""
Assembles a complete site configuration from SLD image(s) using modular LLM extractors.
Usage:
    python src/sld_parse_combine.py <site_id>
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add project root to sys.path so "schema" can be imported
sys.path.append(str(Path(__file__).parent.parent))

from schema.site_config import SiteConfiguration

from src.sld_metadata_extract import parse_site_metadata
from src.sld_inverter_extract import parse_inverter_metadata
from src.sld_dccombiner_extract import parse_dc_combiner_metadata
from src.sld_string_extract import parse_string_metadata
from src.sld_module_extract import parse_module_metadata

import re


def _normalise_id(id_str: str) -> str:
    """Normalise an ID for fuzzy matching: lowercase, strip spaces/hyphens."""
    return re.sub(r"[\s\-_]+", "", id_str).lower()


def _fuzzy_lookup(target_id: str, id_dict: dict) -> dict | None:
    """Look up a value in a dict by normalised ID matching."""
    # Try exact match first
    if target_id in id_dict:
        return id_dict[target_id]
    # Try normalised match
    norm_target = _normalise_id(target_id)
    for key, val in id_dict.items():
        if _normalise_id(key) == norm_target:
            return val
    return None


def assemble_full_site_config(image_path: str, site_id: str) -> SiteConfiguration:

    # Load display_name from projects.json
    with open("data/projects.json", "r", encoding="utf-8") as f:
        projects = json.load(f)
    display_name = projects[site_id]["display_name"]

    # 1. Run all modular extractors
    site_metadata = parse_site_metadata(image_path)
    site_metadata.site_name = display_name
    inverters_resp = parse_inverter_metadata(image_path)
    dc_boxes_resp = parse_dc_combiner_metadata(image_path)
    strings_resp = parse_string_metadata(image_path)
    modules_resp = parse_module_metadata(image_path)

    # 2. Build module lookup: resolve "all" or match by string ID
    def get_module_for_string(string_id: str) -> dict:
        """Find the module spec that applies to a given string."""
        for m in modules_resp.modules:
            if "all" in m.applies_to_strings or string_id in m.applies_to_strings:
                return {
                    "model_name": m.model_name,
                    "peak_power_wp": m.peak_power_wp,
                    "temp_coeff_pmax": m.temp_coeff_pmax,
                    "voc": m.voc,
                    "isc": m.isc,
                    "vmp": m.vmp,
                    "imp": m.imp,
                    "count": m.count_per_string,
                }
        # Fallback: return first module if no match found
        if modules_resp.modules:
            m = modules_resp.modules[0]
            return {
                "model_name": m.model_name,
                "peak_power_wp": m.peak_power_wp,
                "temp_coeff_pmax": m.temp_coeff_pmax,
                "voc": m.voc,
                "isc": m.isc,
                "vmp": m.vmp,
                "imp": m.imp,
                "count": m.count_per_string,
            }
        return None

    # 3. Build string dicts with modules attached
    string_dict = {}
    for s in strings_resp.strings:
        sd = s.model_dump()
        sd.pop("module_ids", None)  # Remove module_ids, not needed in final output
        sd["modules"] = get_module_for_string(s.string_id)
        string_dict[s.string_id] = sd

    # 4. Build DC box dicts with strings attached
    dc_box_dict = {}
    for box in dc_boxes_resp.boxes:
        bd = box.model_dump()
        # Match strings to this box by box_id (fuzzy)
        norm_box_id = _normalise_id(box.box_id)
        matched_strings = [
            sd
            for sid, sd in string_dict.items()
            if _normalise_id(sd.get("box_id", "")) == norm_box_id
        ]
        # Fallback: fuzzy match on string_ids list
        if not matched_strings and bd.get("string_ids"):
            for str_sid in bd["string_ids"]:
                for s_sid, sd in string_dict.items():
                    if str_sid in s_sid or s_sid in str_sid:
                        matched_strings.append(sd)
        bd["strings"] = matched_strings if matched_strings else []
        # Remove fields not in site_config's DCCombinerBox
        bd.pop("string_ids", None)
        bd.pop("inverter_id", None)
        bd.pop("implied_range_end", None)
        dc_box_dict[box.box_id] = bd

    # 5. Build inverter dicts (no range expansion, preserved for SQL step)
    inverter_dicts = []
    for inv in inverters_resp.inverters:
        inv_dict = inv.model_dump()
        inv_dict.pop("input_dc_box_ids", None)

        # Attach the bookend's DC boxes (fuzzy match on box ID)
        bookend_boxes = []
        for bid in inv.input_dc_box_ids:
            matched_box = _fuzzy_lookup(bid, dc_box_dict)
            if matched_box:
                bookend_boxes.append(matched_box)
        inv_dict["input_dc_boxes"] = bookend_boxes if bookend_boxes else []

        inverter_dicts.append(inv_dict)

    # 6. Assemble the full SiteConfiguration
    site_config = SiteConfiguration.model_validate(
        {
            "site_name": site_metadata.site_name,
            "total_capacity_mwp": site_metadata.total_capacity_mwp,
            "commissioning_date": site_metadata.commissioning_date,
            "inverters": inverter_dicts,
            "version": site_metadata.version,
            "valid_from": site_metadata.valid_from,
            "valid_to": site_metadata.valid_to,
        }
    )
    return site_config


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python src/sld_parse_combine.py <site_id>")
        sys.exit(1)
    site_id = sys.argv[1]

    # Set up project paths
    project_base = Path("db") / site_id
    input_folder = project_base / "input"
    output_folder = project_base / "json"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Find the first image file in the input folder
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
    image_files = [
        f for f in input_folder.iterdir() if f.suffix.lower() in image_extensions
    ]
    if not image_files:
        logger.error(f"No image files found in {input_folder}.")
        sys.exit(1)
    image_path = str(image_files[0])
    logger.info(f"Parsing and assembling SLD image: {image_path}")

    site_config = assemble_full_site_config(image_path, site_id)
    logger.info(site_config.model_dump_json(indent=2))

    # Save JSON with a consistent filename in the project json folder
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_filename = f"{site_id}_sld_parse_{timestamp}.json"
    output_path = output_folder / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(site_config.model_dump_json(indent=2))
    logger.info(f"Saved JSON to {output_path}")

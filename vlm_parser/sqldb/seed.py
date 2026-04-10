"""
Seed the SQLite database from a SiteConfiguration JSON file.

Usage:
    python db/seed.py <project_name>                           # latest JSON in db/<project>/json/
    python db/seed.py <project_name> --json path/to/file.json  # specific JSON file
    python db/seed.py <project_name> --db path/to/db.sqlite    # custom DB path
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from sqlmodel import Session, SQLModel, create_engine

# Ensure the db package is importable when running as a script
_DB_DIR = Path(__file__).resolve().parent
if str(_DB_DIR) not in sys.path:
    sys.path.insert(0, str(_DB_DIR))

from models import DCCombinerBox, Inverter, ModuleSpec, Site, StringRow  # noqa: E402

# Ensure schema package is importable for expand_range
_SCHEMA_DIR = str(Path(__file__).resolve().parent.parent / "schema")
if _SCHEMA_DIR not in sys.path:
    sys.path.insert(0, _SCHEMA_DIR)

from expand_range import expand_inverter_ranges  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_PROJECTS_FILE = _ROOT / "data" / "projects.json"

SENTINEL_VALID_TO = date(9999, 12, 31)


# ---------------------------------------------------------------------------
# Project path resolution
# ---------------------------------------------------------------------------
def resolve_project_paths(
    project_name: str,
    json_override: str | Path | None = None,
    db_override: str | Path | None = None,
) -> tuple[Path, Path]:
    """
    Derive JSON and DB paths from a project name.

    - JSON: latest file in db/<project_name>/json/  (or json_override)
    - DB  : db/<project_name>/sqldb/<project_name>.sqlite  (or db_override)

    Raises SystemExit if the project is unknown or no JSON files are found.
    """
    # Validate project exists in the registry
    if _PROJECTS_FILE.exists():
        with open(_PROJECTS_FILE, "r", encoding="utf-8") as f:
            projects = json.load(f)
        if project_name not in projects:
            logger.error(
                f"Unknown project '{project_name}'. "
                f"Run `python src/create_project.py` first, or check data/projects.json."
            )
            sys.exit(1)
    else:
        logger.warning("data/projects.json not found — skipping project validation.")

    project_dir = _ROOT / "db" / project_name

    # Resolve JSON path
    if json_override:
        json_path = Path(json_override)
    else:
        json_dir = project_dir / "json"
        candidates = sorted(json_dir.glob("*.json"))
        if not candidates:
            logger.error(f"No JSON files found in {json_dir}")
            sys.exit(1)
        json_path = candidates[
            -1
        ]  # latest by filename (ISO timestamp sorts lexicographically)
        logger.info(f"Using JSON: {json_path.name}")

    # Resolve DB path
    if db_override:
        db_path = Path(db_override)
    else:
        db_path = project_dir / "sqldb" / f"{project_name}.sqlite"

    return json_path, db_path


# ---------------------------------------------------------------------------
# Core seeding logic
# ---------------------------------------------------------------------------
def seed_from_json(json_path: str | Path, db_path: str | Path) -> str:
    """
    Read a SiteConfiguration JSON file and insert all rows into an SQLite DB.

    Returns the site UUID so callers can query immediately.
    """
    json_path = Path(json_path)
    db_path = Path(db_path)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data: dict = json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        sys.exit(1)

    # --- Expand implied inverter ranges (SLD dotted-line convention) ---
    expand_inverter_ranges(data)

    # --- Create engine + tables ----
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    SQLModel.metadata.create_all(engine)

    commissioning = date.fromisoformat(data["commissioning_date"])
    valid_to_raw = data.get("valid_to")
    valid_to = date.fromisoformat(valid_to_raw) if valid_to_raw else None

    with Session(engine) as session:
        # 1. Site
        site = Site(
            name=data["site_name"],
            total_capacity_mwp=data["total_capacity_mwp"],
            commissioning_date=commissioning,
        )
        session.add(site)
        session.flush()  # populates site.id

        # 2. Inverters
        for inv_data in data.get("inverters", []):
            inv = Inverter(
                site_id=site.id,
                label=inv_data["inverter_id"],
                model_name=inv_data["model_name"],
                max_ac_kva=inv_data["max_ac_output_kva"],
                max_dc_kwp=inv_data.get("max_dc_power_kwp"),
                efficiency_max=inv_data.get("efficiency_max"),
                mpp_trackers=inv_data.get("mpp_trackers"),
                mppt_voltage_min_v=inv_data.get("mppt_voltage_min_v"),
                mppt_voltage_max_v=inv_data.get("mppt_voltage_max_v"),
                valid_from=commissioning,
                valid_to=valid_to or SENTINEL_VALID_TO,
            )
            session.add(inv)
            session.flush()

            # 3. DC Combiner Boxes
            for box_data in inv_data.get("input_dc_boxes", []):
                box = DCCombinerBox(
                    inverter_id=inv.id,
                    label=box_data["box_id"],
                    fuses_rating_a=box_data.get("fuses_rating_a"),
                    switch_type=box_data.get("switch_type"),
                    spd_voltage_v=box_data.get("spd_voltage_v"),
                    spd_current_ka=box_data.get("spd_current_ka"),
                    aggregated_v_mpp_v=box_data.get("aggregated_v_mpp_v"),
                    aggregated_i_mpp_a=box_data.get("aggregated_i_mpp_a"),
                    aggregated_p_mpp_kw=box_data.get("aggregated_p_mpp_kw"),
                    valid_from=commissioning,
                    valid_to=valid_to or SENTINEL_VALID_TO,
                )
                session.add(box)
                session.flush()

                # 4. Strings
                for s_data in box_data.get("strings", []):
                    modules = s_data.get("modules", {})
                    string_row = StringRow(
                        box_id=box.id,
                        label=s_data["string_id"],
                        module_count=modules["count"],
                        module_model_name=modules.get("model_name"),
                        module_p_wp=modules["peak_power_wp"],
                        module_vmp=modules.get("vmp"),
                        module_imp=modules.get("imp"),
                        module_voc=modules.get("voc"),
                        module_isc=modules.get("isc"),
                        module_temp_coeff_pmax=modules.get("temp_coeff_pmax"),
                        tilt=s_data.get("tilt"),
                        azimuth=s_data.get("azimuth"),
                        tracking_type=s_data.get("tracking_type", "Fixed"),
                        v_mpp_v=s_data.get("v_mpp_v"),
                        i_mpp_a=s_data.get("i_mpp_a"),
                        p_mpp_kw=s_data.get("p_mpp_kw"),
                        valid_from=commissioning,
                        valid_to=valid_to or SENTINEL_VALID_TO,
                    )
                    session.add(string_row)

        session.commit()
        logging.info(f"✔ Seeded database: {db_path}")
        logging.info(f"  Site  : {site.name}  (id={site.id})")
        logging.info(f"  Inverters inserted  : {len(data.get('inverters', []))}")
        total_boxes = sum(
            len(inv.get("input_dc_boxes", [])) for inv in data.get("inverters", [])
        )
        total_strings = sum(
            len(s)
            for inv in data.get("inverters", [])
            for box in inv.get("input_dc_boxes", [])
            for s in [box.get("strings", [])]
        )
        logging.info(f"  DC boxes inserted   : {total_boxes}")
        logging.info(f"  Strings inserted    : {total_strings}")
        return site.id


# ---------------------------------------------------------------------------
# Quick verification query
# ---------------------------------------------------------------------------
def verify(db_path: str | Path) -> None:
    """Log row counts for every table as a sanity check."""
    from sqlmodel import select

    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    with Session(engine) as session:
        sites = session.exec(select(Site)).all()
        inverters = session.exec(select(Inverter)).all()
        boxes = session.exec(select(DCCombinerBox)).all()
        strings = session.exec(select(StringRow)).all()

    logging.info("\n--- Database verification ---")
    logging.info(f"  sites              : {len(sites)}")
    logging.info(f"  inverters          : {len(inverters)}")
    logging.info(f"  dc_combiner_boxes  : {len(boxes)}")
    logging.info(f"  strings            : {len(strings)}")

    for s in sites:
        logging.info(f"\n  Site: {s.name}")
    for inv in inverters:
        logging.info(
            f"    Inverter {inv.label}  model={inv.model_name}  ac={inv.max_ac_kva} kVA"
        )
    for box in boxes:
        logging.info(f"      Box {box.label}  fuses={box.fuses_rating_a}A")
    for st in strings:
        logging.info(
            f"        String {st.label}  modules={st.module_count}x{st.module_p_wp}Wp"
            f"  v_mpp={st.v_mpp_v}V  p_mpp={st.p_mpp_kw}kW"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed the SQLite database from a parsed SLD JSON file."
    )
    parser.add_argument(
        "project",
        help=(
            "Project name (site_id) as registered in data/projects.json. "
            "Determines the JSON source directory and SQLite DB location."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Override: path to a specific JSON file (defaults to latest in db/<project>/json/).",
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="Override: path to the SQLite DB file (defaults to db/<project>/sqldb/<project>.sqlite).",
    )
    args = parser.parse_args()

    json_file, db_file = resolve_project_paths(
        args.project, args.json_path, args.db_path
    )

    seed_from_json(json_file, db_file)
    verify(db_file)

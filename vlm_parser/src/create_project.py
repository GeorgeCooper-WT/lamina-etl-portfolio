import sys
import json
import re
from pathlib import Path


def generate_site_id(display_name: str) -> str:
    site_id = re.sub(r"[^a-zA-Z0-9]+", "_", display_name.lower()).strip("_")
    return site_id


def create_project(display_name: str):
    site_id = generate_site_id(display_name)
    projects_file = Path("data/projects.json")
    projects = {}

    if projects_file.exists():
        with open(projects_file, "r", encoding="utf-8") as f:
            projects = json.load(f)

    if site_id in projects:
        print(f"Project '{site_id}' already exists.")
    else:
        projects[site_id] = {
            "site_id": site_id,
            "display_name": display_name,
        }
        projects_file.parent.mkdir(parents=True, exist_ok=True)
        with open(projects_file, "w", encoding="utf-8") as f:
            json.dump(projects, f, indent=2)
        print(f"Project '{site_id}' created.")

    # Create subfolders for this site
    base = Path("db") / site_id
    (base / "json").mkdir(parents=True, exist_ok=True)
    (base / "sqldb").mkdir(parents=True, exist_ok=True)
    (base / "input").mkdir(parents=True, exist_ok=True)  # <-- Add input folder creation
    print(f"Created folders: {base / 'json'}, {base / 'sqldb'}, and {base / 'input'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/create_project.py <display_name>")
        sys.exit(1)
    display_name = " ".join(sys.argv[1:])
    create_project(display_name)

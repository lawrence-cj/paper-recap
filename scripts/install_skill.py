#!/usr/bin/env python3
"""Register the repository-owned Paper Recap skill with Codex."""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / ".codex" / "skills" / "update-paper-recap"


def codex_skills_dir() -> Path:
    configured = os.environ.get("CODEX_HOME")
    base = Path(configured).expanduser() if configured else Path.home() / ".codex"
    return base / "skills"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="move an existing registration to a timestamped backup and replace it",
    )
    args = parser.parse_args()

    if not (SOURCE / "SKILL.md").exists():
        parser.error(f"skill source not found: {SOURCE}")

    skills_dir = codex_skills_dir()
    destination = skills_dir / SOURCE.name
    skills_dir.mkdir(parents=True, exist_ok=True)

    if destination.is_symlink() and destination.resolve() == SOURCE.resolve():
        print(f"already registered: {destination} -> {SOURCE}")
        return 0

    if destination.exists() or destination.is_symlink():
        if not args.replace:
            parser.error(f"destination exists: {destination}; rerun with --replace")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = destination.with_name(f"{destination.name}.backup-{timestamp}")
        shutil.move(str(destination), str(backup))
        print(f"previous registration moved to: {backup}")

    destination.symlink_to(SOURCE, target_is_directory=True)
    print(f"registered: {destination} -> {SOURCE}")
    print("future git pulls update the registered skill automatically")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
